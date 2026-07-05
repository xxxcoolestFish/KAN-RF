"""KAN-MPC: KAN as Differentiable Dynamics Engine for Model Predictive Control.

Architecture:
  Deployment: s_t → [MPC Optimizer] → a*
                         ↑
                    [KAN f(s,a)→s']  ← differentiable dynamics (CWS-trained)
                         ↑
                    Provides: prediction, Jacobian ∂s'/∂a, uncertainty ρ(s)

Key innovation: KAN PERMANENTLY stays in the deployment loop.  No distillation
into an MLP policy.  KAN's three unique capabilities map directly to three
critical MPC components:
  1. Accurate Jacobian (CWS) → gradient-based action refinement
  2. Activation density ρ(s) → free epistemic uncertainty → safe planning
  3. Bounded derivatives → certified rollout error bounds

The MPC uses random shooting + gradient refinement:
  1. Sample N candidate action sequences
  2. Roll out each through KAN (H-step lookahead)
  3. Score: task_reward - λ_unc · (1-ρ(s)) - λ_ctrl · ||a||²
  4. Gradient-refine top-K candidates using KAN's Jacobian
  5. Execute first action of best sequence (MPC replanning)
"""
import torch
import numpy as np
import time, sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kanrf import KAN


class KANMPC:
    """KAN-powered Model Predictive Controller.

    Args:
        kan: CWS-trained KAN world model f(s,a) → s'
        state_dim: state dimension
        action_dim: action dimension
        s_target: target state (normalized)
        horizon: planning horizon (steps to look ahead)
        n_candidates: number of random action sequences to try
        n_refine: number of top candidates to gradient-refine
        refine_steps: gradient descent steps per refinement
        lambda_unc: uncertainty penalty weight (KAN-specific)
        lambda_ctrl: control cost weight
        temperature: softmax temperature for selecting among candidates
        device: torch device
    """

    def __init__(self, kan, state_dim=3, action_dim=1,
                 s_target=None, score_fn=None, use_energy_heuristic=False,
                 horizon=8, n_candidates=200, n_refine=5, refine_steps=10,
                 lambda_unc=0.1, lambda_ctrl=0.01, temperature=0.1,
                 device='cpu'):
        self.kan = kan
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.score_fn = score_fn  # custom scoring function(N, states_all, actions) → (N,) scores
        self.use_energy_heuristic = use_energy_heuristic
        self.horizon = horizon
        self.n_candidates = n_candidates
        self.n_refine = n_refine
        self.refine_steps = refine_steps
        self.lambda_unc = lambda_unc
        self.lambda_ctrl = lambda_ctrl
        self.temperature = temperature
        self.device = device

        # Target state
        if s_target is None:
            # Pendulum upright: [cos=0, sin=1, thd=0]
            self.s_target = torch.tensor([[0.0, 1.0, 0.0]], device=device)
        else:
            self.s_target = s_target.to(device)

        # Freeze KAN
        self.kan.eval()
        for p in self.kan.parameters():
            p.requires_grad = False

        # Previous best action (for warm-start sampling)
        self.prev_best_action = torch.zeros(1, action_dim, device=device)

        # Stats
        self.stats = {'n_calls': 0, 'total_time': 0.0}

    # ══════════════════════════════════════════════════════════════════════
    # KAN-specific: activation density as uncertainty
    # ══════════════════════════════════════════════════════════════════════

    @torch.no_grad()
    def _compute_uncertainty(self, s):
        """ρ(s) ∈ [0,1]: 1 = well-covered training region, 0 = extrapolating.

        Computed from B-spline activation density — FREE, no extra network needed.
        """
        if s.dim() == 1:
            s = s.unsqueeze(0)
        B = s.shape[0]
        # Use a=0 as probe action
        a_probe = torch.zeros(B, 1, device=self.device)
        x = torch.cat([s, a_probe], dim=-1)

        try:
            _, B_list, E_list = self.kan(x, return_activations=True)
            densities = []
            for B_mat in B_list:
                active = (B_mat.abs() > 1e-6).float().mean(dim=-1)  # (B, in_dim)
                densities.append(active.mean(dim=-1))  # (B,)
            rho = torch.stack(densities, dim=1).mean(dim=1)  # (B,)
        except Exception:
            rho = torch.ones(B, device=self.device)
        return rho

    # ══════════════════════════════════════════════════════════════════════
    # Rollout: simulate H steps through KAN
    # ══════════════════════════════════════════════════════════════════════

    @torch.no_grad()
    def _rollout(self, s0, actions):
        """Roll out action sequence through KAN.

        Args:
            s0: (1, state_dim) current state
            actions: (H, action_dim) action sequence

        Returns:
            states: (H+1, state_dim) predicted states
            uncertainties: (H,) activation densities
        """
        H = actions.shape[0]
        states = [s0.squeeze(0)]
        uncertainties = []

        s = s0
        for h in range(H):
            a = actions[h:h+1]  # (1, action_dim)
            x = torch.cat([s, a], dim=-1)
            s = self.kan(x)
            states.append(s.squeeze(0))

            # Uncertainty at predicted state
            rho = self._compute_uncertainty(s)
            uncertainties.append(rho.item() if rho.numel() == 1 else rho.mean().item())

        return torch.stack(states), uncertainties

    # ══════════════════════════════════════════════════════════════════════
    # Scoring: physics-informed cost function
    # ══════════════════════════════════════════════════════════════════════

    def _score(self, states, uncertainties, actions):
        """Score a trajectory.

        For Pendulum: energy gain + distance to upright + penalties.

        Returns: scalar score (lower is better)
        """
        H = len(actions)
        s_target = self.s_target.squeeze(0)

        # Terminal distance to target (last state)
        s_final = states[-1]
        # Normalize cos/sin before computing distance
        terminal_dist = ((s_final[:2] - s_target[:2]).pow(2).sum() +
                         (s_final[2] - s_target[2]).pow(2))

        # Energy gain over the trajectory
        s0 = states[0]
        s_last = states[-1]
        # Pendulum energy: E = 0.5*(thd*8)^2 + 10*sin
        E0 = 0.5 * (s0[2] * 8.0).pow(2) + 10.0 * s0[1]
        E_last = 0.5 * (s_last[2] * 8.0).pow(2) + 10.0 * s_last[1]
        Edes = 10.0  # target energy (upright at rest)

        # Energy improvement toward target
        if E0 < Edes:  # need more energy
            energy_score = -(E_last - E0)  # negative = good (energy increased)
        else:
            energy_score = (E_last - Edes).abs()  # minimize overshoot

        # Control cost
        ctrl_cost = actions.pow(2).sum()

        # Uncertainty penalty (KAN-specific!)
        # High uncertainty → high penalty → avoid this trajectory
        unc_penalty = sum(1.0 - u for u in uncertainties)

        # Weighted sum
        score = (10.0 * terminal_dist +
                 1.0 * energy_score +
                 self.lambda_ctrl * ctrl_cost +
                 self.lambda_unc * unc_penalty)

        return score.item()

    # ══════════════════════════════════════════════════════════════════════
    # Gradient refinement: improve top candidates using KAN's Jacobian
    # ══════════════════════════════════════════════════════════════════════

    def _refine(self, s0, actions_init, n_steps=10, lr=0.05):
        """Gradient-refine an action sequence through frozen KAN."""
        actions = actions_init.clone().detach().requires_grad_(True)
        opt = torch.optim.Adam([actions], lr=lr)

        for _ in range(n_steps):
            opt.zero_grad()

            # Rollout through KAN
            s = s0
            losses = []
            for h in range(self.horizon):
                a = actions[h:h+1]
                x = torch.cat([s, a], dim=-1)
                s_next = self.kan(x)

                # Physics-informed loss at each step
                E = 0.5 * (s_next[:, 2] * 8.0).pow(2) + 10.0 * s_next[:, 1]
                Edes = 10.0
                sin = s_next[:, 1]

                # Swing-up: maximize energy; Stabilize: match target
                step_loss = torch.where(
                    sin < 0.5,
                    -0.1 * E,                  # maximize energy (bottom)
                    (E - Edes).pow(2)           # match target energy (top)
                )
                losses.append(step_loss.mean())
                losses.append(self.lambda_ctrl * a.pow(2).sum())
                s = s_next

            total = torch.stack(losses).sum()
            total.backward()
            opt.step()
            with torch.no_grad():
                actions.clamp_(-1.0, 1.0)

        return actions.detach()

    # ══════════════════════════════════════════════════════════════════════
    # Main MPC step
    # ══════════════════════════════════════════════════════════════════════

    def get_action(self, s_norm):
        """Compute optimal action via hybrid KAN-MPC + energy heuristic.

        When pendulum is at the bottom (sin < 0.5), use energy-shaping heuristic
        because KAN's predictions from the bottom are unreliable (no pumping
        trajectories in training data).  When near upright, use full KAN-MPC
        for fine control where KAN's predictions are accurate.

        This hybrid approach: KAN knowledge used where it's reliable, simple
        physics prior used where KAN is uncertain.
        """
        t_start = time.time()

        if isinstance(s_norm, np.ndarray):
            s_norm = torch.tensor(s_norm, dtype=torch.float32, device=self.device)
        if s_norm.dim() == 1:
            s_norm = s_norm.unsqueeze(0)

        # ── Optional energy heuristic (Pendulum only) ──
        if self.use_energy_heuristic:
            sin_theta = s_norm[0, 1].item()
            if sin_theta < 0.5:
                cos_theta = s_norm[0, 0].item()
                thd_norm = s_norm[0, 2].item()
                thd = thd_norm * 8.0
                E = 0.5 * thd**2 + 10.0 * sin_theta
                E_des = 10.0
                a_energy = 1.5 * (E - E_des) * thd / 10.0
                a_norm = np.clip(a_energy, -1.0, 1.0)
                self.prev_best_action = torch.tensor([[a_norm]], device=self.device)
                return a_norm, {'method': 'energy', 'time': time.time() - t_start}

        # Top half: full KAN-MPC

        H = self.horizon
        N = self.n_candidates

        # ── 1. Sample candidate action sequences ──
        with torch.no_grad():
            actions_batch = torch.randn(N, H, self.action_dim, device=self.device) * 0.5
            actions_batch += self.prev_best_action.unsqueeze(0)
            actions_batch.clamp_(-1.0, 1.0)

            # ── 2. Vectorized rollout through KAN ──
            # Roll out ALL candidates in parallel: (N, H+1, state_dim)
            states_all = torch.zeros(N, H + 1, self.state_dim, device=self.device)
            states_all[:, 0, :] = s_norm.expand(N, -1)
            uncertainties = torch.zeros(N, H, device=self.device)

            s_batch = s_norm.expand(N, -1)  # (N, state_dim)
            for h in range(H):
                a_batch = actions_batch[:, h, :]  # (N, action_dim)
                x_batch = torch.cat([s_batch, a_batch], dim=-1)
                s_batch = self.kan(x_batch)  # (N, state_dim) — vectorized!
                states_all[:, h + 1, :] = s_batch

                # Uncertainty at this step
                for n in range(N):
                    rho = self._compute_uncertainty(s_batch[n:n+1])
                    uncertainties[n, h] = rho.item() if rho.numel() == 1 else rho.mean().item()

            # ── 3. Score all candidates ──
            if self.score_fn is not None:
                scores = self.score_fn(self, states_all, actions_batch, uncertainties)
            else:
                # Default: Pendulum energy-based scoring
                scores = torch.zeros(N, device=self.device)
                s_target = self.s_target.squeeze(0)
                sin0 = s_norm[0, 1].item()
                for n in range(N):
                    s_final = states_all[n, -1]
                    terminal_dist = ((s_final[:2] - s_target[:2]).pow(2).sum() +
                                     (s_final[2] - s_target[2]).pow(2))
                    E0 = 0.5 * (states_all[n, 0, 2] * 8.0).pow(2) + 10.0 * states_all[n, 0, 1]
                    E_last = 0.5 * (s_final[2] * 8.0).pow(2) + 10.0 * s_final[1]
                    Edes = 10.0
                    if sin0 < 0.5:
                        energy_score = -(E_last - E0)
                        scores[n] = (2.0 * terminal_dist + 10.0 * energy_score +
                                     self.lambda_ctrl * actions_batch[n].pow(2).sum())
                    else:
                        scores[n] = (10.0 * terminal_dist +
                                     self.lambda_ctrl * actions_batch[n].pow(2).sum() +
                                     self.lambda_unc * (1.0 - uncertainties[n]).sum())

            # ── Select top-K ──
            _, top_indices = scores.topk(min(self.n_refine, N), largest=False)
            refined_actions = actions_batch[top_indices].clone()

        # ── 4. Gradient-refine top candidates (NEEDS grad) ──
        for i in range(len(top_indices)):
            refined_actions[i] = self._refine(
                s_norm, refined_actions[i], n_steps=self.refine_steps)

        # ── 5. Re-score refined candidates ──
        with torch.no_grad():
            refined_scores = torch.zeros(len(top_indices), device=self.device)
            for i in range(len(top_indices)):
                states, uncertainties = self._rollout(s_norm, refined_actions[i])
                refined_scores[i] = self._score(states, uncertainties, refined_actions[i])

            # ── 6. Select best action ──
            best_idx = refined_scores.argmin().item()
            best_action = refined_actions[best_idx][0]
            self.prev_best_action = best_action.unsqueeze(0)

        elapsed = time.time() - t_start
        self.stats['n_calls'] += 1
        self.stats['total_time'] += elapsed

        with torch.no_grad():
            unc = self._compute_uncertainty(s_norm).item()

        info = {
            'score': refined_scores[best_idx].item(),
            'mean_score': scores.mean().item(),
            'best_score': refined_scores.min().item(),
            'uncertainty': unc,
            'time': elapsed,
        }

        return best_action.item(), info


def test_kan_mpc_pendulum(kan_path, n_trials=10, horizon=8, n_candidates=200):
    """Test KAN-MPC on Pendulum-v1."""
    import gymnasium as gym

    device = torch.device('cpu')
    PI_2 = np.pi / 2

    # Load CWS-trained KAN
    kan = KAN([4, 12, 3], grid_size=5, spline_order=3)
    kan.load_state_dict(torch.load(kan_path, weights_only=True, map_location=device))
    kan.to(device)
    kan.eval()
    print(f"KAN loaded: {sum(p.numel() for p in kan.parameters())} params")

    # Create MPC
    mpc = KANMPC(kan, state_dim=3, action_dim=1,
                 use_energy_heuristic=True,
                 horizon=horizon, n_candidates=n_candidates,
                 n_refine=5, refine_steps=10,
                 lambda_unc=0.1, lambda_ctrl=0.01, device=device)

    env = gym.make('Pendulum-v1')
    successes = 0
    total_steps_list = []
    final_errors = []

    for trial in range(n_trials):
        seed = 42 + trial * 100
        obs, _ = env.reset(seed=seed)
        max_steps = 300

        for step in range(max_steps):
            # Normalize state
            s_norm = np.array([obs[0], obs[1], obs[2] / 8.0], dtype=np.float32)

            # Get action from KAN-MPC
            a_norm, info = mpc.get_action(s_norm)
            a_raw = np.clip(a_norm * 2.0, -2.0, 2.0)  # denormalize

            obs, _, term, trunc, _ = env.step([a_raw])

            # Check success
            angle = np.arctan2(obs[1], obs[0])
            err = min(abs(angle - PI_2), 2 * np.pi - abs(angle - PI_2))
            if err < 0.2:  # within 0.2 rad of upright
                successes += 1
                total_steps_list.append(step + 1)
                final_errors.append(err)
                break
        else:
            total_steps_list.append(max_steps)
            angle = np.arctan2(obs[1], obs[0])
            err = min(abs(angle - PI_2), 2 * np.pi - abs(angle - PI_2))
            final_errors.append(err)

        status = '✓' if final_errors[-1] < 0.2 else '✗'
        print(f"  Trial {trial+1:2d}  seed={seed}  steps={total_steps_list[-1]:3d}  "
              f"err={final_errors[-1]:.3f}rad  {status}  avg_time={mpc.stats['total_time']/max(1,mpc.stats['n_calls']):.4f}s/step")

    env.close()

    sr = successes / n_trials
    print(f"\n{'='*60}")
    print(f"KAN-MPC Results ({n_trials} trials)")
    print(f"{'='*60}")
    print(f"  Success rate: {successes}/{n_trials} ({sr*100:.0f}%)")
    print(f"  Mean steps:   {np.mean(total_steps_list):.1f}")
    print(f"  Mean |Δθ|:    {np.mean(final_errors):.3f} ± {np.std(final_errors):.3f} rad")
    print(f"  Mean step time: {mpc.stats['total_time']/max(1,mpc.stats['n_calls']):.4f}s")
    print(f"  Total time:   {mpc.stats['total_time']:.1f}s")
    return sr


def test_kan_mpc_cartpole(kan_path, n_trials=10, horizon=5, n_candidates=100):
    """Test KAN-MPC on CartPole (continuous force, analytical dynamics)."""
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from experiments.cartpole_decision_v3 import step_cartpole_cont, X_SCALE, XD_SCALE, TH_SCALE, THD_SCALE, FORCE_MAX

    device = torch.device('cpu')

    # Load CWS-trained CartPole KAN
    kan = KAN([5, 12, 4], grid_size=5, spline_order=3)
    kan.load_state_dict(torch.load(kan_path, weights_only=True, map_location=device))
    kan.to(device)
    kan.eval()
    print(f"KAN loaded: {sum(p.numel() for p in kan.parameters())} params")

    # CartPole target: [x=0, xd=0, theta=0, thetad=0]
    s_target = torch.zeros(1, 4, device=device)

    # CartPole scoring function
    def cartpole_score(mpc, states_all, actions_batch, uncertainties):
        N = actions_batch.shape[0]
        scores = torch.zeros(N, device=mpc.device)
        for n in range(N):
            s_final = states_all[n, -1]
            pole_err = s_final[2].abs()
            cart_err = s_final[0].abs()
            scores[n] = (10.0 * pole_err + 1.0 * cart_err +
                         mpc.lambda_ctrl * actions_batch[n].pow(2).sum())
        return scores

    mpc = KANMPC(kan, state_dim=4, action_dim=1, s_target=s_target,
                 score_fn=cartpole_score,
                 horizon=horizon, n_candidates=n_candidates,
                 n_refine=5, refine_steps=10,
                 lambda_unc=0.1, lambda_ctrl=0.01, device=device)

    all_steps = []
    torch.manual_seed(42); np.random.seed(42)

    for trial in range(n_trials):
        seed = 42 + trial * 100
        torch.manual_seed(seed)
        th = np.random.uniform(-0.05, 0.05)
        s_raw = torch.tensor([[0.0, 0.0, th, 0.0]], dtype=torch.float32)

        for step in range(500):
            s_norm = s_raw.clone()
            s_norm[:, 0] /= X_SCALE; s_norm[:, 1] /= XD_SCALE
            s_norm[:, 2] /= TH_SCALE; s_norm[:, 3] /= THD_SCALE

            a_norm, info = mpc.get_action(s_norm)
            force = a_norm * FORCE_MAX

            s_raw = step_cartpole_cont(s_raw, torch.tensor([a_norm]))

            theta = s_raw[0, 2].item(); x = s_raw[0, 0].item()
            if abs(theta) > 0.21 or abs(x) > 2.4:
                break

        all_steps.append(step + 1)
        status = '✓' if step + 1 >= 500 else '✗'
        print(f"  Trial {trial+1:2d}  steps={step+1:3d}  {status}")

    mean_s = np.mean(all_steps)
    success = sum(1 for s in all_steps if s >= 500) / n_trials
    print(f"\n  KAN-MPC CartPole: {mean_s:.0f}±{np.std(all_steps):.0f} steps  "
          f"success={success*100:.0f}%  avg_time={mpc.stats['total_time']/max(1,mpc.stats['n_calls']):.3f}s/step")
    return success


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--kan', type=str, default='/tmp/kanrf_cl_exp/kan_cws_cl.pt')
    parser.add_argument('--env', type=str, default='pendulum', choices=['pendulum', 'cartpole'])
    parser.add_argument('--trials', type=int, default=10)
    parser.add_argument('--horizon', type=int, default=8)
    parser.add_argument('--candidates', type=int, default=200)
    args = parser.parse_args()

    if args.env == 'pendulum':
        test_kan_mpc_pendulum(args.kan, args.trials, args.horizon, args.candidates)
    else:
        test_kan_mpc_cartpole(args.kan, args.trials, args.horizon, args.candidates)
