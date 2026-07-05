"""KAN-Enhanced MPC with combined uncertainty + online learning.

Architecture:
  Mode B: KAN-initialized rollout (Jacobian init + trust region + uncertainty penalty)
  Mode C: Energy-shaping safe fallback
  Online: When Mode C, fine-tune KAN on recent transitions using sliding window

Combined uncertainty: activation density + prediction error std.
"""
import torch
import torch.nn as nn
import numpy as np
import time, sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kanrf import KAN
from control.kan_knowledge import KANKnowledge, CombinedUncertainty


class KANEnhancedMPC:
    def __init__(self, kan, knowledge=None,
                 state_dim=3, action_dim=1, s_target=None,
                 horizon=5, n_refine_steps=20,
                 eta_low=0.20, lambda_ctrl=0.01, trust_delta=0.1,
                 online_lr=1e-3, online_buffer=200, device='cpu'):
        self.kan = kan; self.state_dim = state_dim; self.action_dim = action_dim
        self.horizon = horizon; self.n_refine_steps = n_refine_steps
        self.eta_low = eta_low; self.lambda_ctrl = lambda_ctrl
        self.trust_delta = trust_delta; self.device = device
        self.online_lr = online_lr

        if knowledge is None:
            knowledge = KANKnowledge(kan, device=device)
        self.kk = knowledge

        # Combined uncertainty
        self.unc = CombinedUncertainty(knowledge, alpha=0.3, beta=10.0, window_size=50)

        if s_target is None:
            self.s_target = torch.tensor([[0.0, 1.0, 0.0]], device=device)
        else:
            self.s_target = s_target.to(device)

        self.prev_action = torch.zeros(1, action_dim, device=device)
        self.prev_seq = torch.zeros(horizon, action_dim, device=device)

        # Online learning: replay buffer
        self.buffer_x = []; self.buffer_y = []
        self.buffer_max = online_buffer

        # Tracking
        self.mode_history = []; self.U_history = []; self.err_history = []
        self.n_online_updates = 0

    # ═══ Mode B: KAN-enhanced MPC ═══
    def _plan_mode_b(self, s_norm):
        H = self.horizon; s_target = self.s_target.squeeze(0)

        # Jacobian initialization
        J = self.kk.jacobian(s_norm)
        state_err = (s_target - s_norm.squeeze(0)).unsqueeze(1)
        grad_direction = (J.t() @ state_err).squeeze()
        init_action = torch.clamp(0.5 * grad_direction, -1.0, 1.0)
        if init_action.dim() == 0: init_action = init_action.unsqueeze(0)

        actions = torch.zeros(H, self.action_dim, device=self.device)
        actions[0] = init_action
        for h in range(1, H): actions[h] = 0.7 * actions[h-1]

        delta_max = min(self.kk.trust_region_radius(self.trust_delta), 0.5)
        actions.requires_grad_(True)
        opt = torch.optim.Adam([actions], lr=0.05)

        for _ in range(self.n_refine_steps):
            opt.zero_grad()
            s = s_norm; losses = []
            for h in range(H):
                a = actions[h:h+1]; x = torch.cat([s, a], dim=-1)
                s = self.kan(x)
                E = 0.5*(s[0,2]*8.0).pow(2) + 10.0*s[0,1]
                losses.append(torch.where(s[0,1] < 0.5, -0.1*E, (E-10.0).pow(2)))
                losses.append(self.lambda_ctrl * a.pow(2).sum())
            torch.stack(losses).sum().backward()
            opt.step()
            with torch.no_grad():
                actions.clamp_(-1.0, 1.0)
                for h in range(H):
                    d = actions[h] - self.prev_seq[h]
                    if d.abs() > delta_max: actions[h] = self.prev_seq[h] + d.sign()*delta_max

        best_action = actions[0].detach().clone()
        self.prev_seq = actions.detach().clone()
        self.prev_action = best_action.unsqueeze(0)
        return best_action.item(), {'mode': 'B'}

    # ═══ Mode C: Safe fallback ═══
    @torch.no_grad()
    def _plan_mode_c(self, s_norm):
        sin = s_norm[0,1].item(); thd = s_norm[0,2].item()*8.0
        E = 0.5*thd**2 + 10.0*sin
        a_norm = np.clip(1.5*(E-10.0)*thd/10.0, -1.0, 1.0)
        self.prev_action = torch.tensor([[a_norm]], device=self.device)
        return a_norm, {'mode': 'C'}

    # ═══ Online KAN fine-tuning ═══
    def _online_update(self, s_norm, a_norm, s_true_norm):
        """Fine-tune KAN on recent transitions using replay buffer."""
        # Add to buffer
        x = torch.cat([s_norm, a_norm.unsqueeze(0).to(self.device)
                       if isinstance(a_norm, torch.Tensor) else
                       torch.tensor([[a_norm]], dtype=torch.float32, device=self.device)], dim=-1)
        self.buffer_x.append(x.detach().clone())
        self.buffer_y.append(s_true_norm.detach().clone())
        while len(self.buffer_x) > self.buffer_max:
            self.buffer_x.pop(0); self.buffer_y.pop(0)

        if len(self.buffer_x) < 16:
            return  # wait for enough data

        # Mini-batch update
        idx = np.random.choice(len(self.buffer_x), min(32, len(self.buffer_x)), replace=False)
        xb = torch.cat([self.buffer_x[i] for i in idx], dim=0)
        yb = torch.cat([self.buffer_y[i] for i in idx], dim=0)

        # Ensure KAN is in training mode and requires grad
        self.kan.train()
        for p in self.kan.parameters():
            p.requires_grad = True
        try:
            pred = self.kan(xb)
            loss = nn.functional.mse_loss(pred, yb)
            loss.backward()
            with torch.no_grad():
                for p in self.kan.parameters():
                    if p.grad is not None:
                        p -= self.online_lr * p.grad
                        p.grad.zero_()
        except RuntimeError:
            pass  # skip if grad computation fails
        self.kan.eval()
        for p in self.kan.parameters():
            p.requires_grad = False
        self.n_online_updates += 1

    # ═══ Main get_action ═══
    def get_action(self, s_norm):
        if isinstance(s_norm, np.ndarray):
            s_norm = torch.tensor(s_norm, dtype=torch.float32, device=self.device)
        if s_norm.dim() == 1:
            s_norm = s_norm.unsqueeze(0)

        # Combined uncertainty
        U = self.unc.compute(s_norm)
        self.U_history.append(U)
        window = min(5, len(self.U_history))
        U_recent = np.mean(self.U_history[-window:])

        if U_recent < self.eta_low:
            a_norm, info = self._plan_mode_b(s_norm)
        else:
            with torch.no_grad():
                a_norm, info = self._plan_mode_c(s_norm)
                info['U'] = U; info['U_recent'] = U_recent
        self.mode_history.append(info['mode'])
        return a_norm, info

    def feed_transition(self, s_norm, a_norm, s_true_norm):
        """Feed a real transition for uncertainty tracking + online learning."""
        self.unc.update(s_norm, a_norm, s_true_norm)
        # Online learning: only in Mode C (high uncertainty)
        if self.mode_history and self.mode_history[-1] == 'C':
            if isinstance(s_norm, np.ndarray):
                s_norm = torch.tensor(s_norm, dtype=torch.float32, device=self.device)
            if isinstance(s_true_norm, np.ndarray):
                s_true_norm = torch.tensor(s_true_norm, dtype=torch.float32, device=self.device)
            if s_norm.dim() == 1:
                s_norm = s_norm.unsqueeze(0)
            if s_true_norm.dim() == 1:
                s_true_norm = s_true_norm.unsqueeze(0)
            self._online_update(s_norm, a_norm, s_true_norm)


# ═══════════════════════════════════════════════════════════════════════════════
# Gravity Switch Experiment
# ═══════════════════════════════════════════════════════════════════════════════

def test_gravity_switch(kan_path, n_steps=400, horizon=5, n_refine=20,
                         eta_low=0.20, switch_step=100, online_lr=1e-3):
    """Gravity switch experiment: g=10 → g=3 at switch_step.

    Tests the full closed loop: detect (U spike) → degrade (Mode C)
    → learn (online fine-tune) → recover (U drop → back to Mode B).
    """
    import gymnasium as gym
    from experiments.continual_learning import ConfigurablePendulum, EnergyController

    device = torch.device('cpu')
    PI_2 = np.pi / 2

    kan = KAN([4, 12, 3], grid_size=5, spline_order=3)
    kan.load_state_dict(torch.load(kan_path, weights_only=True, map_location=device))
    kan.to(device); kan.eval()

    mpc = KANEnhancedMPC(kan, state_dim=3, action_dim=1,
                         horizon=horizon, n_refine_steps=n_refine,
                         eta_low=eta_low, online_lr=online_lr, device=device)

    print(f"Gravity Switch Experiment: g=10 (0-{switch_step}), g=3 ({switch_step}-{n_steps})")
    print(f"  H={horizon}, refine={n_refine}, η_low={eta_low}, lr={online_lr}")
    print(f"  Lipschitz L={mpc.kk.lipschitz:.4f}")

    env = ConfigurablePendulum(g=10.0, seed=42)
    ctrl = EnergyController()
    obs = env.reset(seed=42)

    errors = []; modes = []; U_vals = []; pred_errors = []
    current_g = 10.0
    mode_switches = []

    for step in range(n_steps):
        # Gravity switch
        if step == switch_step:
            env.set_g(3.0); ctrl.g = 3.0; current_g = 3.0
            print(f"\n  *** Step {step}: GRAVITY 10 → 3 ***\n")

        s_norm = np.array([obs[0], obs[1], obs[2]/8.0], dtype=np.float32)
        a_norm, info = mpc.get_action(s_norm)
        a_raw = a_norm * 2.0

        # Execute
        obs_next, _, term, trunc = env.step([a_raw])

        # Feed transition for uncertainty tracking + online learning
        s_true_norm = np.array([obs_next[0], obs_next[1], obs_next[2]/8.0], dtype=np.float32)
        mpc.feed_transition(s_norm, a_norm, s_true_norm)

        # Compute prediction error
        with torch.no_grad():
            x = torch.cat([torch.tensor(s_norm, dtype=torch.float32).unsqueeze(0),
                          torch.tensor([[a_norm]], dtype=torch.float32)], dim=-1).to(device)
            pred = kan(x).cpu().squeeze().numpy()
        pred_err = np.linalg.norm(pred - s_true_norm)
        pred_errors.append(pred_err)

        errors.append(min(abs(np.arctan2(obs_next[1], obs_next[0]) - PI_2),
                          2*np.pi - abs(np.arctan2(obs_next[1], obs_next[0]) - PI_2)))
        modes.append(info['mode'])
        U_vals.append(info.get('U', 0))

        # Detect mode switches
        if len(modes) >= 2 and modes[-1] != modes[-2]:
            mode_switches.append((step, modes[-2], modes[-1]))

        # Periodic report
        if (step+1) % 50 == 0 or step < 5 or (switch_step <= step < switch_step+5):
            mode_b_frac = sum(1 for m in modes[-50:] if m == 'B') / min(50, len(modes)) * 100
            U_avg = np.mean(U_vals[-50:]) if len(U_vals) >= 50 else np.mean(U_vals)
            print(f"  Step {step+1:3d}  |Δθ|={errors[-1]:.3f}  mode={info['mode']}  "
                  f"U={U_vals[-1]:.3f}  pred_err={pred_err:.4f}  "
                  f"B%={mode_b_frac:.0f}%  updates={mpc.n_online_updates}")

        if term or trunc:
            obs = env.reset()
        else:
            obs = obs_next

    env.env.close()

    # ── Results ──
    print(f"\n{'='*60}")
    print(f"Gravity Switch Results")
    print(f"{'='*60}")

    # Pre-switch
    pre = slice(20, switch_step)
    post = slice(switch_step+20, n_steps)
    late = slice(switch_step+100, n_steps)

    print(f"\n  Angle error (rad):")
    print(f"    g=10 (pre-switch):  {np.mean(errors[pre]):.3f} ± {np.std(errors[pre]):.3f}")
    if n_steps > switch_step + 20:
        print(f"    g=3  (post-switch): {np.mean(errors[post]):.3f} ± {np.std(errors[post]):.3f}")
        print(f"    g=3  (late, >100st): {np.mean(errors[late]):.3f} ± {np.std(errors[late]):.3f}")

    print(f"\n  Prediction error:")
    print(f"    g=10: {np.mean(pred_errors[pre]):.4f}")
    if n_steps > switch_step + 20:
        print(f"    g=3 (post):  {np.mean(pred_errors[post]):.4f}")
        print(f"    g=3 (late):  {np.mean(pred_errors[late]):.4f}")

    print(f"\n  Mode distribution:")
    n_b_pre = sum(1 for m in modes[pre] if m == 'B')
    n_c_pre = sum(1 for m in modes[pre] if m == 'C')
    print(f"    g=10: B={n_b_pre}/{pre.stop-pre.start} ({n_b_pre/(pre.stop-pre.start)*100:.0f}%)")
    if n_steps > switch_step + 20:
        n_b_post = sum(1 for m in modes[post] if m == 'B')
        n_c_post = sum(1 for m in modes[post] if m == 'C')
        n_b_late = sum(1 for m in modes[late] if m == 'B')
        n_c_late = sum(1 for m in modes[late] if m == 'C')
        print(f"    g=3 (post): B={n_b_post}/{post.stop-post.start} ({n_b_post/(post.stop-post.start)*100:.0f}%)")
        print(f"    g=3 (late): B={n_b_late}/{late.stop-late.start} ({n_b_late/(late.stop-late.start)*100:.0f}%)")

    print(f"\n  Mode switches detected: {len(mode_switches)}")
    for s, frm, to in mode_switches[:10]:
        print(f"    Step {s}: {frm} → {to}")

    print(f"\n  Online KAN updates: {mpc.n_online_updates}")

    return {
        'errors': errors, 'modes': modes, 'U_vals': U_vals,
        'pred_errors': pred_errors, 'mode_switches': mode_switches,
        'online_updates': mpc.n_online_updates,
    }


def test_enhanced_mpc(kan_path, n_trials=10, horizon=5, n_refine=20,
                       eta_low=0.20):
    """Standard Pendulum test (no gravity switch)."""
    import gymnasium as gym

    device = torch.device('cpu'); PI_2 = np.pi / 2
    kan = KAN([4, 12, 3], grid_size=5, spline_order=3)
    kan.load_state_dict(torch.load(kan_path, weights_only=True, map_location=device))
    kan.to(device); kan.eval()

    mpc = KANEnhancedMPC(kan, state_dim=3, action_dim=1,
                         horizon=horizon, n_refine_steps=n_refine,
                         eta_low=eta_low, device=device)
    env = gym.make('Pendulum-v1')
    successes = 0; all_steps = []; all_errors = []
    mode_b = 0; mode_c = 0

    for trial in range(n_trials):
        seed = 42 + trial*100; obs, _ = env.reset(seed=seed)
        for step in range(300):
            s_norm = np.array([obs[0], obs[1], obs[2]/8.0], dtype=np.float32)
            a_norm, info = mpc.get_action(s_norm)
            a_raw = a_norm * 2.0
            obs_next, _, term, trunc, _ = env.step([a_raw])

            # Feed transition
            s_true = np.array([obs_next[0], obs_next[1], obs_next[2]/8.0], dtype=np.float32)
            mpc.feed_transition(s_norm, a_norm, s_true)

            if info['mode'] == 'B': mode_b += 1
            else: mode_c += 1

            err = min(abs(np.arctan2(obs_next[1], obs_next[0]) - PI_2),
                     2*np.pi - abs(np.arctan2(obs_next[1], obs_next[0]) - PI_2))
            if err < 0.2:
                successes += 1; all_steps.append(step+1); all_errors.append(err); break
            obs = obs_next
        else:
            all_steps.append(300)
            err = min(abs(np.arctan2(obs[1], obs[0]) - PI_2),
                     2*np.pi - abs(np.arctan2(obs[1], obs[0]) - PI_2))
            all_errors.append(err)
        status = '✓' if all_errors[-1] < 0.2 else '✗'
        print(f"  Trial {trial+1:2d}  steps={all_steps[-1]:3d}  "
              f"err={all_errors[-1]:.3f}  {status}  B%={mode_b/max(1,mode_b+mode_c)*100:.0f}%")

    env.close()
    sr = successes/n_trials
    print(f"\n  Success: {successes}/{n_trials} ({sr*100:.0f}%)  "
          f"mean|Δθ|={np.mean(all_errors):.3f}±{np.std(all_errors):.3f}  "
          f"ModeB={mode_b}/{mode_b+mode_c} ({mode_b/(mode_b+mode_c)*100:.0f}%)")
    return sr


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--kan', type=str, default='/tmp/kanrf_cl_exp/kan_cws_cl.pt')
    parser.add_argument('--mode', type=str, default='test', choices=['test', 'gravity'])
    parser.add_argument('--trials', type=int, default=10)
    parser.add_argument('--steps', type=int, default=400)
    parser.add_argument('--horizon', type=int, default=5)
    parser.add_argument('--refine', type=int, default=20)
    parser.add_argument('--eta-low', type=float, default=0.20)
    parser.add_argument('--switch-step', type=int, default=150)
    parser.add_argument('--online-lr', type=float, default=1e-3)
    args = parser.parse_args()

    if args.mode == 'gravity':
        test_gravity_switch(args.kan, n_steps=args.steps,
                            horizon=args.horizon, n_refine=args.refine,
                            eta_low=args.eta_low, switch_step=args.switch_step,
                            online_lr=args.online_lr)
    else:
        test_enhanced_mpc(args.kan, args.trials, args.horizon,
                          args.refine, args.eta_low)
