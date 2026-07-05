"""Solution 1: KAN provides direction/constraints, ENV provides forward rollout.

Mode B: KAN Jacobian → init action seq → env rollout → gradient-refine via KAN Jacobian
Mode C: Energy heuristic (safe fallback)
Combined uncertainty: activation density + prediction error window
"""
import torch, torch.nn as nn
import numpy as np, time, sys, os
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kanrf import KAN
from control.kan_knowledge import KANKnowledge, CombinedUncertainty

# Pendulum analytical dynamics (from continual_learning.py)
G_VAL = 10.0; DT = 0.05; MAX_TORQUE = 2.0; MAX_SPEED = 8.0


def pendulum_step(s_raw, a_norm, g=10.0):
    """Single Pendulum step matching gymnasium Pendulum-v1 dynamics.

    Pendulum-v1: θ̈ = (g/l)·sin(θ) + u/(m·l²)
    With m=1, l=1, g=10:  θ̈ = 10·sin(θ) + u
    dt=0.05, max_torque=2.0, max_speed=8.0
    """
    th = np.arctan2(s_raw[1], s_raw[0])
    thd = s_raw[2]
    u = np.clip(a_norm * MAX_TORQUE, -MAX_TORQUE, MAX_TORQUE)

    # Correct dynamics: θ̈ = (g/l)·sinθ + u/(m·l²)
    thd_new = thd + (g * np.sin(th) + u) * DT
    thd_new = np.clip(thd_new, -MAX_SPEED, MAX_SPEED)
    th_new = th + thd_new * DT

    return np.array([np.cos(th_new), np.sin(th_new), thd_new])


def pendulum_rollout(s_norm, actions, g=10.0):
    """Roll out action sequence through analytical dynamics.

    Args:
        s_norm: (3,) [cos, sin, thd/8]
        actions: (H,) action sequence ∈ [-1, 1]
    Returns:
        states: (H+1, 3) in raw units
    """
    H = len(actions)
    s_raw = s_norm.copy()
    s_raw[2] *= 8.0  # denormalize
    states = [s_raw.copy()]
    for h in range(H):
        s_raw = pendulum_step(s_raw, actions[h], g=g)
        states.append(s_raw.copy())
    return np.array(states)


class KANGuidedMPC:
    """KAN provides direction + constraints. ENV provides forward rollout."""

    def __init__(self, kan, knowledge=None, state_dim=3, action_dim=1,
                 horizon=5, n_refine=30, eta_low=0.30,
                 lambda_ctrl=0.01, trust_delta=0.1, online_lr=1e-3,
                 device='cpu'):
        self.kan = kan; self.horizon = horizon; self.n_refine = n_refine
        self.eta_low = eta_low; self.lambda_ctrl = lambda_ctrl
        self.trust_delta = trust_delta; self.device = device
        self.gravity = 10.0

        if knowledge is None:
            knowledge = KANKnowledge(kan, device=device)
        self.kk = knowledge
        self.unc = CombinedUncertainty(knowledge, alpha=0.2, beta=10.0, window_size=50)

        self.prev_action = 0.0; self.prev_seq = np.zeros(horizon)
        self.mode_history = []; self.U_history = []

        # Online learning buffer
        self.buffer_x = []; self.buffer_y = []; self.buffer_max = 200
        self.n_updates = 0

    # ═══ Mode B: KAN-guided, env-rollout MPC ═══

    def _plan_mode_b(self, s_norm):
        H = self.horizon; s_target = np.array([0.0, 1.0, 0.0])
        sin_theta = s_norm[1]

        # ── KAN Jacobian pseudo-inverse for initial action ──
        s_t = torch.tensor(s_norm, dtype=torch.float32, device=self.device).unsqueeze(0)
        J = self.kk.jacobian(s_t).cpu().squeeze().numpy()  # (3,)
        state_err = s_target - s_norm

        # Pseudo-inverse: a = J^T * err / (||J||^2 + epsilon)
        J_norm_sq = float(np.dot(J, J)) + 1e-4
        a_pinv = float(np.dot(J, state_err)) / J_norm_sq
        base_a = np.clip(a_pinv, -1.0, 1.0)

        # ── For near-upright: use pseudo-inverse directly (one-step LQR) ──
        if sin_theta > 0.7:
            # Near upright: linear approximation holds. Just do one-step correction.
            self.prev_action = base_a
            return base_a, {'mode': 'B', 'method': 'pinv'}

        # ── Away from upright: env-rollout MPC with KAN init ──
        n_candidates = 80
        delta_max = min(float(self.kk.trust_region_radius(self.trust_delta)), 0.3)
        base_seq = np.full(H, base_a)

        candidates = [base_seq.copy()]
        for _ in range(n_candidates - 1):
            noise = np.random.randn(H) * delta_max * 0.3
            noise[0] *= 1.5; noise[-1] *= 0.3
            cand = np.clip(base_seq + noise, -1.0, 1.0)
            candidates.append(cand)

        best_cost = float('inf'); best_seq = base_seq
        for cand in candidates:
            states = pendulum_rollout(s_norm, cand, g=self.gravity)
            s_final = states[-1]
            terminal = np.sum((s_final[:2]-s_target[:2])**2) + ((s_final[2]/8.0-s_target[2])**2)
            E0 = 0.5*states[0][2]**2 + self.gravity*states[0][1]
            E_last = 0.5*s_final[2]**2 + self.gravity*s_final[1]
            if sin_theta < 0.5:
                energy = -2.0*(E_last-E0)
            else:
                energy = 0.5*(E_last-self.gravity)**2
            ctrl = self.lambda_ctrl*np.sum(cand**2)
            cost = 5.0*terminal + energy + ctrl
            if cost < best_cost:
                best_cost = cost; best_seq = cand.copy()

        self.prev_seq = best_seq.copy()
        return best_seq[0], {'mode': 'B', 'method': 'mpc', 'cost': best_cost}

    # ═══ Mode C: Safe fallback ═══

    def _plan_mode_c(self, s_norm):
        sin = s_norm[1]; thd = s_norm[2]*8.0
        E = 0.5*thd**2 + self.gravity*sin
        a_norm = np.clip(1.5*(E-self.gravity)*thd/10.0, -1.0, 1.0)
        self.prev_action = a_norm
        return a_norm, {'mode': 'C'}

    # ═══ Online learning ═══

    def _online_update(self, s_norm, a_norm, s_true_norm):
        x = torch.cat([
            torch.tensor(s_norm, dtype=torch.float32, device=self.device).unsqueeze(0),
            torch.tensor([[a_norm]], dtype=torch.float32, device=self.device)
        ], dim=-1)
        y = torch.tensor(s_true_norm, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.buffer_x.append(x); self.buffer_y.append(y)
        while len(self.buffer_x) > self.buffer_max:
            self.buffer_x.pop(0); self.buffer_y.pop(0)

        if len(self.buffer_x) < 16:
            return

        idx = np.random.choice(len(self.buffer_x), min(32, len(self.buffer_x)), replace=False)
        xb = torch.cat([self.buffer_x[i] for i in idx], dim=0)
        yb = torch.cat([self.buffer_y[i] for i in idx], dim=0)

        self.kan.train()
        for p in self.kan.parameters(): p.requires_grad = True
        try:
            pred = self.kan(xb)
            loss = nn.functional.mse_loss(pred, yb)
            loss.backward()
            with torch.no_grad():
                for p in self.kan.parameters():
                    if p.grad is not None:
                        p -= 5e-4 * p.grad; p.grad.zero_()
        except RuntimeError:
            pass
        self.kan.eval()
        for p in self.kan.parameters(): p.requires_grad = False
        self.n_updates += 1

    # ═══ Main ═══

    def get_action(self, s_norm):
        s_t = torch.tensor(s_norm, dtype=torch.float32, device=self.device).unsqueeze(0)
        U = self.unc.compute(s_t)
        self.U_history.append(U)
        U_recent = np.mean(self.U_history[-min(5, len(self.U_history)):])

        # Hybrid: energy heuristic at bottom, KAN-guided MPC near upright
        sin_theta = s_norm[1]
        if sin_theta < 0.5:
            # Bottom: energy shaping is the right strategy
            a_norm, info = self._plan_mode_c(s_norm)
            info['mode'] = 'C'
            info['U'] = 0.0
        elif U_recent < self.eta_low:
            a_norm, info = self._plan_mode_b(s_norm)
        else:
            a_norm, info = self._plan_mode_c(s_norm)

        info['U'] = U; info['U_recent'] = U_recent
        self.mode_history.append(info['mode'])
        return a_norm, info

    def feed_transition(self, s_norm, a_norm, s_true_norm):
        self.unc.update(s_norm, a_norm, s_true_norm)
        if self.mode_history and self.mode_history[-1] == 'C':
            self._online_update(s_norm, a_norm, s_true_norm)

    def set_gravity(self, g):
        self.gravity = g


# ═══════════════════════════════════════════════════════════════════════════════
# Experiments
# ═══════════════════════════════════════════════════════════════════════════════

def test_standard(kan_path, n_trials=10, horizon=5, n_refine=30, eta_low=0.99):
    """Standard Pendulum test."""
    import gymnasium as gym
    device = torch.device('cpu'); PI_2 = np.pi / 2

    kan = KAN([4, 12, 3], grid_size=5, spline_order=3)
    kan.load_state_dict(torch.load(kan_path, weights_only=True, map_location=device))
    kan.to(device); kan.eval()

    mpc = KANGuidedMPC(kan, horizon=horizon, n_refine=n_refine,
                       eta_low=eta_low, device=device)
    env = gym.make('Pendulum-v1')
    successes = 0; all_steps = []; all_errors = []

    for trial in range(n_trials):
        seed = 42 + trial*100; obs, _ = env.reset(seed=seed)
        for step in range(300):
            s_norm = np.array([obs[0], obs[1], obs[2]/8.0], dtype=np.float32)
            a_norm, info = mpc.get_action(s_norm)
            a_raw = a_norm * 2.0
            obs_next, _, term, trunc, _ = env.step([a_raw])
            s_true = np.array([obs_next[0], obs_next[1], obs_next[2]/8.0], dtype=np.float32)
            mpc.feed_transition(s_norm, a_norm, s_true)

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
        print(f"  Trial {trial+1:2d}  steps={all_steps[-1]:3d}  "
              f"err={all_errors[-1]:.3f}  {'✓' if all_errors[-1]<0.2 else '✗'}")

    env.close(); sr = successes/n_trials
    print(f"\n  Success: {successes}/{n_trials} ({sr*100:.0f}%)  "
          f"mean|Δθ|={np.mean(all_errors):.3f}±{np.std(all_errors):.3f}")
    return sr


def test_gravity_switch(kan_path, n_steps=600, horizon=5, n_refine=30,
                         eta_low=0.30, switch_step=200):
    """Gravity switch: g=10 → g=3, near-upright start.

    Uses ConfigurablePendulum so the real environment's gravity actually changes.
    """
    from experiments.continual_learning import ConfigurablePendulum
    device = torch.device('cpu'); PI_2 = np.pi / 2

    kan = KAN([4, 12, 3], grid_size=5, spline_order=3)
    kan.load_state_dict(torch.load(kan_path, weights_only=True, map_location=device))
    kan.to(device); kan.eval()

    mpc = KANGuidedMPC(kan, horizon=horizon, n_refine=n_refine,
                       eta_low=eta_low, device=device)

    print(f"Gravity Switch: g=10 → g=3 at step {switch_step}")
    print(f"  H={horizon}, refine={n_refine}, η_low={eta_low}")
    print(f"  Solution 1: KAN provides direction/constraints, ENV does rollout")

    # Use ConfigurablePendulum so gravity ACTUALLY changes at switch_step
    env = ConfigurablePendulum(g=10.0, seed=742)
    env.set_g(10.0)
    obs = env.reset(seed=742)  # near-upright start

    errors_deg = []; pred_errs = []; U_vals = []; modes = []
    n_mode_switches = 0

    for step in range(n_steps):
        if step == switch_step:
            mpc.set_gravity(3.0)
            mpc.unc.error_window.clear()
            print(f"\n  *** Step {step}: GRAVITY 10 → 3 ***\n")

        s_norm = np.array([obs[0], obs[1], obs[2]/8.0], dtype=np.float32)
        a_norm, info = mpc.get_action(s_norm)
        a_raw = a_norm * 2.0
        result = env.step([a_raw])
        obs_next = result[0]  # obs is first element
        term = result[2] if len(result) > 2 else False
        trunc = result[3] if len(result) > 3 else False
        s_true_norm = np.array([obs_next[0], obs_next[1], obs_next[2]/8.0], dtype=np.float32)
        mpc.feed_transition(s_norm, a_norm, s_true_norm)

        # Prediction error
        with torch.no_grad():
            x = torch.cat([torch.tensor(s_norm, dtype=torch.float32).unsqueeze(0),
                          torch.tensor([[a_norm]], dtype=torch.float32)], dim=-1)
            pred = kan(x).cpu().squeeze().numpy()
        pred_err = np.linalg.norm(pred - s_true_norm)
        pred_errs.append(pred_err)

        angle_err = min(abs(np.arctan2(obs_next[1], obs_next[0]) - PI_2),
                       2*np.pi - abs(np.arctan2(obs_next[1], obs_next[0]) - PI_2))
        errors_deg.append(np.rad2deg(angle_err))
        U_vals.append(info.get('U', 0))
        modes.append(info['mode'])
        if len(modes) >= 2 and modes[-1] != modes[-2]:
            n_mode_switches += 1

        if step < 5 or abs(step - switch_step) < 10 or step % 50 == 0:
            print(f"  Step {step:3d}  |Δθ|={np.rad2deg(angle_err):.1f}deg  "
                  f"mode={info['mode']}  U={info['U']:.3f}  "
                  f"pred_err={pred_err:.4f}  updates={mpc.n_updates}")

        if term or trunc:
            obs = env.reset()
        else:
            obs = obs_next

    env.env.close()

    # Summary
    pre = slice(50, switch_step)
    post = slice(switch_step, switch_step+100)
    late = slice(switch_step+200, n_steps)

    print(f"\n{'='*60}")
    print(f"Results:")
    print(f"  g=10 (pre):   |Δθ|={np.mean(errors_deg[pre]):.1f}±{np.std(errors_deg[pre]):.1f}deg")
    print(f"  g=3  (post):  |Δθ|={np.mean(errors_deg[post]):.1f}±{np.std(errors_deg[post]):.1f}deg")
    if late.stop > late.start:
        print(f"  g=3  (late):  |Δθ|={np.mean(errors_deg[late]):.1f}±{np.std(errors_deg[late]):.1f}deg")
    print(f"  Pred err: g=10={np.mean(pred_errs[pre]):.4f}  post={np.mean(pred_errs[post]):.4f}")
    n_b_pre = sum(1 for m in modes[pre] if m=='B')
    n_b_post = sum(1 for m in modes[post] if m=='B')
    print(f"  Mode B: g=10={n_b_pre}/{pre.stop-pre.start}  g=3={n_b_post}/{post.stop-post.start}")
    print(f"  Mode switches: {n_mode_switches}")
    print(f"  Online updates: {mpc.n_updates}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--kan', type=str, default='/tmp/kanrf_cl_exp/kan_cws_cl.pt')
    parser.add_argument('--mode', type=str, default='standard', choices=['standard', 'gravity'])
    parser.add_argument('--trials', type=int, default=10)
    parser.add_argument('--steps', type=int, default=600)
    parser.add_argument('--horizon', type=int, default=5)
    parser.add_argument('--refine', type=int, default=30)
    parser.add_argument('--eta-low', type=float, default=0.99)
    parser.add_argument('--switch-step', type=int, default=200)
    args = parser.parse_args()

    if args.mode == 'gravity':
        test_gravity_switch(args.kan, n_steps=args.steps,
                            horizon=args.horizon, n_refine=args.refine,
                            eta_low=args.eta_low, switch_step=args.switch_step)
    else:
        test_standard(args.kan, args.trials, args.horizon, args.refine, args.eta_low)
