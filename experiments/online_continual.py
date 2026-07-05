"""True Online Continual Learning — per-step updates for both WM and Policy.

After every environment step (s,a→s'):
  1. WM online update: local B-spline fine-tune on (s,a,s')
  2. Policy online update: one gradient step via updated WM

Key distinction: NO batch collection. Learning happens IN REAL TIME
as the system interacts with the environment. This is what KAN's
B-spline local support uniquely enables.

Metrics tracked per-step: prediction error, policy loss, angle error.
"""
import torch, torch.nn as nn
import numpy as np, time, sys, os
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kanrf import KAN
from control.kan_policy_net import KANPolicy
from experiments.continual_learning import ConfigurablePendulum

PI_2 = np.pi / 2
S_TARGET = torch.tensor([[0.0, 1.0, 0.0]])


class OnlineLearner:
    """Per-step online updates for both WM and Policy."""

    def __init__(self, wm, policy, lr_wm=1e-3, lr_policy=1e-3,
                 buffer_size=100, device='cpu'):
        self.wm = wm
        self.policy = policy
        self.lr_wm = lr_wm
        self.lr_policy = lr_policy
        self.device = device
        self.gravity = 10.0

        # Small replay buffer (for mini-batch, not batch collection)
        self.buffer_x = deque(maxlen=buffer_size)
        self.buffer_y = deque(maxlen=buffer_size)

        # Policy optimizer
        self.policy_opt = torch.optim.Adam(policy.parameters(), lr=lr_policy)

        # Counters
        self.n_wm_updates = 0
        self.n_policy_updates = 0

        # Tracking
        self.pred_error_window = deque(maxlen=100)
        self.policy_loss_window = deque(maxlen=100)

    def set_gravity(self, g):
        self.gravity = g

    def update(self, s_norm, a_norm, s_true_norm):
        """Online update: first WM, then Policy via WM gradient.

        Args:
            s_norm, a_norm, s_true_norm: numpy arrays (single transition)
        """
        s_t = torch.tensor(s_norm, dtype=torch.float32, device=self.device).unsqueeze(0)
        a_t = torch.tensor([[a_norm]], dtype=torch.float32, device=self.device)
        s_true_t = torch.tensor(s_true_norm, dtype=torch.float32, device=self.device).unsqueeze(0)
        x = torch.cat([s_t, a_t], dim=-1)

        # ── 1. WM online update (local B-spline fine-tune) ──
        self.wm.train()
        for p in self.wm.parameters():
            p.requires_grad = True

        pred = self.wm(x)
        wm_loss = nn.functional.mse_loss(pred, s_true_t)
        wm_loss.backward()

        with torch.no_grad():
            for p in self.wm.parameters():
                if p.grad is not None:
                    p -= self.lr_wm * p.grad
                    p.grad.zero_()

        self.wm.eval()
        for p in self.wm.parameters():
            p.requires_grad = False

        self.n_wm_updates += 1
        pred_err = wm_loss.item()
        self.pred_error_window.append(pred_err)

        # ── 2. Add to buffer and policy update ──
        self.buffer_x.append(x.detach().clone())
        self.buffer_y.append(s_true_t.detach().clone())

        # Policy update: mini-batch from buffer via WM gradient
        if len(self.buffer_x) >= 16:
            idx = np.random.choice(len(self.buffer_x), min(32, len(self.buffer_x)), replace=False)
            s_buf = torch.cat([self.buffer_x[i][:, :3] for i in idx], dim=0)
            a_buf = torch.cat([self.buffer_x[i][:, 3:] for i in idx], dim=0)

            self.policy.train()
            self.policy_opt.zero_grad()

            a_pred = self.policy(s_buf)
            wm_in = torch.cat([s_buf, a_pred], dim=-1)

            with torch.no_grad():
                # Note: using no_grad here because WM is frozen during policy update
                # But we need the WM gradient to flow through. Let's use the WM directly:
                pass

            # Use WM in forward mode for gradient computation
            s_pred = self.wm(wm_in)

            # Energy-guided loss
            g = self.gravity
            E_cur = 0.5 * (s_buf[:, 2] * 8) ** 2 + g * s_buf[:, 1]
            E_pred = 0.5 * (s_pred[:, 2] * 8) ** 2 + g * s_pred[:, 1]
            E_des = g
            energy_gain = (E_pred - E_cur) * torch.sign(E_des - E_cur)
            sin = s_buf[:, 1:2]
            w_swing = ((1.0 - sin) / 2.0).clamp(0, 1)
            w_stable = ((1.0 + sin) / 2.0).clamp(0, 1)

            energy_loss = -energy_gain.mean()
            dist_loss = (w_stable * (s_pred[:, :2] - S_TARGET[:, :2].to(self.device)).pow(2).sum(-1, keepdim=True)).mean()
            ctrl_loss = a_pred.pow(2).mean()
            policy_loss = energy_loss + dist_loss + 0.01 * ctrl_loss

            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 10.0)
            self.policy_opt.step()

            self.policy.eval()
            self.n_policy_updates += 1
            self.policy_loss_window.append(policy_loss.item())

    @torch.no_grad()
    def get_action(self, obs):
        s_norm = np.array([obs[0], obs[1], obs[2] / 8.0], dtype=np.float32)
        s_t = torch.tensor(s_norm, dtype=torch.float32, device=self.device).unsqueeze(0)
        a = self.policy(s_t).item()
        return a

    @property
    def recent_pred_error(self):
        if len(self.pred_error_window) == 0:
            return 0
        return np.mean(list(self.pred_error_window))

    @property
    def recent_policy_loss(self):
        if len(self.policy_loss_window) == 0:
            return 0
        return np.mean(list(self.policy_loss_window))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--wm', type=str, default='/tmp/kanrf_cl_exp/kan_cws_cl.pt')
    parser.add_argument('--policy', type=str, default='/tmp/kan_policy_trained.pt')
    parser.add_argument('--switch-step', type=int, default=1000)
    parser.add_argument('--total-steps', type=int, default=3000)
    parser.add_argument('--lr-wm', type=float, default=1e-3)
    parser.add_argument('--lr-policy', type=float, default=5e-4)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load pre-trained models (g=10)
    wm = KAN([4, 12, 3], grid_size=5, spline_order=3)
    wm.load_state_dict(torch.load(args.wm, weights_only=True, map_location=device))
    wm.to(device); wm.eval()

    policy = KANPolicy(state_dim=3, action_dim=1, hidden_dim=12, n_layers=2)
    policy.load_state_dict(torch.load(args.policy, weights_only=True, map_location=device))
    policy.to(device); policy.eval()
    print(f"KAN Policy: {sum(p.numel() for p in policy.parameters())} params")
    print(f"KAN WM:     {sum(p.numel() for p in wm.parameters())} params\n")

    learner = OnlineLearner(wm, policy, lr_wm=args.lr_wm,
                            lr_policy=args.lr_policy, device=device)

    print("=" * 60)
    print(f"ONLINE CONTINUAL LEARNING")
    print(f"  g=10 (0-{args.switch_step}), g=18 ({args.switch_step}-{args.total_steps})")
    print(f"  Per-step updates: WM(lr={args.lr_wm}) + Policy(lr={args.lr_policy})")
    print("=" * 60)

    env = ConfigurablePendulum(g=10.0, seed=42)
    obs = env.reset(seed=42)

    # Tracking
    episode_errors = []  # final |Δθ| per episode
    pred_errors = []     # per-step pred error
    policy_losses = []   # per-update policy loss
    current_ep = 0
    ep_steps = 0

    t_start = time.time()
    success_count = 0
    episode_steps_list = []

    for step in range(args.total_steps):
        # ── Gravity switch ──
        if step == args.switch_step:
            env.set_g(18.0); learner.set_gravity(18.0)
            print(f"\n{'='*60}")
            print(f"  Step {step}: GRAVITY 10 → 18  (online adaptation begins)")
            print(f"{'='*60}\n")

        # ── Get action ──
        a_norm = learner.get_action(obs)
        a_raw = a_norm * 2.0

        # ── Execute ──
        r = env.step([a_raw]); obs_next = r[0]; term = r[2] if len(r) > 2 else False

        # ── Online update (core of the experiment!) ──
        s_true_norm = np.array([obs_next[0], obs_next[1], obs_next[2] / 8.0], dtype=np.float32)
        s_cur_norm = np.array([obs[0], obs[1], obs[2] / 8.0], dtype=np.float32)
        learner.update(s_cur_norm, a_norm, s_true_norm)

        ep_steps += 1

        # ── Check episode end ──
        err = min(abs(np.arctan2(obs_next[1], obs_next[0]) - PI_2),
                  2 * np.pi - abs(np.arctan2(obs_next[1], obs_next[0]) - PI_2))
        if err < 0.2 or term or ep_steps >= 300:
            if err < 0.2: success_count += 1
            episode_errors.append(err)
            episode_steps_list.append(ep_steps)
            obs = env.reset()
            ep_steps = 0
        else:
            obs = obs_next

        # ── Periodic report ──
        if step % 200 == 0 or step == args.switch_step - 1 or step == args.switch_step + 10:
            recent_sr = success_count / max(1, len(episode_steps_list))
            print(f"  Step {step:4d}  pred_err={learner.recent_pred_error:.4f}  "
                  f"pol_loss={learner.recent_policy_loss:.4f}  "
                  f"|Δθ|={np.rad2deg(err):.0f}deg  "
                  f"recent_sr={recent_sr*100:.0f}%  "
                  f"wm_upd={learner.n_wm_updates}  pol_upd={learner.n_policy_updates}")

        pred_errors.append(learner.recent_pred_error)
        if len(learner.policy_loss_window) > 0:
            policy_losses.append(learner.recent_policy_loss)

    env.env.close()
    elapsed = time.time() - t_start
    print(f"\n  Done in {elapsed:.0f}s ({args.total_steps/elapsed:.0f} steps/s)")

    # ═══════════════════════════════════════════════════
    # Analysis
    # ═══════════════════════════════════════════════════
    n_eps = len(episode_errors)
    pre = slice(0, max(1, n_eps // 3))
    mid = slice(n_eps // 3, max(n_eps // 3 + 1, 2 * n_eps // 3))
    late = slice(2 * n_eps // 3, n_eps)

    print(f"\n{'='*60}")
    print("RESULTS")
    print("=" * 60)
    print(f"  Total episodes: {n_eps}")
    print(f"  Steps per second: {args.total_steps/elapsed:.0f}")
    print(f"  Per-step updates: WM={learner.n_wm_updates}  Policy={learner.n_policy_updates}")
    print(f"")
    print(f"  Success rate:")
    print(f"    g=10 (early): {np.mean([e<0.2 for e in episode_errors[pre]] if pre.stop>pre.start else [0])*100:.0f}%")
    print(f"    g=18 (mid):   {np.mean([e<0.2 for e in episode_errors[mid]] if mid.stop>mid.start else [0])*100:.0f}%")
    print(f"    g=18 (late):  {np.mean([e<0.2 for e in episode_errors[late]] if late.stop>late.start else [0])*100:.0f}%")
    print(f"")
    print(f"  Prediction error:")
    pre_pe = pred_errors[:args.switch_step]
    post_pe = pred_errors[args.switch_step:args.switch_step+500]
    late_pe = pred_errors[args.switch_step+500:]
    if pre_pe: print(f"    g=10:  {np.mean(pre_pe):.4f}")
    if post_pe: print(f"    g=18 (post): {np.mean(post_pe):.4f}")
    if late_pe: print(f"    g=18 (late): {np.mean(late_pe):.4f}")


if __name__ == '__main__':
    main()
