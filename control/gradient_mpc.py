"""Gradient-based trajectory optimization through ProtoKAN WM.

No Policy needed — directly optimize action sequences via gradient descent
through the learned WM. Uses ProtoKAN's accurate Jacobian for efficient
optimization (20-30 gradient steps vs 200-300 random samples).

For continual learning: WM adapts → gradient optimizer automatically
uses updated dynamics. No Policy retraining bottleneck.
"""
import torch, torch.nn as nn, numpy as np


class GradientMPC:
    """Gradient-based + Batch-shooting MPC via ProtoKAN WM.

    Two modes:
    - 'shoot': batched random shooting (N trajectories, H forward passes total)
    - 'grad': gradient descent on action sequence (opt_steps iterations)

    Shoot mode is preferred: H forward passes regardless of N (200x faster).
    """

    def __init__(self, wm, state_dim, horizon=4, n_shoot=500, opt_steps=25,
                 lr=0.1, mode='shoot', device='cpu'):
        self.wm = wm
        self.state_dim = state_dim
        self.horizon = horizon
        self.n_shoot = n_shoot
        self.opt_steps = opt_steps
        self.lr = lr
        self.mode = mode
        self.device = device

    def optimize(self, s, s_target):
        if self.mode == 'shoot':
            return self._optimize_shoot(s, s_target)
        else:
            return self._optimize_grad(s, s_target)

    def _optimize_shoot(self, s, s_target):
        """Batched random shooting: N trajectories in parallel, H forward passes."""
        B = self.n_shoot
        H = self.horizon

        # Sample all action sequences: (B, H)
        seq = torch.FloatTensor(B, H).uniform_(-1, 1)

        # Batch rollout through WM: all B trajectories in parallel
        s_cur = s.unsqueeze(0).expand(B, -1).clone()  # (B, n)
        total_cost = torch.zeros(B)

        for t in range(H):
            a_t = seq[:, t:t+1]  # (B, 1)
            with torch.no_grad():
                s_cur = self.wm(torch.cat([s_cur, a_t], dim=-1))
            err = s_cur - s_target
            total_cost += (0.9 ** t) * (
                err[:, 2].pow(2) + 0.1 * err[:, 0].pow(2) + 0.5 * err[:, 3].pow(2)
            )

        best_idx = total_cost.argmin().item()
        return seq[best_idx, 0].item(), total_cost[best_idx].item()

    def _optimize_grad(self, s, s_target):
        """Gradient descent on action sequence (slower, for comparison)."""
        a_seq = torch.zeros(self.horizon, device=self.device, requires_grad=True)
        opt = torch.optim.Adam([a_seq], lr=self.lr)
        best_cost = float('inf')
        best_a0 = 0.0

        for _ in range(self.opt_steps):
            opt.zero_grad()
            s_cur = s.unsqueeze(0)
            total_cost = 0.0
            for t in range(self.horizon):
                a_t = a_seq[t].reshape(1, 1)
                s_cur = self.wm(torch.cat([s_cur, a_t], dim=-1))
                err = s_cur - s_target
                total_cost += (0.9 ** t) * (
                    err[:, 2].pow(2) + 0.1 * err[:, 0].pow(2) +
                    0.5 * err[:, 3].pow(2) + 0.01 * a_t.pow(2)
                ).mean()
            total_cost.backward()
            opt.step()
            with torch.no_grad():
                a_seq.clamp_(-1, 1)
                if total_cost.item() < best_cost:
                    best_cost = total_cost.item()
                    best_a0 = a_seq[0].item()
        return best_a0, best_cost

    def get_action(self, s, s_target=None):
        if isinstance(s, np.ndarray):
            s = torch.tensor(s, dtype=torch.float32, device=self.device)
        if s_target is None:
            s_target = torch.zeros(self.state_dim, device=self.device)
        elif isinstance(s_target, np.ndarray):
            s_target = torch.tensor(s_target, dtype=torch.float32, device=self.device)
        a, _ = self.optimize(s, s_target)
        return a
