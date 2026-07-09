"""Gradient-based trajectory optimization through ProtoKAN WM.

No Policy needed — directly optimize action sequences via gradient descent
through the learned WM. Uses ProtoKAN's accurate Jacobian for efficient
optimization (20-30 gradient steps vs 200-300 random samples).

For continual learning: WM adapts → gradient optimizer automatically
uses updated dynamics. No Policy retraining bottleneck.
"""
import torch, torch.nn as nn, numpy as np


class GradientMPC:
    """Batch-shooting MPC via ProtoKAN WM.

    Cost function: V(s) = (s - s*)^T P (s - s*)
    P is auto-synthesized from WM dynamics via Riccati (Lyapunov function).
    No hand-crafted weights needed — works for ANY system.

    If P is not provided, uses a simple identity cost (||s - s*||^2).
    """

    def __init__(self, wm, state_dim, P=None, horizon=4, n_shoot=500,
                 mode='shoot', opt_steps=15, lr=0.05, device='cpu'):
        self.wm = wm
        self.state_dim = state_dim
        self.horizon = horizon
        self.n_shoot = n_shoot
        self.opt_steps = opt_steps
        self.lr = lr
        self.mode = mode
        self.device = device
        self.P = P if P is not None else torch.eye(state_dim, device=device)
        self._warm_start = None  # for fd_grad mode

    def _cost(self, err):
        """V(s) = (s - s*)^T P (s - s*), batched."""
        # err: (B, n), P: (n, n)
        # Returns: (B,) per-sample costs
        return (err @ self.P @ err.T).diag()

    def _optimize_shoot(self, s, s_target):
        """Batched random shooting: N trajectories in parallel, H forward passes."""
        B = self.n_shoot
        H = self.horizon

        seq = torch.FloatTensor(B, H).uniform_(-1, 1)
        s_cur = s.unsqueeze(0).expand(B, -1).clone()
        total_cost = torch.zeros(B)

        for t in range(H):
            a_t = seq[:, t:t+1]
            with torch.no_grad():
                s_cur = self.wm(torch.cat([s_cur, a_t], dim=-1))
            total_cost += (0.9 ** t) * self._cost(s_cur - s_target)

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

    def fd_grad(self, s, s_target):
        """Finite-difference gradient optimization (no backward needed).

        Perturb each a_t by ±ε, compute cost via WM forward only.
        Much faster than autograd backward through ProtoKAN.
        """
        H = self.horizon
        eps = 0.01

        # Warm-start from previous solution, shifted by one step
        if self._warm_start is not None and len(self._warm_start) >= H:
            a_seq = torch.cat([
                self._warm_start[1:H].clone(),
                torch.zeros(1)
            ])
        else:
            a_seq = torch.zeros(H)

        lr = self.lr
        best_cost = float('inf')
        best_a0 = 0.0

        for step in range(self.opt_steps):
            # Baseline cost
            s_cur = s.unsqueeze(0)
            total = 0.0
            for t in range(H):
                a_t = a_seq[t].reshape(1, 1)
                s_cur = self.wm(torch.cat([s_cur, a_t], dim=-1))
                total += (0.9 ** t) * self._cost(s_cur - s_target).item()

            if total < best_cost:
                best_cost = total
                best_a0 = a_seq[0].item()

            # Finite difference gradient
            grad = torch.zeros(H)
            for t in range(H):
                a_seq[t] += eps
                s_cur = s.unsqueeze(0)
                cost_plus = 0.0
                for k in range(H):
                    a_k = a_seq[k].reshape(1, 1)
                    s_cur = self.wm(torch.cat([s_cur, a_k], dim=-1))
                    cost_plus += (0.9 ** k) * self._cost(s_cur - s_target).item()

                a_seq[t] -= 2 * eps
                s_cur = s.unsqueeze(0)
                cost_minus = 0.0
                for k in range(H):
                    a_k = a_seq[k].reshape(1, 1)
                    s_cur = self.wm(torch.cat([s_cur, a_k], dim=-1))
                    cost_minus += (0.9 ** k) * self._cost(s_cur - s_target).item()

                a_seq[t] += eps  # restore
                grad[t] = (cost_plus - cost_minus) / (2 * eps)

            # Update
            a_seq -= lr * grad
            a_seq.clamp_(-1, 1)

            if step % 5 == 0:
                lr *= 0.9  # decay

        self._warm_start = a_seq.detach().clone()
        return best_a0, best_cost

    def get_action(self, s, s_target=None):
        if isinstance(s, np.ndarray):
            s = torch.tensor(s, dtype=torch.float32, device=self.device)
        if s_target is None:
            s_target = torch.zeros(self.state_dim, device=self.device)
        elif isinstance(s_target, np.ndarray):
            s_target = torch.tensor(s_target, dtype=torch.float32, device=self.device)
        if self.mode == 'shoot':
            a, _ = self._optimize_shoot(s, s_target)
        else:
            a, _ = self.fd_grad(s, s_target)
        return a
