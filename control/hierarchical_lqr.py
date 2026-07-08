"""Hierarchical LQR: use temporal hierarchy to structure Q matrix,
then solve LQR for optimal feedback gain K and cost-to-go.

Key improvement over THTP pseudo-inverse routing:
- Handles parallel causal branches (DAG, not chain)
- Handles cross-branch coupling via full A = ∂s'/∂s matrix
- Multi-step horizon breaks the 1-step myopia ceiling
- Q matrix derived from hierarchy: Tier k weight = γ^k (deeper = higher)
"""
import torch, numpy as np
from control.thtp import TemporalHierarchy


class HierarchicalLQR:
    """Structure Q from hierarchy, solve LQR for optimal feedback."""

    def __init__(self, wm, hierarchy: TemporalHierarchy, horizon=4,
                 q_base=10.0, r_weight=0.1, device='cpu'):
        self.wm = wm
        self.h = hierarchy
        self.horizon = horizon
        self.q_base = q_base     # base Q weight for deepest tier
        self.r_weight = r_weight
        self.state_dim = hierarchy.state_dim
        self.device = device

        # Build Q matrix from hierarchy
        self.Q = self._build_Q()

    def _build_Q(self):
        """Q[i,i] = q_base * γ^{tier_of[i]} — deeper tiers get higher weight."""
        Q = torch.zeros(self.state_dim, self.state_dim, device=self.device)
        for i in range(self.state_dim):
            tier = self.h.tier_of[i]
            # Deeper tier = larger tier index = higher weight
            Q[i, i] = self.q_base * (2.0 ** tier)
        return Q

    def linearize(self, s):
        """Compute A=∂s'/∂s and B=∂s'/∂a at state s (with a=0).

        Returns A (n×n), B (n×1).
        """
        wm = self.wm
        wm.eval()
        n = self.state_dim

        was_frozen = not next(wm.parameters()).requires_grad
        if was_frozen:
            for p in wm.parameters():
                p.requires_grad = True

        s_batch = s.unsqueeze(0).clone().detach()

        # B = ∂s'/∂a
        a = torch.zeros(1, 1, device=self.device, requires_grad=True)
        s_pred = wm(torch.cat([s_batch, a], dim=-1))
        B = torch.zeros(n, 1, device=self.device)
        for i in range(n):
            g = torch.autograd.grad(s_pred[0, i], a, retain_graph=True)[0]
            B[i, 0] = g[0, 0]

        # A = ∂s'/∂s
        s_j = s_batch.clone().detach().requires_grad_(True)
        a_fixed = torch.zeros(1, 1, device=self.device)
        s_pred = wm(torch.cat([s_j, a_fixed], dim=-1))
        A = torch.zeros(n, n, device=self.device)
        for j in range(n):
            for i in range(n):
                g = torch.autograd.grad(s_pred[0, i], s_j, retain_graph=True)[0]
                A[i, j] = g[0, j]

        if was_frozen:
            for p in wm.parameters():
                p.requires_grad = False

        # Also compute f(s, 0) — the zero-action prediction
        with torch.no_grad():
            f0 = wm(torch.cat([s_batch, a_fixed], dim=-1)).squeeze(0)

        return A, B, f0

    def solve(self, s, s_target):
        """Solve multi-step LQR around current state.

        Args:
            s: (n,) current state
            s_target: (n,) target state

        Returns:
            K: (horizon, 1, n) optimal feedback gains
            k: (horizon, 1,) optimal feedforward terms
            u_0: optimal first action
            cost_to_go: scalar
        """
        n = self.state_dim
        A, B, f0 = self.linearize(s)

        # Linearized dynamics around (s, a=0):
        # s_{t+1} = f(s_t, a_t) ≈ A s_t + B a_t + c
        # where c = f(s,0) - A s
        c = f0 - A @ s

        # Convert to error coordinates: e_t = s_t - s_target
        # e_{t+1} = A e_t + B a_t + d
        # where d = (A - I) s_target + c
        d = (A - torch.eye(n, device=self.device)) @ s_target + c

        # Riccati backward recursion
        P = self.Q.clone()  # terminal cost = Q
        K_list = []
        k_list = []

        for _ in range(self.horizon):
            # Optimal gain: K = -(R + B^T P B)^{-1} B^T P A
            BtPB = B.T @ P @ B  # (1, 1)
            R_eff = self.r_weight + BtPB
            BtPA = B.T @ P @ A  # (1, n)

            K = -BtPA / R_eff  # (1, n)

            # Feedforward term for disturbance d
            BtPd = B.T @ P @ d  # (1,)
            k_d = -BtPd / R_eff  # (1,)

            K_list.append(K)
            k_list.append(k_d)

            # Riccati update: P = Q + A^T P A - A^T P B (R + B^T P B)^{-1} B^T P A
            AtPA = A.T @ P @ A
            AtPB = A.T @ P @ B  # (n, 1)
            P = self.Q + AtPA - (AtPB @ BtPA) / R_eff

        # Reverse to get time order (t=0 first)
        K_list = K_list[::-1]
        k_list = k_list[::-1]

        # First optimal action: u_0 = K_0 e_0 + k_0
        e_0 = s - s_target
        u_0 = (K_list[0] @ e_0 + k_list[0]).squeeze()
        u_0 = u_0.clamp(-2.0, 2.0)  # torque limit

        # Cost-to-go estimate
        cost_to_go = (e_0 @ P @ e_0).item()

        return u_0, K_list, k_list, cost_to_go

    def compute_batch(self, s_batch, s_target, n_samples=500):
        """Pre-compute LQR solutions for a batch of states.

        Returns (samples, a_optimal) pairs for Policy distillation.
        """
        N = min(n_samples, s_batch.shape[0])
        idx = torch.randperm(s_batch.shape[0])[:N]
        s_sel = s_batch[idx]
        a_opt = torch.zeros(N, 1, device=self.device)

        for i in range(N):
            u_0, _, _, _ = self.solve(s_sel[i], s_target.squeeze(0))
            a_opt[i, 0] = u_0

        return s_sel, a_opt
