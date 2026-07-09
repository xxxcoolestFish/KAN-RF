"""Hierarchical Cost Function: Tier-guided dynamic cost for swing-up tasks.

Key insight: Tier 0 dimensions (means, directly controllable) should NOT
be penalized for being "far from zero". Instead, they should be penalized
for deviating from the value that would BEST reduce Tier 1 errors.

Tier 1 error → (WM Jacobian) → Tier 0 desired value → cost penalizes deviation.
"""
import torch, numpy as np


class HierarchicalCost:
    """Dynamic cost function: penalizes Tier 1 errors + Tier 0 deviation from desired."""

    def __init__(self, wm, state_dim, s_target, tier_of,
                 damping=0.1, n_jac_samples=200, device='cpu'):
        self.wm = wm
        self.state_dim = state_dim
        self.s_target = s_target.to(device)
        self.tier_of = tier_of  # (n,) 0 = means, 1+ = goals
        self.tier0 = [i for i in range(state_dim) if tier_of[i] == 0]
        self.tier1 = [i for i in range(state_dim) if tier_of[i] > 0]
        self.damping = damping
        self.device = device

        # Pre-compute average Jacobian: ∂(Tier1)/∂(Tier0)
        self.J_1_0 = self._compute_jacobian(wm, state_dim, n_jac_samples, device)
        print(f"  HierarchicalCost: Tier0={self.tier0}, Tier1={self.tier1}")
        print(f"  J(Tier1←Tier0) shape=({len(self.tier1)},{len(self.tier0)})")

    def _compute_jacobian(self, wm, n, n_samples, device):
        """Compute ∂(Tier1)/∂(Tier0): how Tier 0 changes affect next Tier 1 values."""
        wm.eval()
        was_frozen = not next(wm.parameters()).requires_grad
        if was_frozen:
            for p in wm.parameters(): p.requires_grad = True

        J_sum = torch.zeros(len(self.tier1), len(self.tier0), device=device)
        N = n_samples
        for _ in range(N):
            s = torch.randn(1, n, device=device).clamp(-1, 1)
            s.requires_grad_(True)
            a0 = torch.zeros(1, 1, device=device)
            sp = wm(torch.cat([s, a0], dim=-1))
            for i, t1 in enumerate(self.tier1):
                for j, t0 in enumerate(self.tier0):
                    g = torch.autograd.grad(sp[0, t1], s, retain_graph=True)[0]
                    J_sum[i, j] += g[0, t0].abs()

        if was_frozen:
            for p in wm.parameters(): p.requires_grad = False

        return J_sum / N

    def __call__(self, s):
        """Compute hierarchical cost for a batch of states.

        For each Tier 0 dimension j:
            error_Tier1[i] for each goal dimension i
            → desired Tier0[j] = Σ_i (error[i] / J[i,j] + damping) * weight
            → penalize (s[j] - desired_Tier0[j])²

        Args:
            s: (B, state_dim) predicted next states
        Returns:
            cost: (B,) per-sample costs
        """
        B = s.shape[0]
        s_tgt = self.s_target.expand(B, -1)

        # Tier 1 error (goal dimensions)
        err_t1 = s[:, self.tier1] - s_tgt[:, self.tier1]  # (B, |Tier1|)
        cost_t1 = err_t1.pow(2).sum(dim=-1)  # (B,)

        # Tier 0: compute desired value from Tier 1 errors
        # desired_Tier0 = -J^+ · error_Tier1
        # For each Tier 0 dim j: desired[j] = current[j] - Σ_i (err[i] / J[i,j])
        # Since we want to CANCEL the projected change from current Tier0
        J = self.J_1_0  # (|Tier1|, |Tier0|)
        # Simple: desired_t0 = -err_t1 @ J_pinv
        # Use scaled pseudo-inverse: (J^T J + λI)^-1 J^T
        JTJ = J @ J.T + self.damping * torch.eye(len(self.tier1), device=self.device)
        J_pinv = torch.linalg.solve(JTJ, J)  # (|Tier1|, |Tier0|)
        desired_t0 = -(err_t1 @ J_pinv)  # (B, |Tier0|)

        # Tier 0 deviation cost
        cost_t0 = (s[:, self.tier0] - desired_t0).pow(2).sum(dim=-1)  # (B,)

        return cost_t1 + 0.5 * cost_t0
