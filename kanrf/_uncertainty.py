"""B-spline activation density as epistemic uncertainty.

Key insight: when an input falls in a region rarely visited during training,
the B-spline control points c_{i,j,k} are near initialization, and the
spline contribution ||Σ c_{i,j,k} B_k(x)||² is low.

The uncertainty penalty is:
  L_unc = (1 / N_edges) * Σ_{l,i,j} σ² / (σ² + ||φ_{i,j}||²)

This is bounded in [0, 1], smooth, differentiable, and zero-cost
(activations are already computed during the KAN forward pass).
"""
import torch


def compute_uncertainty(B_list, E_list, sigma2: float = 0.01):
    """Compute B-spline activation uncertainty.

    Args:
        B_list: list of (batch, in_dim, n_basis) — B-spline basis per layer
        E_list: list of (batch, out_dim, in_dim) — spline energy per edge per layer
        sigma2: scale parameter (smaller = more sensitive to low activation)

    Returns:
        scalar uncertainty penalty in [0, 1]
    """
    total = 0.0
    count = 0
    for E in E_list:
        # E: (batch, out_dim, in_dim) — each entry is ||φ_{i,j}||²
        total += (sigma2 / (sigma2 + E + 1e-8)).sum()
        count += E.numel()
    if count == 0:
        return torch.tensor(0.0)
    return total / count


def compute_per_step_uncertainty(B_list, E_list, sigma2: float = 0.01):
    """Compute uncertainty for each step (not averaged over batch).

    Returns (batch,) tensor of per-sample uncertainty.
    """
    total = 0.0
    count = 0
    for E in E_list:
        # E: (batch, out_dim, in_dim)
        total += (sigma2 / (sigma2 + E + 1e-8)).sum(dim=(1, 2))  # (batch,)
        count += E.shape[1] * E.shape[2]
    if count == 0:
        return torch.zeros(1)
    return total / count
