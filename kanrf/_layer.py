import torch
import torch.nn as nn
import torch.nn.functional as F
from kanrf._bspline import bspline_basis


class KANLayer(nn.Module):
    """A single KAN layer: output_i = Σ_j (w_{i,j}·silu(x_j) + Σ_k c_{i,j,k}·B_k(x_j))."""

    def __init__(self, in_dim: int, out_dim: int, grid_size: int = 5,
                 spline_order: int = 3, grid_range: float = 1.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.grid_size = grid_size
        self.spline_order = spline_order

        # Shared grid for all input dimensions
        grid = torch.linspace(-grid_range, grid_range, grid_size + 1)
        self.register_buffer('grid', grid)

        # Base weight: (out_dim, in_dim) — scales the SiLU term
        self.base_weight = nn.Parameter(torch.empty(out_dim, in_dim))
        nn.init.xavier_uniform_(self.base_weight)

        # Spline control points: (out_dim, in_dim, grid_size + spline_order)
        n_basis = grid_size + spline_order
        self.spline_weight = nn.Parameter(torch.empty(out_dim, in_dim, n_basis))
        nn.init.normal_(self.spline_weight, std=0.1)

    def forward(self, x: torch.Tensor, return_activations: bool = False):
        """Forward pass.

        Args:
            x: (batch, in_dim)
            return_activations: if True, also return B-spline basis and spline energies

        Returns:
            y: (batch, out_dim)
            (if return_activations): also returns (B, spline_energy)
              B: (batch, in_dim, n_basis) — basis function values
              spline_energy: (batch, out_dim, in_dim) — ||Σ c_{i,j,k} B_k||² per edge
        """
        batch = x.shape[0]

        # Compute B-spline basis: vectorized over all input dims
        # x: (batch, in_dim) → flatten → bspline on all at once → reshape
        x_flat = x.reshape(-1)  # (batch * in_dim,)
        B_flat = bspline_basis(x_flat, self.grid, self.spline_order)  # (N, n_basis)
        B = B_flat.reshape(batch, self.in_dim, -1)  # (batch, in_dim, n_basis)

        # Spline part: Σ_j Σ_k c_{i,j,k} · B_k(x_j)
        spline_out = torch.einsum('bjk,oik->bo', B, self.spline_weight)

        # Base part: Σ_j w_{i,j} · silu(x_j)
        base_out = F.silu(x) @ self.base_weight.T

        if return_activations:
            # Per-edge spline energy: (Σ_k c_{i,j,k} B_k(x_j))² aggregated over batch
            # spline_weight: (out_dim, in_dim, n_basis)
            # B: (batch, in_dim, n_basis)
            # edge_contrib: (batch, out_dim, in_dim) = Σ_k c_{i,j,k} * B_k(x_j)
            edge_contrib = torch.einsum('bjk,oik->boj', B, self.spline_weight)
            spline_energy = edge_contrib ** 2  # (batch, out_dim, in_dim)
            return base_out + spline_out, B, spline_energy

        return base_out + spline_out
