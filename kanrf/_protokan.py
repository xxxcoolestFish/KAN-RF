"""ProtoKAN: Prototype-based Kolmogorov-Arnold Network.

Replaces B-spline edge functions with learnable prototype points + kernel
interpolation. Each edge function is:

    φ(x) = Σ_n w_n(x) · [y_n + d_n · (x - x_n)]

    w_n(x) = softmax(-(x - x_n)² / 2σ²)

Where {x_n, y_n, d_n} are learnable prototype (position, value, derivative)
and σ is a learnable kernel width.

Key advantages over B-spline KAN:
- No fixed grid: prototypes adaptively concentrate where needed
- Fully parallel: all ops are matmul + exp, no recursion
- Smoother: Gaussian kernel provides natural multi-scale smoothing
- More expressive: N prototypes with first-order info ≫ G+K control points
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProtoKANLayer(nn.Module):
    """Single ProtoKAN layer: each edge is a learnable prototype interpolant.

    output_i = Σ_j ( w_{i,j}·silu(x_j) + φ_{i,j}(x_j) )
    where φ_{i,j} is defined by N shared prototype positions + per-edge values/derivs.
    """

    def __init__(self, in_dim: int, out_dim: int, n_prototypes: int = 16,
                 grid_range: float = 1.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_prototypes = n_prototypes

        # Shared prototype positions across all edges (like B-spline grid)
        proto_pos = torch.linspace(-grid_range, grid_range, n_prototypes)
        # Add small noise so prototypes can differentiate
        proto_pos = proto_pos + torch.randn(n_prototypes) * 0.02
        self.proto_pos = nn.Parameter(proto_pos)

        # Per-edge prototype values: what the edge outputs near each prototype
        self.proto_val = nn.Parameter(
            torch.randn(out_dim, in_dim, n_prototypes) * 0.01)

        # Per-edge prototype derivatives: first-order slope at each prototype
        self.proto_der = nn.Parameter(
            torch.randn(out_dim, in_dim, n_prototypes) * 0.01)

        # Kernel width: larger σ = smoother but less expressive
        # Initialize σ ≈ 1.0 (covers full [-1,1] range) → start smooth
        self.log_sigma = nn.Parameter(torch.tensor(0.0))

        # Base weight: same silu-bypass as original KAN
        self.base_weight = nn.Parameter(torch.empty(out_dim, in_dim))
        nn.init.xavier_uniform_(self.base_weight)

    def forward(self, x: torch.Tensor, return_activations: bool = False):
        """Forward pass.

        Args:
            x: (batch, in_dim)
            return_activations: if True, also return kernel weights and edge energies

        Returns:
            y: (batch, out_dim)
            (if return_activations): (y, weights, edge_energy)
        """
        batch = x.shape[0]
        sigma = torch.exp(self.log_sigma).clamp(1e-4, 10.0)

        # ── Compute kernel weights ──
        # x: (B, in_dim), proto_pos: (N,)
        # diff: (B, in_dim, N)
        diff = x.unsqueeze(-1) - self.proto_pos.unsqueeze(0).unsqueeze(0)

        # Weights: softmax over prototypes
        # logits = -diff² / (2σ²) → weights = softmax(logits)
        logits = -diff.pow(2) / (2.0 * sigma ** 2)
        weights = torch.softmax(logits, dim=-1)  # (B, in_dim, N)

        # ── Per-edge predictions ──
        # proto_val, proto_der: (out_dim, in_dim, N)
        # diff.unsqueeze(1): (B, 1, in_dim, N)
        # preds: (B, out_dim, in_dim, N) = val + der * (x - x_n)
        preds = (self.proto_val.unsqueeze(0) +
                 self.proto_der.unsqueeze(0) * diff.unsqueeze(1))

        # Edge outputs: sum over prototypes
        # weights.unsqueeze(1): (B, 1, in_dim, N)
        # edge_out: (B, out_dim, in_dim)
        edge_out = (weights.unsqueeze(1) * preds).sum(dim=-1)

        # Sum over input dimensions → (B, out_dim)
        proto_out = edge_out.sum(dim=-1)

        # Base path: silu(x) @ base_weight^T
        base_out = F.silu(x) @ self.base_weight.T

        if return_activations:
            # Return kernel weights and per-edge energy (compatible with KAN API)
            # B: (B, in_dim, N) — kernel weights
            # E: (B, out_dim, in_dim) — per-edge energy
            B = weights
            E = edge_out.pow(2)
            return base_out + proto_out, B, E

        return base_out + proto_out

    def repulsion_loss(self, tau: float = 0.1) -> torch.Tensor:
        """Regularizer to prevent prototype collapse.

        Penalizes prototype pairs that are closer than `tau` in input space.

        Args:
            tau: characteristic distance below which prototypes repel

        Returns:
            scalar penalty (mean over all pairs)
        """
        N = self.n_prototypes
        diff = self.proto_pos.unsqueeze(-1) - self.proto_pos.unsqueeze(0)  # (N, N)
        dist2 = diff.pow(2)
        # Only off-diagonal pairs
        mask = ~torch.eye(N, dtype=torch.bool, device=self.proto_pos.device)
        # exp(-d²/2τ²): high penalty when d << τ, low when d >> τ
        penalty = torch.exp(-dist2[mask] / (2.0 * tau ** 2))
        return penalty.mean()


class ProtoKAN(nn.Module):
    """Multi-layer ProtoKAN network.

    Drop-in replacement for KAN with prototype-based edge functions.
    """

    def __init__(self, layer_dims: list, n_prototypes: int = 16,
                 grid_range: float = 1.0):
        super().__init__()
        self.n_prototypes = n_prototypes
        self.layers = nn.ModuleList([
            ProtoKANLayer(layer_dims[i], layer_dims[i + 1],
                          n_prototypes=n_prototypes,
                          grid_range=grid_range)
            for i in range(len(layer_dims) - 1)
        ])

    def forward(self, x, return_activations: bool = False):
        if return_activations:
            B_list, E_list = [], []
            for layer in self.layers:
                x, B, E = layer(x, return_activations=True)
                B_list.append(B)
                E_list.append(E)
            return x, B_list, E_list
        for layer in self.layers:
            x = layer(x)
        return x

    def repulsion_loss(self, tau: float = 0.1) -> torch.Tensor:
        """Sum of repulsion losses across all layers."""
        return sum(layer.repulsion_loss(tau) for layer in self.layers)
