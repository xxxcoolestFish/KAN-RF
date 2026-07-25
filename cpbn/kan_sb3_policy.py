"""SB3-compatible KAN feature extractor with smoothness regularisation."""

from __future__ import annotations

import torch
import torch.nn as nn
from gymnasium import spaces
from itertools import combinations
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from kanrf import bspline_basis


class BsplineFeaturesExtractor(BaseFeaturesExtractor):
    """B-spline feature expansion as an SB3 FeaturesExtractor.

    Each state dimension is expanded via B-spline basis functions; pair
    interactions are added via compressed outer products.  The output
    feeds directly into SB3's mlp_extractor + actor/value heads.

    A smoothness penalty on per-dimension B-spline coefficients is exposed
    via ``smoothness_loss()`` for external regularisation callbacks.
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        *,
        grid_size: int = 5,
        spline_order: int = 3,
        pair_modes: int = 3,
        state_scale: float = 5.0,
    ):
        super().__init__(observation_space, features_dim=1)  # placeholder
        state_dim = observation_space.shape[0]
        self.state_dim = state_dim
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.contrast_dim = grid_size + spline_order - 1
        self.pair_modes = min(pair_modes, self.contrast_dim)

        grid = torch.linspace(-1.0, 1.0, grid_size + 1)
        self.register_buffer("grid", grid)
        self.register_buffer("state_scale",
                             torch.full((state_dim,), float(state_scale)))

        # DCT compression for pair interactions
        index = torch.arange(self.contrast_dim, dtype=torch.float32)[:, None]
        mode = torch.arange(self.pair_modes, dtype=torch.float32)[None, :]
        proj = torch.cos(torch.pi * (index + 0.5) * mode / self.contrast_dim)
        self.register_buffer("pair_projection",
                             torch.linalg.qr(proj, mode="reduced").Q)

        feature_dim = (
            1
            + state_dim * self.contrast_dim
            + len(tuple(combinations(range(state_dim), 2))) * self.pair_modes**2
        )
        self._features_dim = feature_dim

        # ── learnable linear combination of B-spline features ──────────
        self.coefficients = nn.Parameter(torch.zeros(feature_dim) * 0.01)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        normalized = (observations / self.state_scale).clamp(-1.0, 1.0)
        local = bspline_basis(
            normalized.reshape(-1), self.grid, self.spline_order,
        ).reshape(*observations.shape, -1)[..., :-1]
        blocks = [local.flatten(start_dim=-2)]
        compressed = local @ self.pair_projection
        for first, second in combinations(range(self.state_dim), 2):
            blocks.append(torch.einsum(
                "...i,...j->...ij",
                compressed[..., first, :],
                compressed[..., second, :],
            ).flatten(start_dim=-2))
        constant = torch.ones(
            *observations.shape[:-1], 1, device=observations.device,
        )
        raw = torch.cat((constant, *blocks), dim=-1)
        return raw * self.coefficients  # modulated by learned coefficients

    def smoothness_loss(self) -> torch.Tensor:
        """Laplacian smoothness on per-state-dimension coefficient blocks."""
        loss = torch.tensor(0.0, device=self.grid.device)
        for dim_idx in range(self.state_dim):
            start = 1 + dim_idx * self.contrast_dim
            end = start + self.contrast_dim
            c = self.coefficients[start:end]
            loss = loss + (c[1:] - c[:-1]).pow(2).sum()
        return loss
