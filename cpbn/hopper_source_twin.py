"""High-capacity control-affine source digital twin for Hopper."""

from __future__ import annotations

import torch
from torch import nn

from kanrf import bspline_basis


class HopperSourceAffineTwin(nn.Module):
    """Predict source transition effects around a frozen source policy."""

    def __init__(
        self,
        state_scale,
        delta_scale,
        *,
        action_dim: int = 3,
        hidden_dim: int = 256,
        depth: int = 3,
    ):
        super().__init__()
        state_scale = torch.as_tensor(state_scale, dtype=torch.float32)
        delta_scale = torch.as_tensor(delta_scale, dtype=torch.float32)
        self.state_dim = int(state_scale.numel())
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.register_buffer("state_scale", state_scale)
        self.register_buffer("delta_scale", delta_scale)
        layers = []
        dimension = self.state_dim
        for _ in range(depth):
            layers.extend((nn.Linear(dimension, hidden_dim), nn.SiLU()))
            dimension = hidden_dim
        self.trunk = nn.Sequential(*layers)
        self.baseline_head = nn.Linear(hidden_dim, self.state_dim)
        self.gain_head = nn.Linear(
            hidden_dim, self.state_dim * self.action_dim,
        )

    def normalized_state(self, state):
        return (state / self.state_scale).clamp(-5.0, 5.0)

    def drift_and_gain(self, state):
        feature = self.trunk(self.normalized_state(state))
        baseline = self.baseline_head(feature) * self.delta_scale
        gain = self.gain_head(feature).reshape(
            *state.shape[:-1], self.state_dim, self.action_dim,
        ) * self.delta_scale[..., None]
        return baseline, gain

    def forward(self, state, innovation):
        baseline, gain = self.drift_and_gain(state)
        return baseline + (
            gain @ innovation.unsqueeze(-1)
        ).squeeze(-1)


class SparseComposableKANTwin(nn.Module):
    """Sparse ANOVA KAN twin for recombining low-dimensional mechanisms."""

    def __init__(
        self,
        state_scale,
        delta_scale,
        *,
        action_dim: int = 3,
        grid_size: int = 6,
        spline_order: int = 2,
        pair_modes: int = 4,
        support: float = 1.5,
    ):
        super().__init__()
        state_scale = torch.as_tensor(state_scale, dtype=torch.float32)
        delta_scale = torch.as_tensor(delta_scale, dtype=torch.float32)
        self.state_dim = int(state_scale.numel())
        self.action_dim = int(action_dim)
        self.grid_size = int(grid_size)
        self.spline_order = int(spline_order)
        self.contrast_dim = grid_size + spline_order - 1
        self.pair_modes = min(int(pair_modes), self.contrast_dim)
        self.support = float(support)
        self.register_buffer("state_scale", state_scale)
        self.register_buffer("delta_scale", delta_scale)
        self.register_buffer(
            "grid",
            torch.linspace(-support, support, grid_size + 1),
        )
        index = torch.arange(self.contrast_dim, dtype=torch.float32)[:, None]
        mode = torch.arange(self.pair_modes, dtype=torch.float32)[None, :]
        projection = torch.cos(
            torch.pi * (index + 0.5) * mode / self.contrast_dim,
        )
        self.register_buffer(
            "pair_projection",
            torch.linalg.qr(projection, mode="reduced").Q,
        )
        self.pairs = tuple(
            (first, second)
            for first in range(self.state_dim)
            for second in range(first + 1, self.state_dim)
        )
        self.group_slices = [slice(0, 1)]
        offset = 1
        for _ in range(self.state_dim):
            self.group_slices.append(
                slice(offset, offset + self.contrast_dim),
            )
            offset += self.contrast_dim
        for _ in self.pairs:
            self.group_slices.append(
                slice(offset, offset + self.pair_modes**2),
            )
            offset += self.pair_modes**2
        self.feature_dim = offset
        self.baseline_coefficient = nn.Parameter(torch.zeros(
            self.feature_dim, self.state_dim,
        ))
        self.gain_coefficient = nn.Parameter(torch.zeros(
            self.feature_dim, self.state_dim, self.action_dim,
        ))

    def features(self, state):
        normalized = (
            state / self.state_scale
        ).clamp(-self.support, self.support)
        local = bspline_basis(
            normalized.reshape(-1),
            self.grid,
            self.spline_order,
        ).reshape(*state.shape, -1)[..., :-1]
        blocks = [torch.ones(
            *state.shape[:-1], 1, device=state.device,
        )]
        blocks.extend(
            local[..., index, :]
            for index in range(self.state_dim)
        )
        compressed = local @ self.pair_projection
        blocks.extend(
            torch.einsum(
                "...i,...j->...ij",
                compressed[..., first, :],
                compressed[..., second, :],
            ).flatten(start_dim=-2)
            for first, second in self.pairs
        )
        return torch.cat(blocks, dim=-1)

    def drift_and_gain(self, state):
        feature = self.features(state)
        baseline = (
            feature @ self.baseline_coefficient
        ) * self.delta_scale
        gain = torch.einsum(
            "...f,foa->...oa",
            feature,
            self.gain_coefficient,
        ) * self.delta_scale[..., None]
        return baseline, gain

    def forward(self, state, innovation):
        baseline, gain = self.drift_and_gain(state)
        return baseline + (
            gain @ innovation.unsqueeze(-1)
        ).squeeze(-1)

    def group_sparsity(self):
        penalties = []
        joint = torch.cat(
            (
                self.baseline_coefficient.unsqueeze(-1),
                self.gain_coefficient,
            ),
            dim=-1,
        )
        for group in self.group_slices[1:]:
            penalties.append(
                joint[group].square().sum(dim=0).add(1e-12).sqrt().mean()
            )
        return torch.stack(penalties).mean()

    @torch.no_grad()
    def active_group_fraction(self, threshold: float = 1e-3):
        joint = torch.cat(
            (
                self.baseline_coefficient.unsqueeze(-1),
                self.gain_coefficient,
            ),
            dim=-1,
        )
        norms = torch.stack([
            joint[group].square().sum(dim=0).sqrt().mean()
            for group in self.group_slices[1:]
        ])
        return float((norms > threshold).float().mean())
