"""Immutable local KAN charts with control-excitation bookkeeping."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import torch
from torch import nn

from kanrf import bspline_basis


def _pool(input_dim: int, output_dim: int) -> torch.Tensor:
    projection = torch.zeros(input_dim, output_dim)
    target = torch.linspace(0, output_dim - 1, input_dim).round().long()
    projection[torch.arange(input_dim), target] = 1.0
    return projection


@torch.no_grad()
def discover_chart_centers(
    state: torch.Tensor,
    chart_scale: torch.Tensor,
    *,
    coverage_radius: float = 1.0,
    max_charts: int = 64,
    initial_centers: torch.Tensor | None = None,
) -> torch.Tensor:
    """Greedily add a chart whenever all immutable charts are out of support."""
    centers = [] if initial_centers is None else list(initial_centers.unbind())
    for point in state:
        if not centers:
            centers.append(point.clone())
            continue
        existing = torch.stack(centers)
        distance = ((existing - point) / chart_scale).square().mean(dim=-1).sqrt()
        if distance.min() > coverage_radius:
            centers.append(point.clone())
            if len(centers) >= max_charts:
                break
    return torch.stack(centers)


class AdaptiveLocalKANDictionary(nn.Module):
    """Partition-of-unity atlas whose previously created charts never move."""

    def __init__(
        self,
        centers: torch.Tensor,
        *,
        chart_scale: tuple[float, ...] = (1.6, 1.6, 2.0, 2.0),
        grid_size: int = 4,
        spline_order: int = 2,
        pair_modes: int = 2,
        action_scale: float = 80.0,
    ):
        super().__init__()
        if centers.ndim != 2 or centers.shape[0] == 0:
            raise ValueError("centers must contain at least one state")
        self.state_dim = centers.shape[-1]
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.contrast_dim = grid_size + spline_order - 1
        self.pair_modes = pair_modes
        self.action_scale = action_scale
        self.register_buffer("centers", centers.detach().clone())
        self.register_buffer("chart_scale", torch.tensor(chart_scale, dtype=centers.dtype))
        self.register_buffer("grid", torch.linspace(-1.0, 1.0, grid_size + 1))
        self.register_buffer("pair_projection", _pool(self.contrast_dim, pair_modes))
        self.local_feature_dim = (
            1 + self.state_dim * self.contrast_dim
            + self.state_dim * (self.state_dim - 1) // 2 * pair_modes**2
        )
        self.feature_dim = self.num_charts * self.local_feature_dim

    @property
    def num_charts(self) -> int:
        return self.centers.shape[0]

    def normalized_coordinates(self, state: torch.Tensor) -> torch.Tensor:
        return (state[:, None, :] - self.centers[None, :, :]) / self.chart_scale

    def chart_weights(self, state: torch.Tensor) -> torch.Tensor:
        coordinate = self.normalized_coordinates(state)
        radius = coordinate.square().mean(dim=-1).sqrt()
        raw = (1.0 - radius).clamp_min(0.0).square()
        uncovered = raw.sum(dim=-1) <= 1e-12
        if uncovered.any():
            nearest = radius[uncovered].argmin(dim=-1)
            raw[uncovered] = torch.nn.functional.one_hot(
                nearest, self.num_charts,
            ).to(raw.dtype)
        return raw / raw.sum(dim=-1, keepdim=True)

    def local_features(self, state: torch.Tensor) -> torch.Tensor:
        coordinate = self.normalized_coordinates(state).clamp(-1.0, 1.0)
        basis = bspline_basis(
            coordinate.reshape(-1), self.grid, self.spline_order,
        ).reshape(*coordinate.shape, -1)[..., :-1]
        blocks = [torch.ones(
            *state.shape[:-1], self.num_charts, 1,
            device=state.device, dtype=state.dtype,
        ), basis.flatten(start_dim=-2)]
        compressed = basis @ self.pair_projection
        for first, second in combinations(range(self.state_dim), 2):
            interaction = torch.einsum(
                "nki,nkj->nkij", compressed[..., first, :], compressed[..., second, :],
            )
            blocks.append(interaction.flatten(start_dim=-2))
        return torch.cat(blocks, dim=-1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        weights = self.chart_weights(state)
        local = self.local_features(state)
        return (weights[..., None] * local).flatten(start_dim=-2)

    def context_features(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        features = self(state)
        scaled_action = action / self.action_scale
        return torch.cat((
            features,
            scaled_action[:, :1] * features,
            scaled_action[:, 1:] * features,
        ), dim=-1)

    @torch.no_grad()
    def expanded_with(
        self,
        state: torch.Tensor,
        *,
        coverage_radius: float = 1.0,
        max_charts: int = 64,
    ) -> "AdaptiveLocalKANDictionary":
        centers = discover_chart_centers(
            state, self.chart_scale,
            coverage_radius=coverage_radius,
            max_charts=max_charts,
            initial_centers=self.centers,
        )
        return AdaptiveLocalKANDictionary(
            centers,
            chart_scale=tuple(float(value) for value in self.chart_scale),
            grid_size=self.grid_size,
            spline_order=self.spline_order,
            pair_modes=self.pair_modes,
            action_scale=self.action_scale,
        ).to(self.centers.device)


@dataclass
class LocalControlExcitation:
    gram: torch.Tensor
    effective_samples: torch.Tensor

    @classmethod
    def empty(cls, atlas: AdaptiveLocalKANDictionary) -> "LocalControlExcitation":
        device, dtype = atlas.centers.device, atlas.centers.dtype
        return cls(
            torch.zeros(atlas.num_charts, 2, 2, device=device, dtype=dtype),
            torch.zeros(atlas.num_charts, device=device, dtype=dtype),
        )

    @torch.no_grad()
    def update(
        self,
        atlas: AdaptiveLocalKANDictionary,
        state: torch.Tensor,
        innovation: torch.Tensor,
    ) -> None:
        weight = atlas.chart_weights(state)
        outer = innovation[..., :, None] * innovation[..., None, :]
        self.gram.add_(torch.einsum("nk,nij->kij", weight, outer))
        self.effective_samples.add_(weight.sum(dim=0))

    def normalized_eigenvalues(self, ridge: float = 1e-8) -> torch.Tensor:
        normalized = self.gram / self.effective_samples.clamp_min(1.0)[:, None, None]
        identity = torch.eye(2, device=self.gram.device, dtype=self.gram.dtype)
        return torch.linalg.eigvalsh(normalized + ridge * identity)

    def identifiable_rank(self, threshold: float) -> torch.Tensor:
        return (self.normalized_eigenvalues() >= threshold).sum(dim=-1)
