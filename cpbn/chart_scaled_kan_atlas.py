"""Adaptive KAN charts with immutable chart-wise action coordinates."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from cpbn.adaptive_local_kan_atlas import AdaptiveLocalKANDictionary


class ChartScaledLocalKANDictionary(AdaptiveLocalKANDictionary):
    """Use a fixed two-dimensional action scale in every immutable chart."""

    def __init__(self, centers: torch.Tensor, action_scales: torch.Tensor, **kwargs):
        super().__init__(centers, action_scale=1.0, **kwargs)
        if action_scales.shape != (self.num_charts, 2):
            raise ValueError("action_scales must have shape (num_charts, 2)")
        if (action_scales <= 0.0).any():
            raise ValueError("action scales must be positive")
        self.register_buffer("action_scales", action_scales.detach().clone())

    def weighted_local_features(self, state: torch.Tensor) -> torch.Tensor:
        return self(state).reshape(-1, self.num_charts, self.local_feature_dim)

    def context_features(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        features = self.weighted_local_features(state)
        blocks = [features]
        for index in range(2):
            normalized = action[:, index, None] / self.action_scales[:, index]
            blocks.append(normalized[..., None] * features)
        return torch.cat(tuple(block.flatten(start_dim=-2) for block in blocks), dim=-1)


@dataclass
class ChartScaledLocalKANContext:
    coefficients: torch.Tensor

    @classmethod
    def neutral(cls, atlas: ChartScaledLocalKANDictionary):
        coefficients = torch.zeros(
            3 * atlas.feature_dim, 2,
            device=atlas.centers.device, dtype=atlas.centers.dtype,
        )
        width = atlas.local_feature_dim
        for chart in range(atlas.num_charts):
            offset = chart * width
            coefficients[atlas.feature_dim + offset, 0] = atlas.action_scales[chart, 0]
            coefficients[2 * atlas.feature_dim + offset, 1] = atlas.action_scales[chart, 1]
        return cls(coefficients)

    def acceleration(self, atlas, state, action):
        return atlas.context_features(state, action) @ self.coefficients

    def drift_and_gain(self, atlas, state):
        features = atlas.weighted_local_features(state)
        width = atlas.local_feature_dim
        blocks = self.coefficients.reshape(3, atlas.num_charts, width, 2)
        drift = torch.einsum("nkl,klo->no", features, blocks[0])
        gains = []
        for index in range(2):
            scaled = blocks[index + 1] / atlas.action_scales[:, index, None, None]
            gains.append(torch.einsum("nkl,klo->no", features, scaled))
        return drift, torch.stack(gains, dim=-1)

    def decode_action(self, atlas, state, virtual_action, singular_floor: float = 1e-3):
        drift, gain = self.drift_and_gain(atlas, state)
        left, singular, right_h = torch.linalg.svd(gain)
        inverse_singular = singular / (singular.square() + singular_floor**2)
        inverse = (
            right_h.transpose(-1, -2)
            @ torch.diag_embed(inverse_singular)
            @ left.transpose(-1, -2)
        )
        return (inverse @ (virtual_action - drift).unsqueeze(-1)).squeeze(-1)


def _context_blocks(context, atlas):
    return context.coefficients.reshape(
        3, atlas.num_charts, atlas.local_feature_dim, 2,
    ).permute(1, 0, 2, 3).reshape(atlas.num_charts, -1, 2)


def _assemble_context(blocks, atlas):
    coefficients = blocks.reshape(
        atlas.num_charts, 3, atlas.local_feature_dim, 2,
    ).permute(1, 0, 2, 3).reshape(3 * atlas.feature_dim, 2)
    return ChartScaledLocalKANContext(coefficients)


def _local_design(atlas, local, action, chart):
    return torch.cat((
        local,
        action[:, :1] / atlas.action_scales[chart, 0] * local,
        action[:, 1:] / atlas.action_scales[chart, 1] * local,
    ), dim=-1)


@torch.no_grad()
def fit_chart_scaled_context(
    atlas: ChartScaledLocalKANDictionary,
    state: torch.Tensor,
    action: torch.Tensor,
    next_state: torch.Tensor,
    prior: ChartScaledLocalKANContext | None = None,
    *,
    ridge: float = 0.1,
    dt: float = 0.02,
) -> ChartScaledLocalKANContext:
    prior = prior or ChartScaledLocalKANContext.neutral(atlas)
    accumulator_dtype = torch.float64
    prior_blocks = _context_blocks(prior, atlas).to(accumulator_dtype)
    weights = atlas.chart_weights(state)
    local = atlas.local_features(state)
    target = (next_state[:, 2:] - state[:, 2:]) / dt
    fitted = []
    for chart in range(atlas.num_charts):
        design = _local_design(atlas, local[:, chart], action, chart)
        root_weight = weights[:, chart:chart + 1].sqrt()
        weighted_design = (root_weight * design).to(accumulator_dtype)
        weighted_target = (root_weight * target).to(accumulator_dtype)
        identity = torch.eye(design.shape[-1], device=state.device, dtype=accumulator_dtype)
        fitted.append(torch.linalg.solve(
            weighted_design.T @ weighted_design + ridge * identity,
            weighted_design.T @ weighted_target + ridge * prior_blocks[chart],
        ))
    return _assemble_context(torch.stack(fitted).to(state.dtype), atlas)


class RecursiveChartScaledEstimator:
    """Exact local sufficient statistics; no raw transition replay is required."""

    def __init__(
        self,
        atlas: ChartScaledLocalKANDictionary,
        prior: ChartScaledLocalKANContext,
        *,
        ridge: float = 0.1,
    ):
        self.atlas = atlas
        dimension = 3 * atlas.local_feature_dim
        self.accumulator_dtype = torch.float64
        identity = torch.eye(
            dimension, device=atlas.centers.device, dtype=self.accumulator_dtype,
        )
        self.precision = ridge * identity.expand(atlas.num_charts, -1, -1).clone()
        self.right = ridge * _context_blocks(prior, atlas).to(self.accumulator_dtype)

    @torch.no_grad()
    def update(self, state, action, next_state, *, dt: float = 0.02):
        weights = self.atlas.chart_weights(state)
        local = self.atlas.local_features(state)
        target = (next_state[:, 2:] - state[:, 2:]) / dt
        for chart in range(self.atlas.num_charts):
            design = _local_design(self.atlas, local[:, chart], action, chart)
            root_weight = weights[:, chart:chart + 1].sqrt()
            weighted_design = (root_weight * design).to(self.accumulator_dtype)
            weighted_target = (root_weight * target).to(self.accumulator_dtype)
            self.precision[chart].add_(weighted_design.T @ weighted_design)
            self.right[chart].add_(weighted_design.T @ weighted_target)

    @torch.no_grad()
    def context(self) -> ChartScaledLocalKANContext:
        blocks = torch.linalg.solve(self.precision, self.right)
        return _assemble_context(blocks.to(self.atlas.centers.dtype), self.atlas)


@torch.no_grad()
def distill_teacher_to_atlas(
    atlas: ChartScaledLocalKANDictionary,
    teacher_basis,
    teacher_context,
    *,
    samples_per_chart: int = 1024,
    ridge: float = 1e-3,
    dt: float = 0.02,
) -> ChartScaledLocalKANContext:
    states, actions = [], []
    for chart in range(atlas.num_charts):
        state = atlas.centers[chart] + (
            1.6 * torch.rand(
                samples_per_chart, atlas.state_dim,
                device=atlas.centers.device, dtype=atlas.centers.dtype,
            ) - 0.8
        ) * atlas.chart_scale
        action = (
            2.0 * torch.rand(samples_per_chart, 2, device=state.device) - 1.0
        ) * atlas.action_scales[chart]
        states.append(state)
        actions.append(action)
    state = torch.cat(states)
    action = torch.cat(actions)
    acceleration = teacher_context.acceleration(teacher_basis, state, action)
    next_state = state.clone()
    next_state[:, :2] += dt * state[:, 2:]
    next_state[:, 2:] += dt * acceleration
    return fit_chart_scaled_context(
        atlas, state, action, next_state, ridge=ridge, dt=dt,
    )
