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


class JointStateSupportCalibrator:
    """Source-only joint-state coverage and local-error calibrator.

    Coordinate-wise ranges cannot detect unseen combinations of individually
    familiar state coordinates.  This calibrator whitens the joint source
    cloud, measures k-nearest-neighbour coverage in that joint geometry, and
    attaches a source-only local counterfactual error estimate.
    """

    def __init__(
        self,
        reference_state,
        reference_error,
        state_scale,
        *,
        neighbors: int = 16,
        covariance_ridge: float = 0.05,
        chunk_size: int = 512,
    ):
        reference_state = torch.as_tensor(
            reference_state, dtype=torch.float32,
        )
        reference_error = torch.as_tensor(
            reference_error,
            dtype=torch.float32,
            device=reference_state.device,
        ).flatten()
        state_scale = torch.as_tensor(
            state_scale,
            dtype=torch.float32,
            device=reference_state.device,
        )
        if reference_state.ndim != 2:
            raise ValueError("reference_state must be a matrix")
        if reference_error.shape[0] != reference_state.shape[0]:
            raise ValueError("reference error count must match states")
        if reference_state.shape[0] <= neighbors:
            raise ValueError("more reference states than neighbors required")
        self.neighbors = int(neighbors)
        self.chunk_size = int(chunk_size)
        self.state_scale = state_scale.clamp_min(1e-6)
        normalized = reference_state / self.state_scale
        self.center = normalized.mean(dim=0)
        centered = normalized - self.center
        covariance = centered.T @ centered / max(centered.shape[0] - 1, 1)
        average_variance = covariance.diagonal().mean().clamp_min(1e-6)
        eigenvalue, eigenvector = torch.linalg.eigh(covariance)
        regularized = eigenvalue.clamp_min(
            float(covariance_ridge) * average_variance,
        )
        self.whitener = eigenvector @ torch.diag(regularized.rsqrt())
        self.reference = centered @ self.whitener
        self.reference_error = reference_error.clamp_min(0.0)
        calibration = self._neighbor_summary(
            self.reference,
            exclude_reference_self=True,
        )
        self.distance_scale = torch.quantile(
            calibration["coverage_distance"], 0.95,
        ).clamp_min(1e-6)
        self.error_scale = torch.quantile(
            self.reference_error, 0.95,
        ).clamp_min(1e-6)

    def _neighbor_summary(self, query, *, exclude_reference_self=False):
        distances = []
        local_errors = []
        count = query.shape[0]
        requested = self.neighbors + int(exclude_reference_self)
        for start in range(0, count, self.chunk_size):
            stop = min(start + self.chunk_size, count)
            distance = torch.cdist(query[start:stop], self.reference)
            if exclude_reference_self:
                rows = torch.arange(stop - start, device=distance.device)
                columns = torch.arange(start, stop, device=distance.device)
                distance[rows, columns] = torch.inf
            nearest_distance, nearest_index = torch.topk(
                distance,
                k=requested - int(exclude_reference_self),
                dim=-1,
                largest=False,
            )
            weight = nearest_distance.clamp_min(1e-4).reciprocal()
            neighbor_error = self.reference_error[nearest_index]
            local_error = (
                weight * neighbor_error
            ).sum(dim=-1) / weight.sum(dim=-1)
            distances.append(nearest_distance[:, -1])
            local_errors.append(local_error)
        return {
            "coverage_distance": torch.cat(distances),
            "local_source_error": torch.cat(local_errors),
        }

    def score(self, state):
        """Return source-only uncertainty components and confidence."""
        state = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=self.reference.device,
        )
        whitened = (
            state / self.state_scale - self.center
        ) @ self.whitener
        summary = self._neighbor_summary(whitened)
        coverage_ratio = (
            summary["coverage_distance"] / self.distance_scale
        )
        local_error_ratio = (
            summary["local_source_error"] / self.error_scale
        )
        # Coverage is the primary identifiability term.  Local source error
        # modulates it without allowing a fortuitously small local fit error
        # to declare a distributionally distant state safe.
        uncertainty = coverage_ratio * (
            0.5 + 0.5 * local_error_ratio.clamp_min(0.0)
        )
        confidence = torch.exp(-uncertainty)
        return {
            **summary,
            "coverage_ratio": coverage_ratio,
            "local_error_ratio": local_error_ratio,
            "uncertainty": uncertainty,
            "confidence": confidence,
        }
