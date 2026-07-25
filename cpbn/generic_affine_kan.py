"""Dimension-independent compact interaction KAN for control-affine dynamics."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import torch
from torch import nn

from kanrf import bspline_basis
from cpbn.bayesian_recursive_kan_pullback import posterior_risk_pullback


class CompactInteractionKANDictionary(nn.Module):
    def __init__(
        self,
        state_scale,
        action_scale,
        *,
        grid_size: int = 4,
        spline_order: int = 2,
        pair_modes: int = 3,
    ):
        super().__init__()
        state_scale = torch.as_tensor(state_scale, dtype=torch.float32)
        action_scale = torch.as_tensor(action_scale, dtype=torch.float32)
        self.state_dim = state_scale.numel()
        self.action_dim = action_scale.numel()
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.contrast_dim = grid_size + spline_order - 1
        self.pair_modes = min(pair_modes, self.contrast_dim)
        self.register_buffer("state_scale", state_scale)
        self.register_buffer("action_scale", action_scale)
        self.register_buffer("grid", torch.linspace(-1.0, 1.0, grid_size + 1))
        index = torch.arange(self.contrast_dim, dtype=torch.float32)[:, None]
        mode = torch.arange(self.pair_modes, dtype=torch.float32)[None, :]
        projection = torch.cos(
            torch.pi * (index + 0.5) * mode / self.contrast_dim,
        )
        self.register_buffer(
            "pair_projection", torch.linalg.qr(projection, mode="reduced").Q,
        )
        self.feature_dim = (
            1
            + self.state_dim * self.contrast_dim
            + len(tuple(combinations(range(self.state_dim), 2)))
            * self.pair_modes**2
        )

    def forward(self, state):
        normalized = (state / self.state_scale).clamp(-1.0, 1.0)
        local = bspline_basis(
            normalized.reshape(-1), self.grid, self.spline_order,
        ).reshape(*state.shape, -1)[..., :-1]
        blocks = [local.flatten(start_dim=-2)]
        compressed = local @ self.pair_projection
        for first, second in combinations(range(self.state_dim), 2):
            blocks.append(torch.einsum(
                "...i,...j->...ij",
                compressed[..., first, :],
                compressed[..., second, :],
            ).flatten(start_dim=-2))
        constant = torch.ones(*state.shape[:-1], 1, device=state.device)
        return torch.cat((constant, *blocks), dim=-1)

    def context_features(self, state, action):
        basis = self(state)
        blocks = [basis]
        for index in range(self.action_dim):
            blocks.append(
                action[..., index:index + 1] / self.action_scale[index] * basis,
            )
        return torch.cat(blocks, dim=-1)

    def build_smoothness_matrix(self, dtype=None, device=None):
        """1-D Laplacian smoothness prior for each per-dimension B-spline block.

        The penalty  Σ_i (w_{i+1} - w_i)^2  is encoded as w^T L w with
        Neumann-boundary Laplacians on every state-dimension B-spline group.
        """
        if dtype is None:
            dtype = self.grid.dtype
        if device is None:
            device = self.grid.device
        L = torch.zeros(self.feature_dim, self.feature_dim,
                        dtype=dtype, device=device)
        for dim_idx in range(self.state_dim):
            start = 1 + dim_idx * self.contrast_dim
            end = start + self.contrast_dim
            size = self.contrast_dim
            block = torch.zeros(size, size, dtype=dtype, device=device)
            for i in range(size):
                if i == 0:
                    block[i, i] = 1.0
                    block[i, i + 1] = -1.0
                elif i == size - 1:
                    block[i, i] = 1.0
                    block[i, i - 1] = -1.0
                else:
                    block[i, i] = 2.0
                    block[i, i - 1] = -1.0
                    block[i, i + 1] = -1.0
            L[start:end, start:end] = block
        return L


class LearnedMLPDictionary(nn.Module):
    """A conventional learned state dictionary for fair cognition ablations.

    It preserves the same control-affine linear context interface as the KAN
    dictionary.  Consequently, online inference and mechanism transport remain
    identical; only the state feature family changes.
    """

    def __init__(
        self,
        state_scale,
        action_scale,
        *,
        feature_dim: int,
        hidden_dim: int = 64,
    ):
        super().__init__()
        if feature_dim < 2:
            raise ValueError("feature_dim must include a constant and features")
        state_scale = torch.as_tensor(state_scale, dtype=torch.float32)
        action_scale = torch.as_tensor(action_scale, dtype=torch.float32)
        self.state_dim = state_scale.numel()
        self.action_dim = action_scale.numel()
        self.feature_dim = int(feature_dim)
        self.register_buffer("state_scale", state_scale)
        self.register_buffer("action_scale", action_scale)
        # RecursiveAffineKANEstimator only needs a device/dtype anchor.
        self.register_buffer("grid", torch.zeros(1))
        self.network = nn.Sequential(
            nn.Linear(self.state_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.feature_dim - 1),
            nn.Tanh(),
        )

    def forward(self, state):
        normalized = (state / self.state_scale).clamp(-10.0, 10.0)
        learned = self.network(normalized)
        constant = torch.ones(*state.shape[:-1], 1, device=state.device)
        return torch.cat((constant, learned), dim=-1)

    def context_features(self, state, action):
        basis = self(state)
        blocks = [basis]
        for index in range(self.action_dim):
            blocks.append(
                action[..., index:index + 1] / self.action_scale[index] * basis,
            )
        return torch.cat(blocks, dim=-1)


@dataclass
class AffineKANContext:
    coefficients: torch.Tensor

    def acceleration(self, basis, state, action):
        return basis.context_features(state, action) @ self.coefficients

    def drift_and_gain(self, basis, state):
        features = basis(state)
        width = basis.feature_dim
        blocks = self.coefficients.reshape(
            1 + basis.action_dim, width, -1,
        )
        drift = features @ blocks[0]
        gains = [
            features @ blocks[index + 1] / basis.action_scale[index]
            for index in range(basis.action_dim)
        ]
        return drift, torch.stack(gains, dim=-1)

    def decode_action(self, basis, state, virtual_action, damping: float = 1e-3):
        drift, gain = self.drift_and_gain(basis, state)
        left, singular, right_h = torch.linalg.svd(gain, full_matrices=False)
        inverse_singular = singular / (singular.square() + damping**2)
        inverse = (
            right_h.transpose(-1, -2)
            @ torch.diag_embed(inverse_singular)
            @ left.transpose(-1, -2)
        )
        return (inverse @ (virtual_action - drift).unsqueeze(-1)).squeeze(-1)

    def transport_action(
        self,
        basis,
        state,
        desired_effect,
        nominal_action,
        *,
        regularization: float = 1e-2,
        effect_metric: torch.Tensor | None = None,
    ):
        """Match an effect while staying close to a nominal source action.

        This solves

            min_u ||G u + f - desired_effect||_W^2
                  + regularization ||u - nominal_action||_2^2.

        Unlike ``decode_action``, the underdetermined inverse is anchored at
        the source policy action instead of at zero.  That distinction is
        important when the learned dynamics expose many state effects but only
        a few control dimensions.
        """
        if regularization <= 0.0:
            raise ValueError("regularization must be positive")
        drift, gain = self.drift_and_gain(basis, state)
        residual = desired_effect - (
            drift
            + (gain @ nominal_action.unsqueeze(-1)).squeeze(-1)
        )
        if effect_metric is None:
            weighted_gain = gain
            weighted_residual = residual
        else:
            metric = torch.as_tensor(
                effect_metric,
                dtype=gain.dtype,
                device=gain.device,
            )
            weighted_gain = metric @ gain
            weighted_residual = (
                metric @ residual.unsqueeze(-1)
            ).squeeze(-1)
        system = (
            gain.transpose(-1, -2) @ weighted_gain
            + regularization
            * torch.eye(
                gain.shape[-1],
                dtype=gain.dtype,
                device=gain.device,
            )
        )
        right = (
            gain.transpose(-1, -2)
            @ weighted_residual.unsqueeze(-1)
        )
        correction = torch.linalg.solve(system, right).squeeze(-1)
        return nominal_action + correction


@torch.no_grad()
def fit_affine_kan_context(
    basis,
    state,
    action,
    acceleration,
    *,
    ridge: float = 1e-2,
):
    design = basis.context_features(state, action)
    identity = torch.eye(design.shape[-1], device=state.device, dtype=state.dtype)
    coefficients = torch.linalg.solve(
        design.T @ design + ridge * identity,
        design.T @ acceleration,
    )
    return AffineKANContext(coefficients)


class RecursiveAffineKANEstimator:
    def __init__(
        self,
        basis,
        prior,
        *,
        ridge: float = 0.1,
        forgetting_factor: float = 1.0,
    ):
        if not 0.0 < forgetting_factor <= 1.0:
            raise ValueError("forgetting_factor must lie in (0, 1]")
        self.basis = basis
        self.ridge = ridge
        self.forgetting_factor = float(forgetting_factor)
        dimension = (1 + basis.action_dim) * basis.feature_dim
        identity = torch.eye(dimension, device=basis.grid.device, dtype=torch.float64)
        self.base_precision = ridge * identity
        self.base_right = ridge * prior.coefficients.to(torch.float64)
        self.precision = self.base_precision.clone()
        self.right = self.base_right.clone()

    @torch.no_grad()
    def update(
        self,
        state,
        action,
        acceleration,
        *,
        forgetting_factor: float | None = None,
    ):
        design = self.basis.context_features(state, action).to(torch.float64)
        target = acceleration.to(torch.float64)
        factor = (
            self.forgetting_factor
            if forgetting_factor is None
            else float(forgetting_factor)
        )
        if not 0.0 < factor <= 1.0:
            raise ValueError("forgetting_factor must lie in (0, 1]")
        retention = factor ** design.shape[0]
        if retention < 1.0:
            self.precision.mul_(retention).add_(
                self.base_precision, alpha=1.0 - retention,
            )
            self.right.mul_(retention).add_(
                self.base_right, alpha=1.0 - retention,
            )
        self.precision.add_(design.T @ design)
        self.right.add_(design.T @ target)

    @torch.no_grad()
    def context(self):
        return AffineKANContext(
            torch.linalg.solve(self.precision, self.right).to(self.basis.grid.dtype),
        )

    @torch.no_grad()
    def posterior(self):
        return AffineKANPosterior(self.context(), torch.linalg.inv(self.precision))


@dataclass
class AffineKANPosterior:
    mean: AffineKANContext
    covariance: torch.Tensor

    def acceleration(self, basis, state, action):
        return self.mean.acceleration(basis, state, action)

    def drift_and_gain(self, basis, state):
        return self.mean.drift_and_gain(basis, state)

    def gain_uncertainty(self, basis, state):
        features = basis(state).to(self.covariance.dtype)
        width = basis.feature_dim
        action_covariance = self.covariance[width:, width:].reshape(
            basis.action_dim, width, basis.action_dim, width,
        )
        scaled = features[:, None, :] / basis.action_scale[None, :, None]
        uncertainty = torch.einsum(
            "npi,piqj,nqj->npq", scaled, action_covariance, scaled,
        )
        output_dim = self.mean.coefficients.shape[-1]
        return (output_dim * uncertainty).to(state.dtype)


@dataclass
class AffinePosteriorPullback:
    posterior: AffineKANPosterior
    source: AffineKANContext
    risk_weight: float = 1.0
    risk_floor: float = 1e-5
    effect_metric: torch.Tensor | None = None

    def risk_matrix(self, basis, state):
        risk = self.risk_weight * self.posterior.gain_uncertainty(basis, state)
        identity = torch.eye(
            basis.action_dim, device=state.device, dtype=state.dtype,
        ).expand_as(risk)
        return risk + self.risk_floor * identity

    def decode_action(self, basis, state, virtual_action):
        drift, gain = self.posterior.drift_and_gain(basis, state)
        source_action = self.source.decode_action(basis, state, virtual_action)
        effect_metric = (
            self.effect_metric(state, virtual_action)
            if callable(self.effect_metric)
            else self.effect_metric
        )
        return posterior_risk_pullback(
            gain,
            virtual_action - drift,
            source_action,
            self.risk_matrix(basis, state),
            effect_metric=effect_metric,
        )
