"""Bayesian local-KAN cognition and posterior-risk control pullback."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from cpbn.chart_scaled_kan_atlas import RecursiveChartScaledEstimator


@dataclass
class BayesianChartScaledContext:
    """Posterior mean dynamics with chart-wise parameter covariance."""

    mean: object
    covariance: torch.Tensor

    def acceleration(self, atlas, state, action):
        return self.mean.acceleration(atlas, state, action)

    def drift_and_gain(self, atlas, state):
        return self.mean.drift_and_gain(atlas, state)

    def decode_action(self, atlas, state, virtual_action):
        return self.mean.decode_action(atlas, state, virtual_action)

    def gain_uncertainty(self, atlas, state):
        """Return posterior response variance for each action direction."""
        width = atlas.local_feature_dim
        local = atlas.local_features(state)
        weights = atlas.chart_weights(state)
        scaled = local[:, :, None, :] / atlas.action_scales[None, :, :, None]
        gain_covariance = self.covariance[:, width:, width:].reshape(
            atlas.num_charts, 2, width, 2, width,
        )
        uncertainty = torch.einsum(
            "nkpi,kpiqj,nkqj,nk->npq",
            scaled.to(gain_covariance.dtype),
            gain_covariance,
            scaled.to(gain_covariance.dtype),
            weights.square().to(gain_covariance.dtype),
        )
        output_dimension = self.mean.coefficients.shape[-1]
        return (output_dimension * uncertainty).to(state.dtype)


class BayesianRecursiveChartScaledEstimator(RecursiveChartScaledEstimator):
    """Exact recursive sufficient statistics interpreted as a posterior."""

    @torch.no_grad()
    def posterior(self) -> BayesianChartScaledContext:
        covariance = torch.linalg.inv(self.precision)
        return BayesianChartScaledContext(self.context(), covariance)


def posterior_risk_pullback(
    gain: torch.Tensor,
    target_response: torch.Tensor,
    source_action: torch.Tensor,
    risk: torch.Tensor,
    effect_metric: torch.Tensor | None = None,
) -> torch.Tensor:
    """Match target response while anchoring uncertain action directions."""
    if effect_metric is None:
        weighted_gain = gain
        weighted_response = target_response
    else:
        weighted_gain = effect_metric @ gain
        weighted_response = (
            effect_metric @ target_response.unsqueeze(-1)
        ).squeeze(-1)
    normal = gain.transpose(-1, -2) @ weighted_gain + risk
    right = (
        gain.transpose(-1, -2) @ weighted_response.unsqueeze(-1)
        + risk @ source_action.unsqueeze(-1)
    )
    return torch.linalg.solve(normal, right).squeeze(-1)


@dataclass
class PosteriorRiskPullbackContext:
    """Mandatory cognition-to-decision interface using posterior risk."""

    posterior: BayesianChartScaledContext
    source_prior: object
    risk_weight: float = 1.0
    risk_floor: float = 1e-6

    def acceleration(self, atlas, state, action):
        return self.posterior.acceleration(atlas, state, action)

    def drift_and_gain(self, atlas, state):
        return self.posterior.drift_and_gain(atlas, state)

    def risk_matrix(self, atlas, state):
        uncertainty = self.posterior.gain_uncertainty(atlas, state)
        identity = torch.eye(
            uncertainty.shape[-1], device=state.device, dtype=state.dtype,
        ).expand_as(uncertainty)
        return self.risk_weight * uncertainty + self.risk_floor * identity

    def decode_action(self, atlas, state, virtual_action):
        drift, gain = self.drift_and_gain(atlas, state)
        source_action = self.source_prior.decode_action(atlas, state, virtual_action)
        return posterior_risk_pullback(
            gain,
            virtual_action - drift,
            source_action,
            self.risk_matrix(atlas, state),
        )

    def pullback_diagnostics(self, atlas, state):
        risk = self.risk_matrix(atlas, state)
        eigenvalues = torch.linalg.eigvalsh(risk)
        return {
            "risk_eigenvalue_median": float(eigenvalues.median()),
            "risk_eigenvalue_p95": float(eigenvalues.quantile(0.95)),
            "risk_anisotropy_p95": float(
                (eigenvalues[:, -1] / eigenvalues[:, 0].clamp_min(1e-12)).quantile(0.95)
            ),
        }
