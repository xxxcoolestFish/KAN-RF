"""Trajectory-global mechanism coordinates decoded by a state-dependent KAN."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from cpbn.generic_affine_kan import AffineKANContext


@dataclass
class GlobalMechanismKANDynamics:
    """A global latent whose effects are decoded through a KAN dictionary.

    The mechanism tensor has shape ``[latent, feature, output]``.  The latent
    is constant over a trajectory, while the KAN feature vector makes its
    physical effect depend on the current state and action.
    """

    source: AffineKANContext
    mechanisms: torch.Tensor

    @classmethod
    def from_contexts(
        cls,
        source: AffineKANContext,
        contexts: list[AffineKANContext],
        *,
        minimum_norm: float = 1e-8,
    ) -> "GlobalMechanismKANDynamics":
        directions = torch.stack(
            [
                context.coefficients - source.coefficients
                for context in contexts
            ],
            dim=0,
        )
        flat = directions.flatten(start_dim=1)
        norms = flat.norm(dim=1)
        keep = norms > minimum_norm
        if not bool(keep.any()):
            raise ValueError("no non-zero cognitive mechanism was observed")
        return cls(source, directions[keep])

    def context(self, latent: torch.Tensor) -> AffineKANContext:
        correction = torch.einsum(
            "k,kdo->do", latent, self.mechanisms,
        )
        return AffineKANContext(
            self.source.coefficients + correction,
        )

    def effect_features(
        self,
        basis,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        design = basis.context_features(state, action)
        return torch.einsum(
            "nd,kdo->nko", design, self.mechanisms,
        )

    @torch.no_grad()
    def whiten_effects(
        self,
        basis,
        state: torch.Tensor,
        action: torch.Tensor,
        delta_scale: torch.Tensor,
        *,
        floor: float = 1e-4,
    ) -> "GlobalMechanismKANDynamics":
        """Whiten directions in predictive function space, not parameter space."""
        effects = self.effect_features(basis, state, action)
        effects = effects / delta_scale[None, None, :]
        matrix = effects.permute(0, 2, 1).reshape(-1, effects.shape[1])
        gram = matrix.T @ matrix / matrix.shape[0]
        eigenvalues, eigenvectors = torch.linalg.eigh(gram)
        inverse_root = (
            eigenvectors
            @ torch.diag(eigenvalues.clamp_min(floor).rsqrt())
            @ eigenvectors.T
        )
        mechanisms = torch.einsum(
            "jk,jdo->kdo", inverse_root, self.mechanisms,
        )
        return GlobalMechanismKANDynamics(self.source, mechanisms)

    @torch.no_grad()
    def infer_latent(
        self,
        basis,
        state: torch.Tensor,
        action: torch.Tensor,
        next_delta: torch.Tensor,
        delta_scale: torch.Tensor,
        *,
        ridge: float = 1e-2,
    ) -> torch.Tensor:
        """Infer one trajectory-global latent by predictive least squares."""
        source_prediction = self.source.acceleration(
            basis, state, action,
        )
        target = (next_delta - source_prediction) / delta_scale
        effects = self.effect_features(basis, state, action)
        effects = effects / delta_scale[None, None, :]
        matrix = effects.permute(0, 2, 1).reshape(-1, effects.shape[1])
        vector = target.reshape(-1)
        identity = torch.eye(
            matrix.shape[1],
            dtype=matrix.dtype,
            device=matrix.device,
        )
        return torch.linalg.solve(
            matrix.T @ matrix + ridge * identity,
            matrix.T @ vector,
        )


class RecursiveGlobalMechanismEstimator:
    """Online least-squares posterior over reusable mechanism coordinates."""

    def __init__(
        self,
        model: GlobalMechanismKANDynamics,
        basis,
        delta_scale: torch.Tensor,
        *,
        ridge: float = 1e-2,
        forgetting_factor: float = 1.0,
    ):
        if not 0.0 < forgetting_factor <= 1.0:
            raise ValueError("forgetting_factor must lie in (0, 1]")
        self.model = model
        self.basis = basis
        self.delta_scale = delta_scale
        self.ridge = float(ridge)
        self.forgetting_factor = float(forgetting_factor)
        dimension = model.mechanisms.shape[0]
        self.precision = ridge * torch.eye(
            dimension,
            dtype=model.mechanisms.dtype,
            device=model.mechanisms.device,
        )
        self.right = torch.zeros(
            dimension,
            dtype=model.mechanisms.dtype,
            device=model.mechanisms.device,
        )
        self.update_count = 0

    @torch.no_grad()
    def update(
        self,
        state,
        action,
        next_delta,
        *,
        evidence_weight: float = 1.0,
    ):
        if evidence_weight < 0.0:
            raise ValueError("evidence_weight must be non-negative")
        source_prediction = self.model.source.acceleration(
            self.basis, state, action,
        )
        target = (
            (next_delta - source_prediction) / self.delta_scale
        ).reshape(-1)
        effects = self.model.effect_features(
            self.basis, state, action,
        )
        effects = effects / self.delta_scale[None, None, :]
        matrix = effects.permute(0, 2, 1).reshape(
            -1, effects.shape[1],
        )
        retention = self.forgetting_factor ** state.shape[0]
        if retention < 1.0:
            identity = torch.eye(
                self.precision.shape[0],
                dtype=self.precision.dtype,
                device=self.precision.device,
            )
            self.precision.mul_(retention).add_(
                self.ridge * identity,
                alpha=1.0 - retention,
            )
            self.right.mul_(retention)
        self.precision.add_(
            matrix.T @ matrix,
            alpha=float(evidence_weight),
        )
        self.right.add_(
            matrix.T @ target,
            alpha=float(evidence_weight),
        )
        self.update_count += state.shape[0]

    @torch.no_grad()
    def latent(self):
        return torch.linalg.solve(self.precision, self.right)

    @torch.no_grad()
    def context(self):
        return self.model.context(self.latent())
