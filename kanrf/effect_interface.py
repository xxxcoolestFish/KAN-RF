"""Core modules for a task-controllable effect interface.

The dynamics model is trained only by transition prediction. Task losses may
train the effect encoder and router, but must not update the dynamics model.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from kanrf._protokan import ProtoKAN


def _mlp(dims: list[int], activation: type[nn.Module] = nn.SiLU) -> nn.Sequential:
    layers: list[nn.Module] = []
    for in_dim, out_dim in zip(dims[:-2], dims[1:-1]):
        layers.extend((nn.Linear(in_dim, out_dim), activation()))
    layers.append(nn.Linear(dims[-2], dims[-1]))
    return nn.Sequential(*layers)


class EffectEncoder(nn.Module):
    """Maps full observations into a normalized, low-dimensional effect space."""

    def __init__(self, obs_dim: int, effect_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.effect_dim = effect_dim
        self.net = _mlp([obs_dim, hidden_dim, hidden_dim, effect_dim])

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.net(states)


class TaskEffectValue(nn.Module):
    """Low-dimensional task effect whose readout approximates source value."""

    def __init__(
        self,
        obs_dim: int,
        effect_dim: int,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.effect_dim = effect_dim
        self.hidden_dim = hidden_dim
        self.encoder = EffectEncoder(obs_dim, effect_dim, hidden_dim)
        self.value_head = _mlp([effect_dim, hidden_dim, hidden_dim, 1])
        self.register_buffer("obs_mean", torch.zeros(obs_dim))
        self.register_buffer("obs_std", torch.ones(obs_dim))
        self.register_buffer("value_mean", torch.zeros(()))
        self.register_buffer("value_std", torch.ones(()))

    def set_normalization(
        self,
        observations: torch.Tensor,
        values: torch.Tensor,
        eps: float = 1e-6,
    ) -> None:
        """Store training-set normalization inside the model."""
        with torch.no_grad():
            self.obs_mean.copy_(observations.mean(dim=0))
            self.obs_std.copy_(
                observations.std(dim=0, unbiased=False).clamp_min(eps)
            )
            self.value_mean.copy_(values.mean())
            self.value_std.copy_(
                values.std(unbiased=False).clamp_min(eps)
            )

    def encode(self, states: torch.Tensor) -> torch.Tensor:
        normalized = (states - self.obs_mean) / self.obs_std
        return self.encoder(normalized)

    def normalized_value_from_effects(
        self,
        effects: torch.Tensor,
    ) -> torch.Tensor:
        return self.value_head(effects).squeeze(-1)

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        effects = self.encode(states)
        normalized_values = self.normalized_value_from_effects(effects)
        values = normalized_values * self.value_std + self.value_mean
        return effects, values


class MLPDynamics(nn.Module):
    """Residual one-step dynamics baseline."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.obs_dim = obs_dim
        self.net = _mlp(
            [obs_dim + action_dim, hidden_dim, hidden_dim, obs_dim]
        )

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        delta = self.net(torch.cat((states, actions), dim=-1))
        return states + delta


class ProtoKANDynamics(nn.Module):
    """Residual ProtoKAN dynamics model with the same public interface as MLP."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        n_prototypes: int = 12,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.net = ProtoKAN(
            [obs_dim + action_dim, hidden_dim, obs_dim],
            n_prototypes=n_prototypes,
        )

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        delta = self.net(torch.cat((states, actions), dim=-1))
        return states + delta


def effect_action_jacobian(
    encoder: nn.Module,
    dynamics: nn.Module,
    states: torch.Tensor,
    actions: torch.Tensor,
    create_graph: bool = True,
) -> torch.Tensor:
    """Return d psi(F(s,a)) / d a with shape [B, effect_dim, action_dim]."""
    if states.ndim != 2 or actions.ndim != 2:
        raise ValueError("states and actions must be rank-2 batches")
    actions = actions.requires_grad_(True)
    effects = encoder(dynamics(states, actions))
    rows = []
    for index in range(effects.shape[-1]):
        gradient = torch.autograd.grad(
            effects[:, index].sum(),
            actions,
            create_graph=create_graph,
            retain_graph=True,
        )[0]
        rows.append(gradient)
    return torch.stack(rows, dim=1)


@dataclass(frozen=True)
class ControllabilityStats:
    loss: torch.Tensor
    min_singular_value: torch.Tensor
    max_singular_value: torch.Tensor
    condition_number: torch.Tensor


def controllability_loss(
    jacobian: torch.Tensor,
    eps: float = 1e-4,
) -> ControllabilityStats:
    """Penalize locally uncontrollable or ill-conditioned effect coordinates.

    The Jacobian is row-normalized before computing the Gram matrix, preventing
    the encoder from winning merely by scaling its output.
    """
    normalized = F.normalize(jacobian, p=2, dim=-1, eps=eps)
    gram = normalized @ normalized.transpose(-1, -2)
    eye = torch.eye(
        gram.shape[-1], device=gram.device, dtype=gram.dtype
    ).expand_as(gram)
    regularized = gram + eps * eye
    singular_values = torch.linalg.svdvals(regularized)
    minimum = singular_values[..., -1]
    maximum = singular_values[..., 0]
    condition = maximum / minimum.clamp_min(eps)
    loss = (-torch.logdet(regularized) + torch.log(condition)).mean()
    return ControllabilityStats(
        loss=loss,
        min_singular_value=minimum.mean(),
        max_singular_value=maximum.mean(),
        condition_number=condition.mean(),
    )


def effect_covariance_loss(effects: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """VICReg-style variance/covariance term that prevents latent collapse."""
    centered = effects - effects.mean(dim=0, keepdim=True)
    std = torch.sqrt(centered.var(dim=0, unbiased=False) + eps)
    variance_loss = F.relu(1.0 - std).mean()
    covariance = centered.T @ centered / max(effects.shape[0] - 1, 1)
    off_diagonal = covariance - torch.diag(torch.diagonal(covariance))
    return variance_loss + off_diagonal.pow(2).mean()
