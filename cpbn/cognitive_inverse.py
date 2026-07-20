"""Regularized multi-step cognitive inverse used as a direct policy layer."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal

from cpbn.cognitive_pullback import cognitive_tokens


def action_to_future_sensitivities(
    state_jacobians: torch.Tensor,
    action_jacobians: torch.Tensor,
) -> torch.Tensor:
    """Build d(x_{t+k})/d(a_t) by chaining future state Jacobians."""
    sensitivity = action_jacobians[:, 0]
    sequence = [sensitivity]
    for step in range(1, state_jacobians.shape[1]):
        sensitivity = torch.bmm(state_jacobians[:, step], sensitivity)
        sequence.append(sensitivity)
    return torch.stack(sequence, dim=1)


def regularized_cognitive_inverse(
    desired_effect: torch.Tensor,
    effect_weight: torch.Tensor,
    state_jacobians: torch.Tensor,
    action_jacobians: torch.Tensor,
    ridge: torch.Tensor,
) -> torch.Tensor:
    """Solve a weighted one-action, multi-step ridge least-squares problem."""
    sensitivity = action_to_future_sensitivities(
        state_jacobians, action_jacobians,
    ).squeeze(-1)
    numerator = (
        sensitivity * effect_weight * desired_effect
    ).sum(dim=(1, 2), keepdim=False).unsqueeze(-1)
    denominator = (
        sensitivity.square() * effect_weight
    ).sum(dim=(1, 2), keepdim=False).unsqueeze(-1)
    return numerator / (denominator + ridge)


class CognitiveInverseEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 64, effect_scale: float = 0.02):
        super().__init__()
        self.gru = nn.GRU(31, hidden_dim, batch_first=True)
        self.effect_head = nn.Linear(hidden_dim, 4)
        self.weight_head = nn.Linear(hidden_dim, 4)
        self.summary = nn.Linear(hidden_dim, hidden_dim)
        self.effect_scale = effect_scale
        nn.init.normal_(self.effect_head.weight, std=0.01)
        nn.init.zeros_(self.effect_head.bias)

    def forward(self, state, corridor, state_jacobians, action_jacobians):
        tokens = cognitive_tokens(
            state, corridor, state_jacobians, action_jacobians,
        )
        encoded, hidden = self.gru(tokens)
        desired_effect = self.effect_scale * torch.tanh(
            self.effect_head(encoded),
        )
        effect_weight = F.softplus(self.weight_head(encoded)) + 0.05
        effect_weight = effect_weight / effect_weight.mean(
            dim=(1, 2), keepdim=True,
        )
        return (
            torch.tanh(self.summary(hidden.squeeze(0))),
            desired_effect,
            effect_weight,
        )


class CognitiveInverseActor(nn.Module):
    """Direct Actor whose only action path is a regularized cognitive inverse."""

    def __init__(
        self,
        hidden_dim: int = 64,
        log_std: float = 0.0,
        effect_scale: float = 0.02,
        ridge: float = 1e-4,
    ):
        super().__init__()
        self.encoder = CognitiveInverseEncoder(hidden_dim, effect_scale)
        self.ridge_modulation = nn.Linear(hidden_dim, 1)
        self.log_ridge = nn.Parameter(torch.tensor(math.log(ridge)))
        self.log_std = nn.Parameter(torch.tensor([log_std]))

    def mean(self, state, corridor, state_jacobians, action_jacobians):
        summary, desired_effect, effect_weight = self.encoder(
            state, corridor, state_jacobians, action_jacobians,
        )
        log_ridge = self.log_ridge + 0.5 * torch.tanh(
            self.ridge_modulation(summary),
        )
        ridge = log_ridge.clamp(-12.0, -4.0).exp()
        return regularized_cognitive_inverse(
            desired_effect, effect_weight,
            state_jacobians, action_jacobians, ridge,
        )

    def distribution(self, state, corridor, state_jacobians, action_jacobians):
        mean = self.mean(state, corridor, state_jacobians, action_jacobians)
        std = self.log_std.clamp(-5.0, 1.0).exp().expand_as(mean)
        return Normal(mean, std)

    @staticmethod
    def log_prob(distribution, raw, action):
        return (
            distribution.log_prob(raw)
            - torch.log(1.0 - action.square() + 1e-6)
        ).sum(dim=-1)

    def sample(
        self, state, corridor, state_jacobians, action_jacobians,
        deterministic=False,
    ):
        distribution = self.distribution(
            state, corridor, state_jacobians, action_jacobians,
        )
        raw = distribution.mean if deterministic else distribution.rsample()
        action = torch.tanh(raw)
        return action, self.log_prob(distribution, raw, action)

    def evaluate(self, state, corridor, state_jacobians, action_jacobians, action):
        distribution = self.distribution(
            state, corridor, state_jacobians, action_jacobians,
        )
        action = action.clamp(-0.999999, 0.999999)
        raw = torch.atanh(action)
        return self.log_prob(distribution, raw, action), distribution.entropy().sum(-1)


class CognitiveInverseCritic(nn.Module):
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.encoder = CognitiveInverseEncoder(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1),
        )

    def forward(self, state, corridor, state_jacobians, action_jacobians):
        summary, _, _ = self.encoder(
            state, corridor, state_jacobians, action_jacobians,
        )
        return self.head(summary).squeeze(-1)
