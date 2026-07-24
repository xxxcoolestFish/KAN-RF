"""A state-dependent closed-loop policy dictionary gated by cognition."""

from __future__ import annotations

import torch
from torch import nn


class PolicyMechanismDecoder(nn.Module):
    """Decode global cognitive coordinates into state-feedback corrections."""

    def __init__(
        self,
        observation_mean,
        observation_variance,
        *,
        state_dim: int = 11,
        action_dim: int = 3,
        mechanism_dim: int = 3,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.register_buffer(
            "observation_mean",
            torch.as_tensor(observation_mean, dtype=torch.float32),
        )
        self.register_buffer(
            "observation_variance",
            torch.as_tensor(observation_variance, dtype=torch.float32),
        )
        self.mechanisms = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(state_dim, hidden_dim),
                    nn.Tanh(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.Tanh(),
                    nn.Linear(hidden_dim, action_dim),
                )
                for _ in range(mechanism_dim)
            ]
        )

    def normalized_state(self, state):
        return (
            (state - self.observation_mean)
            / (self.observation_variance + 1e-8).sqrt()
        ).clamp(-10.0, 10.0)

    def mechanism_effects(self, state):
        normalized = self.normalized_state(state)
        return torch.stack(
            [mechanism(normalized) for mechanism in self.mechanisms],
            dim=1,
        )

    def forward(self, state, coordinates):
        effects = self.mechanism_effects(state)
        return torch.einsum("bk,bka->ba", coordinates, effects)

