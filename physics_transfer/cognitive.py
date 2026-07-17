"""Cognitive network: dynamics prediction plus canonical physics encoding."""

from __future__ import annotations

import torch
from torch import nn

from kanrf._protokan import ProtoKAN


class CognitivePredictor(nn.Module):
    """Encode recent interaction history and predict the next state."""

    def __init__(self, state_dim: int, action_dim: int, history_dim: int,
                 physics_dim: int = 1, hidden_dim: int = 32,
                 n_prototypes: int = 16):
        super().__init__()
        self.encoder = ProtoKAN([history_dim, hidden_dim, physics_dim],
                                n_prototypes=n_prototypes)
        self.dynamics = ProtoKAN(
            [state_dim + action_dim + physics_dim, hidden_dim, state_dim],
            n_prototypes=n_prototypes)

    def encode(self, history: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.encoder(history))

    def predict(self, state: torch.Tensor, action: torch.Tensor,
                physics: torch.Tensor) -> torch.Tensor:
        return self.dynamics(torch.cat([state, action, physics], dim=-1))

    def forward(self, history: torch.Tensor, state: torch.Tensor,
                action: torch.Tensor | None = None):
        physics = self.encode(history)
        next_state = None if action is None else self.predict(state, action, physics)
        return physics, next_state


def semantic_consistency_loss(physics_a: torch.Tensor,
                              physics_b: torch.Tensor) -> torch.Tensor:
    """Same physical system under different trajectories -> same encoding."""
    return (physics_a - physics_b).square().mean()
