"""Cognitive predictor with separate physics and trajectory representations."""

from __future__ import annotations

import torch
from torch import nn

from kanrf._protokan import ProtoKAN

from .adaptive import AdaptivePhysicsTokens


class SplitCognitivePredictor(nn.Module):
    """Separate slow physics tokens from fast trajectory memory.

    The physics branch receives transition differences and actions rather than
    absolute state history.  The state-memory branch retains full history for
    short-term prediction but is not intended for the decision interface.
    """

    def __init__(self, state_dim: int, action_dim: int, history_steps: int,
                 token_count: int = 8, token_dim: int = 8,
                 state_memory_dim: int = 16, hidden_dim: int = 24,
                 n_prototypes: int = 8):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.history_steps = history_steps
        physics_history_dim = (history_steps - 1) * state_dim + history_steps * action_dim
        self.physics_encoder = AdaptivePhysicsTokens(
            physics_history_dim, token_count, token_dim, hidden_dim, n_prototypes
        )
        self.state_memory = ProtoKAN(
            [history_steps * (state_dim + action_dim), hidden_dim, state_memory_dim],
            n_prototypes=n_prototypes,
        )
        self.dynamics = ProtoKAN(
            [state_dim + action_dim + token_count * token_dim + state_memory_dim,
             hidden_dim, state_dim],
            n_prototypes=n_prototypes,
        )

    def _physics_input(self, history: torch.Tensor) -> torch.Tensor:
        sequence = history.view(
            history.shape[0], self.history_steps, self.state_dim + self.action_dim
        )
        states = sequence[..., :self.state_dim]
        actions = sequence[..., self.state_dim:]
        differences = states[:, 1:] - states[:, :-1]
        return torch.cat([differences.flatten(1), actions.flatten(1)], dim=-1)

    def forward(self, history: torch.Tensor, state: torch.Tensor,
                action: torch.Tensor):
        physics_tokens, gates = self.physics_encoder(self._physics_input(history))
        weighted_physics = (physics_tokens * gates.unsqueeze(-1)).flatten(start_dim=1)
        state_memory = torch.tanh(self.state_memory(history))
        next_state = self.dynamics(torch.cat([
            state, action, weighted_physics, state_memory,
        ], dim=-1))
        return {
            "physics_tokens": physics_tokens,
            "physics_gates": gates,
            "physics_pooled": self.physics_encoder.pool(physics_tokens, gates),
            "state_memory": state_memory,
            "next_state": next_state,
        }
