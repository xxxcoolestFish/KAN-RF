"""Split cognitive predictor with a non-bypassable physics residual branch."""

from __future__ import annotations

import torch
from torch import nn

from kanrf._protokan import ProtoKAN

from .adaptive import AdaptivePhysicsTokens


class SplitCognitivePredictorV2(nn.Module):
    """Current state branch plus a history-derived physics residual branch.

    The fast branch receives only the current state/action.  Consequently, for
    counterfactual samples sharing the same state/action, only the physics
    branch can explain different transitions.
    """

    def __init__(self, state_dim: int, action_dim: int, history_steps: int,
                 token_count: int = 8, token_dim: int = 8,
                 state_context_dim: int = 16, hidden_dim: int = 24,
                 n_prototypes: int = 8):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.history_steps = history_steps
        physics_history_dim = (history_steps - 1) * state_dim + history_steps * action_dim
        self.physics_encoder = AdaptivePhysicsTokens(
            physics_history_dim, token_count, token_dim, hidden_dim, n_prototypes
        )
        self.state_context = ProtoKAN(
            [state_dim, hidden_dim, state_context_dim], n_prototypes=n_prototypes
        )
        self.base_dynamics = ProtoKAN(
            [state_dim + action_dim + state_context_dim, hidden_dim, state_dim],
            n_prototypes=n_prototypes,
        )
        self.physics_dynamics = ProtoKAN(
            [state_dim + action_dim + token_count * token_dim,
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
        tokens, gates = self.physics_encoder(self._physics_input(history))
        weighted = (tokens * gates.unsqueeze(-1)).flatten(start_dim=1)
        context = torch.tanh(self.state_context(state))
        base = self.base_dynamics(torch.cat([state, action, context], dim=-1))
        residual = self.physics_dynamics(torch.cat([state, action, weighted], dim=-1))
        return {
            "physics_tokens": tokens,
            "physics_gates": gates,
            "physics_pooled": self.physics_encoder.pool(tokens, gates),
            "state_context": context,
            "base_next_state": base,
            "physics_residual": residual,
            "next_state": base + residual,
        }
