"""Adaptive, formula-agnostic physics representation for stage 1."""

from __future__ import annotations

import torch
from torch import nn

from kanrf._protokan import ProtoKAN


class AdaptivePhysicsTokens(nn.Module):
    """Produce an over-complete set of gated dynamics tokens.

    ``token_count`` is a representation capacity, not the number of physical
    parameters. No token is assigned a hand-written meaning.
    """

    def __init__(self, history_dim: int, token_count: int = 8,
                 token_dim: int = 8, hidden_dim: int = 32,
                 n_prototypes: int = 8):
        super().__init__()
        self.token_count = token_count
        self.token_dim = token_dim
        self.encoder = ProtoKAN(
            [history_dim, hidden_dim, token_count * (token_dim + 1)],
            n_prototypes=n_prototypes,
        )

    def forward(self, history: torch.Tensor):
        raw = self.encoder(history).view(
            history.shape[0], self.token_count, self.token_dim + 1
        )
        tokens = torch.tanh(raw[..., :-1])
        gates = torch.sigmoid(raw[..., -1])
        return tokens, gates

    @staticmethod
    def pool(tokens: torch.Tensor, gates: torch.Tensor) -> torch.Tensor:
        weights = gates / gates.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return (tokens * weights.unsqueeze(-1)).sum(dim=1)

    @staticmethod
    def effective_rank(gates: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        return (gates > threshold).float().sum(dim=-1)


class AdaptiveCognitivePredictor(nn.Module):
    """Dynamics predictor driven by the adaptive token representation."""

    def __init__(self, state_dim: int, action_dim: int, history_dim: int,
                 token_count: int = 8, token_dim: int = 8,
                 hidden_dim: int = 32, n_prototypes: int = 8):
        super().__init__()
        self.representation = AdaptivePhysicsTokens(
            history_dim, token_count, token_dim, hidden_dim, n_prototypes
        )
        self.dynamics = ProtoKAN(
            [state_dim + action_dim + token_count * token_dim,
             hidden_dim, state_dim],
            n_prototypes=n_prototypes,
        )

    def forward(self, history: torch.Tensor, state: torch.Tensor,
                action: torch.Tensor):
        tokens, gates = self.representation(history)
        weighted_tokens = (tokens * gates.unsqueeze(-1)).flatten(start_dim=1)
        next_state = self.dynamics(torch.cat([state, action, weighted_tokens], dim=-1))
        return {
            "tokens": tokens,
            "gates": gates,
            "pooled": self.representation.pool(tokens, gates),
            "next_state": next_state,
        }
