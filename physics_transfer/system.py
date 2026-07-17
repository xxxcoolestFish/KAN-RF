"""End-to-end composition of the cognitive and decision networks."""

from __future__ import annotations

import torch
from torch import nn

from .cognitive import CognitivePredictor
from .decision import PhysicsAwareDecision
from .interface import PhysicsTransport


class CognitiveDecisionSystem(nn.Module):
    """One cognitive predictor, one decision network, one physics interface."""

    def __init__(self, state_dim: int, action_dim: int, history_dim: int,
                 physics_dim: int = 1, hidden_dim: int = 32,
                 n_prototypes: int = 16):
        super().__init__()
        self.cognitive = CognitivePredictor(
            state_dim, action_dim, history_dim, physics_dim, hidden_dim,
            n_prototypes)
        self.transport = PhysicsTransport(physics_dim, physics_dim)
        self.decision = PhysicsAwareDecision(
            state_dim, action_dim, physics_dim, hidden_dim, n_prototypes)

    def forward(self, history: torch.Tensor, state: torch.Tensor,
                action: torch.Tensor | None = None):
        physics, next_state = self.cognitive(history, state, action)
        receiver = self.transport(physics)
        action_out = self.decision(state, receiver)
        return {
            "physics": physics,
            "receiver": receiver,
            "next_state": next_state,
            "action": action_out,
        }

    def interface_loss(self) -> torch.Tensor:
        return self.transport.alignment_loss()
