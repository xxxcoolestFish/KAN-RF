"""Small online critic for transition-cost-driven decision adaptation."""

from __future__ import annotations

import torch
from torch import nn

from kanrf._protokan import ProtoKAN


class TransitionCostCritic(nn.Module):
    def __init__(self, state_dim: int, physics_dim: int, action_dim: int = 1,
                 hidden_dim: int = 24, n_prototypes: int = 8):
        super().__init__()
        self.network = ProtoKAN(
            [state_dim + physics_dim + action_dim, hidden_dim, 1],
            n_prototypes=n_prototypes,
        )

    def forward(self, state, physics, action):
        return self.network(torch.cat([state, physics, action], dim=-1))
