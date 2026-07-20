"""Mandatory forward coupling to the full cognitive parameter vector."""

from __future__ import annotations

import torch
from torch import nn

from .full_parameter_transport import FullParameterTransport


class MandatoryFullParameterPolicy(nn.Module):
    """Every action path contains a dense contraction with all omega entries."""

    def __init__(self, cognitive: nn.Module, transport: FullParameterTransport,
                 state_dim: int = 6, goal_dim: int = 6,
                 physical_features: int = 16, action_dim: int = 1,
                 action_limit: float = 0.9):
        super().__init__()
        self.cognitive, self.transport = cognitive, transport
        self.input_dim = state_dim + goal_dim
        self.physical_features = physical_features
        self.action_limit = action_limit
        self.query = nn.Linear(
            self.input_dim, physical_features * transport.theta_dim,
        )
        self.head = nn.Sequential(
            nn.Linear(physical_features, physical_features), nn.Tanh(),
            nn.Linear(physical_features, action_dim),
        )

    def forward(self, state, goal):
        if goal.shape[0] == 1 and state.shape[0] != 1:
            goal = goal.expand(state.shape[0], -1)
        x = torch.cat([state, goal], dim=-1)
        omega = self.transport(self.cognitive)
        query = torch.tanh(self.query(x)).view(
            x.shape[0], self.physical_features, self.transport.theta_dim,
        )
        physical = torch.einsum("bkn,n->bk", query, omega)
        physical = physical / self.transport.theta_dim ** 0.5
        return self.action_limit * torch.tanh(self.head(physical))

    def transported_parameters(self):
        return self.transport(self.cognitive)
