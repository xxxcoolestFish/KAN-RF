"""Small decision-side networks used by CPBN."""

from __future__ import annotations

import torch
from torch import nn

from cpbn.acrobot import STATE_DIM


class ValueNetwork(nn.Module):
    """Goal-conditioned scalar value; the action layer has no actor weights."""

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, goal], dim=-1)).squeeze(-1)
