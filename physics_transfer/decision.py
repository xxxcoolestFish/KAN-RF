"""Decision network with an explicit physical receiver subspace."""

from __future__ import annotations

import torch
from torch import nn

from kanrf._protokan import ProtoKAN


class PhysicsAwareDecision(nn.Module):
    """KAN task trunk plus a low-rank, parameter-level physics receiver."""

    def __init__(self, state_dim: int, action_dim: int, physics_slots: int = 1,
                 hidden_dim: int = 32, n_prototypes: int = 16):
        super().__init__()
        self.task_trunk = ProtoKAN([state_dim, hidden_dim],
                                   n_prototypes=n_prototypes)
        self.task_head = nn.Linear(hidden_dim, action_dim)
        self.physics_basis = nn.Parameter(
            torch.randn(physics_slots, action_dim, hidden_dim) * 0.01)

    def forward(self, state: torch.Tensor,
                receiver_parameters: torch.Tensor) -> torch.Tensor:
        hidden = self.task_trunk(state)
        task_logits = self.task_head(hidden)
        physics_logits = torch.einsum(
            "bp,pah,bh->ba", receiver_parameters, self.physics_basis, hidden)
        return torch.tanh(task_logits + physics_logits)
