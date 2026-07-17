"""Semantic physical-parameter interface between cognition and decision."""

from __future__ import annotations

import torch
from torch import nn


class PhysicsTransport(nn.Module):
    """Map canonical cognitive parameters to decision receiver slots.

    The first implementation is deliberately linear and low-capacity.  This
    makes the interface inspectable and prevents it from becoming a second
    policy network.  A KAN coefficient transport can replace this module later.
    """

    def __init__(self, physics_dim: int, receiver_dim: int | None = None):
        super().__init__()
        receiver_dim = physics_dim if receiver_dim is None else receiver_dim
        self.physics_dim = physics_dim
        self.receiver_dim = receiver_dim
        self.map = nn.Linear(physics_dim, receiver_dim, bias=False)
        with torch.no_grad():
            self.map.weight.zero_()
            for i in range(min(physics_dim, receiver_dim)):
                self.map.weight[i, i] = 1.0

    def forward(self, physics: torch.Tensor) -> torch.Tensor:
        return self.map(physics)

    def alignment_loss(self) -> torch.Tensor:
        """Keep the learned transport close to an interpretable map."""
        gram = self.map.weight @ self.map.weight.T
        eye = torch.eye(self.receiver_dim, device=gram.device, dtype=gram.dtype)
        return (gram - eye).square().mean()
