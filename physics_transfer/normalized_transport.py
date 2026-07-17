"""Normalized transport for an identifiable physical receiver dictionary."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class NormalizedPhysicsTransport(nn.Module):
    """Linear coefficient transport with normalized receiver directions."""

    def __init__(self, physics_dim: int, receiver_dim: int | None = None):
        super().__init__()
        receiver_dim = physics_dim if receiver_dim is None else receiver_dim
        self.physics_dim = physics_dim
        self.receiver_dim = receiver_dim
        self.weight = nn.Parameter(torch.empty(receiver_dim, physics_dim))
        nn.init.orthogonal_(self.weight)

    def normalized_weight(self) -> torch.Tensor:
        return F.normalize(self.weight, dim=1)

    def forward(self, physics: torch.Tensor) -> torch.Tensor:
        return F.linear(physics, self.normalized_weight())

    def alignment_loss(self) -> torch.Tensor:
        weight = self.normalized_weight()
        gram = weight @ weight.T
        eye = torch.eye(self.receiver_dim, dtype=weight.dtype, device=weight.device)
        return (gram - eye).square().mean()
