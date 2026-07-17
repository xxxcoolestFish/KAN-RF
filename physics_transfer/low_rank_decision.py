"""Parameter-efficient physics-only adaptation for separated decisions."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class LowRankPhysicsAdapterDecision(nn.Module):
    """Freeze the task branch and adapt only a low-rank physics basis update.

    The effective physical basis is

    ``B_t = B_0 + U V^T``.

    ``B_0`` and the task branch are frozen after single-environment
    pretraining.  The rank is an adaptation capacity, not a physical parameter
    count.
    """

    def __init__(self, pretrained, rank: int = 4):
        super().__init__()
        self.task_trunk = pretrained.task_trunk
        self.task_head = pretrained.task_head
        self.physics_dim = pretrained.physics_dim
        self.action_dim = pretrained.action_dim
        self.hidden_dim = pretrained.hidden_dim
        flat_dim = self.action_dim * self.hidden_dim
        self.register_buffer("base_basis", pretrained.physics_basis.detach().clone())
        self.adapter_left = nn.Parameter(torch.randn(self.physics_dim, rank) * 0.01)
        self.adapter_right = nn.Parameter(torch.randn(rank, flat_dim) * 0.01)
        for parameter in self.task_trunk.parameters():
            parameter.requires_grad = False
        for parameter in self.task_head.parameters():
            parameter.requires_grad = False

    def effective_basis(self):
        update = self.adapter_left @ self.adapter_right
        return self.base_basis + update.view(self.physics_dim, self.action_dim, self.hidden_dim)

    def forward(self, state: torch.Tensor, physics: torch.Tensor):
        task_hidden = torch.tanh(self.task_trunk(state))
        task_logits = self.task_head(task_hidden)
        basis = self.effective_basis()
        physics_logits = torch.einsum("bp,pah,bh->ba", physics, basis, task_hidden)
        logits = task_logits + physics_logits
        return {
            "task_hidden": task_hidden,
            "task_logits": task_logits,
            "physics_logits": physics_logits,
            "logits": logits,
            "action": torch.tanh(logits),
        }

    def adaptation_norm(self):
        return torch.linalg.vector_norm(self.adapter_left @ self.adapter_right)

    def stabilization_loss(self):
        return self.adaptation_norm().square()
