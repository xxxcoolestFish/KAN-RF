"""Map cognitive operator queries to decision-network physical parameters."""

from __future__ import annotations

import torch
from torch import nn

from kanrf._protokan import ProtoKAN


class OperatorToLowRankAdapter(nn.Module):
    """Generate a low-rank decision basis update from a cognitive operator."""

    def __init__(self, operator_dim: int, physics_dim: int, action_dim: int,
                 hidden_dim: int = 32, rank: int = 1, n_prototypes: int = 8):
        super().__init__()
        flat_dim = action_dim * hidden_dim
        self.physics_dim = physics_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.rank = rank
        self.generator = ProtoKAN(
            [operator_dim, 32, rank * (physics_dim + flat_dim)],
            n_prototypes=n_prototypes,
        )

    def forward(self, operator):
        raw = torch.tanh(self.generator(operator))
        left, right = raw.split(
            [self.rank * self.physics_dim,
             self.rank * self.action_dim * self.hidden_dim], dim=-1
        )
        left = left.view(-1, self.physics_dim, self.rank)
        right = right.view(-1, self.rank, self.action_dim * self.hidden_dim)
        update = torch.bmm(left, right).view(
            -1, self.physics_dim, self.action_dim, self.hidden_dim
        )
        return 0.01 * update


class OperatorMappedDecision(nn.Module):
    """Frozen task/base decision plus operator-conditioned physical parameters."""

    def __init__(self, base_decision, operator_dim: int, rank: int = 1):
        super().__init__()
        self.task_trunk = base_decision.task_trunk
        self.task_head = base_decision.task_head
        self.physics_dim = base_decision.physics_dim
        self.action_dim = base_decision.action_dim
        self.hidden_dim = base_decision.hidden_dim
        self.register_buffer("base_basis", base_decision.physics_basis.detach().clone())
        self.mapper = OperatorToLowRankAdapter(
            operator_dim, self.physics_dim, self.action_dim,
            hidden_dim=self.hidden_dim, rank=rank,
        )
        for parameter in self.task_trunk.parameters(): parameter.requires_grad = False
        for parameter in self.task_head.parameters(): parameter.requires_grad = False

    def forward(self, state, operator):
        task_hidden = torch.tanh(self.task_trunk(state))
        task_logits = self.task_head(task_hidden)
        update = self.mapper(operator)
        basis = self.base_basis.unsqueeze(0) + update
        # The cognitive operator has already generated the decision parameters;
        # do not multiply by the operator a second time.
        physics_logits = torch.einsum("bpah,bh->ba", basis, task_hidden)
        logits = task_logits + physics_logits
        return {
            "task_logits": task_logits,
            "physics_logits": physics_logits,
            "adapter": update,
            "logits": logits,
            "action": torch.tanh(logits),
        }

    def adapter_norm(self, operator):
        return torch.linalg.vector_norm(self.mapper(operator), dim=(1, 2, 3)).mean()
