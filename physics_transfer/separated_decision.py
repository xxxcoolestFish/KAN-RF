"""Decision network with functionally separated task and physics branches."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from kanrf._protokan import ProtoKAN


class SeparatedPhysicsDecision(nn.Module):
    """Task trunk plus a low-rank physical parameter-modulation branch.

    The task branch sees only the current state.  Physics codes can affect the
    action only through ``physics_basis``; this makes counterfactual action
    differences attributable to the physical receiver path.
    """

    def __init__(self, state_dim: int, action_dim: int, physics_dim: int,
                 hidden_dim: int = 32, n_prototypes: int = 16):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.physics_dim = physics_dim
        self.hidden_dim = hidden_dim
        self.task_trunk = ProtoKAN([state_dim, hidden_dim], n_prototypes=n_prototypes)
        self.task_head = nn.Linear(hidden_dim, action_dim)
        self.physics_basis = nn.Parameter(
            torch.randn(physics_dim, action_dim, hidden_dim) * 0.01
        )

    def forward(self, state: torch.Tensor,
                physics: torch.Tensor) -> torch.Tensor:
        task_hidden = torch.tanh(self.task_trunk(state))
        task_logits = self.task_head(task_hidden)
        physics_logits = torch.einsum(
            "bp,pah,bh->ba", physics, self.physics_basis, task_hidden
        )
        return {
            "task_hidden": task_hidden,
            "task_logits": task_logits,
            "physics_logits": physics_logits,
            "logits": task_logits + physics_logits,
            "action": torch.tanh(task_logits + physics_logits),
        }

    def separation_loss(self) -> torch.Tensor:
        """Discourage the physical modulation directions from duplicating task head."""
        task = F.normalize(self.task_head.weight, dim=1)
        basis = F.normalize(self.physics_basis.flatten(1), dim=1)
        task_flat = F.normalize(self.task_head.weight.flatten(), dim=0)
        overlap = (basis * task_flat.unsqueeze(0)).sum(dim=1)
        return overlap.square().mean()

    def physics_basis_orthogonality_loss(self) -> torch.Tensor:
        basis = F.normalize(self.physics_basis.flatten(1), dim=1)
        gram = basis @ basis.T
        eye = torch.eye(self.physics_dim, dtype=basis.dtype, device=basis.device)
        return (gram - eye).square().mean()


class ConcatPhysicsDecision(nn.Module):
    """Control baseline that concatenates state and physics code."""

    def __init__(self, state_dim: int, action_dim: int, physics_dim: int,
                 hidden_dim: int = 32, n_prototypes: int = 16):
        super().__init__()
        self.network = ProtoKAN(
            [state_dim + physics_dim, hidden_dim, action_dim],
            n_prototypes=n_prototypes,
        )

    def forward(self, state: torch.Tensor, physics: torch.Tensor):
        logits = self.network(torch.cat([state, physics], dim=-1))
        return {"logits": logits, "action": torch.tanh(logits)}
