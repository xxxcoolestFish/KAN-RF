"""Full-dimensional, non-compressive cognitive parameter transport."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def cognitive_parameter_vector(cognitive: nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in cognitive.parameters()])


class FullParameterTransport(nn.Module):
    """An injective diagonal transport with the same dimension as cognition.

    The initial map is a scaled identity.  No parameter coordinates are
    discarded; ``theta_dim`` is identical to the complete cognitive parameter
    count.  The affine scale and shift are available for later calibration but
    are frozen in the first validation so that policy training cannot hide a
    lossy transport.
    """

    def __init__(self, cognitive: nn.Module):
        super().__init__()
        reference = cognitive_parameter_vector(cognitive)
        self.theta_dim = int(reference.numel())
        global_scale = reference.square().mean().sqrt().clamp_min(0.1)
        self.register_buffer("global_scale", global_scale)
        self.log_scale = nn.Parameter(torch.zeros(self.theta_dim))
        self.shift = nn.Parameter(torch.zeros(self.theta_dim))

    def forward(self, cognitive: nn.Module) -> torch.Tensor:
        theta = cognitive_parameter_vector(cognitive).to(self.global_scale)
        normalized = theta / self.global_scale
        return torch.exp(self.log_scale) * normalized + self.shift

    def freeze(self):
        for parameter in self.parameters():
            parameter.requires_grad = False


class FullParameterReceiverPolicy(nn.Module):
    """Direct policy with a dense query over the complete physical vector."""

    def __init__(self, cognitive: nn.Module, transport: FullParameterTransport,
                 state_dim: int = 6, goal_dim: int = 6, hidden_dim: int = 32,
                 physical_features: int = 8, action_dim: int = 1):
        super().__init__()
        self.cognitive, self.transport = cognitive, transport
        self.input_dim = state_dim + goal_dim
        self.physical_features = physical_features
        self.task_trunk = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim), nn.Tanh(),
        )
        self.query = nn.Linear(
            self.input_dim, physical_features * transport.theta_dim,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + physical_features, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state, goal, use_physics: bool = True):
        if goal.shape[0] == 1 and state.shape[0] != 1:
            goal = goal.expand(state.shape[0], -1)
        x = torch.cat([state, goal], dim=-1)
        task = self.task_trunk(x)
        if use_physics:
            omega = self.transport(self.cognitive)
            query = torch.tanh(self.query(x)).view(
                x.shape[0], self.physical_features, self.transport.theta_dim,
            )
            physical = torch.einsum("bkn,n->bk", query, omega)
            physical = physical / self.transport.theta_dim ** 0.5
        else:
            physical = torch.zeros(
                x.shape[0], self.physical_features,
                dtype=x.dtype, device=x.device,
            )
        return torch.tanh(self.head(torch.cat([task, physical], dim=-1)))

    def transported_parameters(self):
        return self.transport(self.cognitive)
