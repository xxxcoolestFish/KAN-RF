"""Mandatory full-parameter policy with an explicit sensitivity objective."""

from __future__ import annotations

import torch
from torch import nn

from .full_parameter_transport import FullParameterTransport


class SensitivityMandatoryPolicy(nn.Module):
    """No bypass: actions are produced only from queried full parameters."""

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

    def forward_with_omega(self, state, goal, omega):
        if goal.shape[0] == 1 and state.shape[0] != 1:
            goal = goal.expand(state.shape[0], -1)
        x = torch.cat([state, goal], dim=-1)
        query = torch.tanh(self.query(x)).view(
            x.shape[0], self.physical_features, omega.numel(),
        )
        physical = torch.einsum("bkn,n->bk", query, omega)
        physical = physical / omega.numel() ** 0.5
        return self.action_limit * torch.tanh(self.head(physical))

    def forward(self, state, goal):
        return self.forward_with_omega(
            state, goal, self.transport(self.cognitive),
        )

    def transported_parameters(self):
        return self.transport(self.cognitive)


def parameter_sensitivity_loss(policy, state, goal, epsilon=0.05,
                               target_response=0.01, block_size=256):
    """Require a bounded non-zero action response to a parameter perturbation.

    The perturbation touches one randomly selected block, so repeated batches
    cover the complete cognitive parameter vector rather than only its norm.
    """
    omega = policy.transported_parameters().detach()
    count = omega.numel()
    block_count = max(1, (count + block_size - 1) // block_size)
    block = int(torch.randint(block_count, ()).item())
    start, end = block * block_size, min(count, (block + 1) * block_size)
    direction = torch.zeros_like(omega)
    direction[start:end] = torch.randn(end - start, dtype=omega.dtype)
    direction = direction / direction.norm().clamp_min(1e-8)
    delta = epsilon * direction * count ** 0.5
    action = policy.forward_with_omega(state, goal, omega)
    perturbed = policy.forward_with_omega(state, goal, omega + delta)
    response = (perturbed - action).square().mean().sqrt()
    loss = torch.relu(torch.as_tensor(target_response, dtype=response.dtype) - response).square()
    return loss, {"response": response.detach(), "block": block}
