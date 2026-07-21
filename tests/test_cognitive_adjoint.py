"""Structural checks for the one-step Bellman-adjoint Actor."""

import torch
from torch import nn

from cpbn.cognitive_adjoint import BellmanAdjointActor


class LinearActionDynamics(nn.Module):
    def __init__(self, gain):
        super().__init__()
        self.gain = gain

    def forward(self, state, action):
        delta = torch.zeros_like(state)
        delta[:, 5] = self.gain * action[:, 0]
        return state + delta


class VelocityPotential(nn.Module):
    def forward(self, state, corridor):
        del corridor
        return state[:, 5]


def _actor():
    actor = BellmanAdjointActor(
        hidden_dim=8, corridor_horizon=3,
        ridge=1e-6, log_gain=0.0,
    )
    actor.potential = VelocityPotential()
    return actor


def test_zero_cognitive_action_map_forces_zero_mean():
    actor = _actor()
    state = torch.zeros(4, 6)
    state[:, 0] = 1.0
    state[:, 2] = 1.0
    corridor = state.unsqueeze(1).expand(-1, 3, -1).clone()
    mean = actor.mean(state, corridor, LinearActionDynamics(0.0))
    assert torch.equal(mean, torch.zeros_like(mean))


def test_regularized_adjoint_compensates_for_weaker_control_map():
    actor = _actor()
    state = torch.zeros(4, 6)
    state[:, 0] = 1.0
    state[:, 2] = 1.0
    corridor = state.unsqueeze(1).expand(-1, 3, -1).clone()
    strong = actor.mean(state, corridor, LinearActionDynamics(1.0))
    weak = actor.mean(state, corridor, LinearActionDynamics(0.5))
    assert torch.all(weak.abs() > strong.abs())


def test_actor_gradient_reaches_scalar_potential_parameters():
    torch.manual_seed(9)
    actor = BellmanAdjointActor(
        hidden_dim=8, corridor_horizon=3,
        ridge=1e-4, log_gain=-2.0,
    )
    state = torch.tensor([
        [1.0, 0.0, 1.0, 0.0, 0.1, -0.2],
        [1.0, 0.0, 1.0, 0.0, -0.1, 0.2],
    ])
    corridor = state.unsqueeze(1).expand(-1, 3, -1).clone()
    action_loss = actor.mean(
        state, corridor, LinearActionDynamics(0.8),
    ).square().mean()
    loss = action_loss + actor.potential_value(state, corridor).square().mean()
    loss.backward()
    gradients = [
        parameter.grad for parameter in actor.potential.parameters()
    ]
    assert gradients
    assert all(gradient is not None for gradient in gradients)
    assert sum(float(gradient.abs().sum()) for gradient in gradients) > 0.0
