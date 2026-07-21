"""Structural checks for counterfactual Bellman/Sobolev supervision."""

import torch
from torch import nn

from cpbn.counterfactual_adjoint import CounterfactualAdjointActor
from scripts.validate_oracle_counterfactual_adjoint_actor import (
    counterfactual_losses,
)


class LinearActionDynamics(nn.Module):
    def __init__(self, gain=1.0):
        super().__init__()
        self.gain = gain

    def forward(self, state, action):
        delta = torch.zeros_like(state)
        delta[:, 5] = self.gain * action[:, 0]
        return state + delta


class ZeroValue(nn.Module):
    def forward(self, state, corridor):
        del state, corridor
        return torch.zeros(1).expand(4)


def _inputs():
    state = torch.tensor([
        [1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    ]).expand(4, -1).clone()
    target = state.clone()
    target[:, 5] = 0.3
    corridor = target.unsqueeze(1).expand(-1, 3, -1).clone()
    return state, corridor


def test_immediate_bellman_term_points_toward_next_corridor_state():
    torch.manual_seed(3)
    actor = CounterfactualAdjointActor(
        hidden_dim=8, corridor_horizon=3,
        ridge=1e-6, log_gain=-2.0,
    )
    for parameter in actor.potential.parameters():
        parameter.data.zero_()
    state, corridor = _inputs()
    mean = actor.mean(state, corridor, LinearActionDynamics())
    assert torch.all(mean > 0.0)


def test_counterfactual_value_and_slope_losses_are_finite_and_trainable():
    torch.manual_seed(5)
    actor = CounterfactualAdjointActor(
        hidden_dim=8, corridor_horizon=3,
    )
    target = ZeroValue()
    state, corridor = _inputs()
    level, slope = counterfactual_losses(
        actor, target, LinearActionDynamics(),
        state, corridor, delta=0.5,
    )
    loss = level + slope
    loss.backward()
    assert torch.isfinite(level)
    assert torch.isfinite(slope)
    assert any(
        parameter.grad is not None
        and bool(torch.isfinite(parameter.grad).all())
        for parameter in actor.potential.parameters()
    )
