"""Checks for the regularized multi-step cognitive inverse."""

import torch

from cpbn.cognitive_inverse import (
    CognitiveInverseActor,
    action_to_future_sensitivities,
    regularized_cognitive_inverse,
)
from cpbn.time_varying_tube import apply_tangent_error


def inputs(batch=4, horizon=5):
    center = torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    state = center.view(1, 6).expand(batch, -1).clone()
    corridor = apply_tangent_error(
        center.view(1, 1, 6).expand(batch, horizon, -1),
        0.02 * torch.randn(batch, horizon, 4),
    )
    state_jacobian = torch.eye(4).view(1, 1, 4, 4).expand(
        batch, horizon, -1, -1,
    ).clone()
    action_jacobian = torch.zeros(batch, horizon, 4, 1)
    action_jacobian[..., 2, 0] = 0.01
    return state, corridor, state_jacobian, action_jacobian


def test_future_sensitivity_chains_state_jacobians():
    _, _, state_jacobian, action_jacobian = inputs(batch=1, horizon=3)
    state_jacobian[:, 1:, 2, 2] = 2.0
    sensitivity = action_to_future_sensitivities(
        state_jacobian, action_jacobian,
    )
    assert torch.allclose(
        sensitivity[0, :, 2, 0], torch.tensor([0.01, 0.02, 0.04]),
    )


def test_zero_cognitive_control_map_forces_zero_action():
    torch.manual_seed(5)
    actor = CognitiveInverseActor(hidden_dim=16)
    state, corridor, state_jacobian, action_jacobian = inputs()
    mean = actor.mean(
        state, corridor, state_jacobian, torch.zeros_like(action_jacobian),
    )
    assert torch.equal(mean, torch.zeros_like(mean))


def test_inverse_increases_compensation_for_weaker_effect_map():
    _, _, state_jacobian, action_jacobian = inputs(batch=1)
    desired = torch.zeros(1, 5, 4)
    desired[..., 2] = 0.01
    weight = torch.ones_like(desired)
    ridge = torch.tensor([[1e-8]])
    strong = regularized_cognitive_inverse(
        desired, weight, state_jacobian, action_jacobian, ridge,
    )
    weak = regularized_cognitive_inverse(
        desired, weight, state_jacobian, 0.6 * action_jacobian, ridge,
    )
    assert float(weak.abs()) > float(strong.abs())
