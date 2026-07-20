"""Structural checks for the cognitive pullback policy."""

import torch

from cpbn import OracleAcrobotDynamics
from cpbn.cognitive_pullback import (
    CognitivePullbackActor,
    local_jacobians_batch,
)
from cpbn.time_varying_tube import apply_tangent_error


def sample_inputs(batch=5, horizon=6):
    center = torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    state = center.view(1, 6).expand(batch, -1).clone()
    error = 0.02 * torch.randn(batch, horizon, 4)
    corridor = apply_tangent_error(
        center.view(1, 1, 6).expand(batch, horizon, -1), error,
    )
    state_jacobian = torch.eye(4).view(1, 1, 4, 4).expand(
        batch, horizon, -1, -1,
    ).clone()
    action_jacobian = 0.01 * torch.randn(batch, horizon, 4, 1)
    return state, corridor, state_jacobian, action_jacobian


def test_oracle_batch_jacobians_have_expected_shapes():
    state, _, _, _ = sample_inputs(batch=7)
    state_jacobian, action_jacobian = local_jacobians_batch(
        OracleAcrobotDynamics(), state,
    )
    assert state_jacobian.shape == (7, 4, 4)
    assert action_jacobian.shape == (7, 4, 1)
    assert float(action_jacobian.norm(dim=(1, 2)).min()) > 0.0


def test_pullback_actor_has_no_action_when_cognitive_action_map_is_zero():
    torch.manual_seed(7)
    actor = CognitivePullbackActor(hidden_dim=16)
    state, corridor, state_jacobian, action_jacobian = sample_inputs()
    mean = actor.mean(
        state, corridor, state_jacobian, torch.zeros_like(action_jacobian),
    )
    assert torch.equal(mean, torch.zeros_like(mean))


def test_pullback_action_changes_when_cognitive_action_map_changes():
    torch.manual_seed(13)
    actor = CognitivePullbackActor(hidden_dim=16)
    state, corridor, state_jacobian, action_jacobian = sample_inputs()
    first = actor.mean(state, corridor, state_jacobian, action_jacobian)
    second = actor.mean(state, corridor, state_jacobian, -action_jacobian)
    assert float((first - second).abs().max()) > 1e-5
