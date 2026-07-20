"""Behavioral checks for the clean CPBN operator chain."""

import torch

from cpbn import (
    GOAL,
    ImplicitBellmanAction,
    OracleAcrobotDynamics,
    ValueNetwork,
    bellman_return,
    random_states,
)


def test_oracle_dynamics_preserves_state_encoding():
    dynamics = OracleAcrobotDynamics()
    next_state = dynamics(random_states(12), torch.randn(12, 1).clamp(-1, 1))
    assert next_state.shape == (12, 6)
    assert next_state.isfinite().all()
    first_norm = next_state[:, :2].square().sum(dim=-1)
    second_norm = next_state[:, 2:4].square().sum(dim=-1)
    assert torch.allclose(first_norm, torch.ones_like(first_norm), atol=1e-5)
    assert torch.allclose(second_norm, torch.ones_like(second_norm), atol=1e-5)


def test_bellman_pullback_depends_on_action():
    torch.manual_seed(1)
    dynamics = OracleAcrobotDynamics()
    value = ValueNetwork(16)
    state = random_states(8)
    goal = GOAL.view(1, -1).expand(8, -1)
    action = torch.zeros(8, 1, requires_grad=True)
    objective, next_state, _, _ = bellman_return(
        value, dynamics, state, goal, action,
    )
    gradient = torch.autograd.grad(objective.sum(), action)[0]
    assert next_state.shape == state.shape
    assert gradient.isfinite().all()
    assert float(gradient.abs().max()) > 0.0


def test_implicit_action_has_no_actor_parameters():
    torch.manual_seed(2)
    layer = ImplicitBellmanAction(iterations=2)
    dynamics = OracleAcrobotDynamics()
    value = ValueNetwork(16)
    state = random_states(6)
    goal = GOAL.view(1, -1).expand(6, -1)
    action = layer(value, dynamics, state, goal)
    assert sum(parameter.numel() for parameter in layer.parameters()) == 0
    assert action.shape == (6, 1)
    assert bool((action.abs() <= 1.0).all())
