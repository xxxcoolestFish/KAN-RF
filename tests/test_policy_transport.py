"""Checks for implicit cognition-to-policy parameter transport."""

import torch

from cpbn import OracleAcrobotDynamics
from cpbn.corridor_policy import DirectCorridorActor, future_corridor
from cpbn.policy_transport import (
    apply_parameter_delta,
    implicit_transport_delta,
)
from cpbn.time_varying_tube import apply_tangent_error


def transport_inputs(batch=6):
    center = torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    reference = center.view(1, 6).expand(20, -1).clone()
    phase = torch.arange(batch) % 8
    error = torch.zeros(batch, 4)
    error[:, 0] = torch.linspace(-0.03, 0.03, batch)
    state = apply_tangent_error(reference[phase], error)
    return reference, state, phase


def test_identical_cognition_produces_zero_parameter_transport():
    torch.manual_seed(3)
    actor = DirectCorridorActor(hidden_dim=8)
    reference, state, phase = transport_inputs()
    dynamics = OracleAcrobotDynamics()
    delta, diagnostics = implicit_transport_delta(
        actor, dynamics, dynamics, reference, state, phase,
        rollout_steps=2, corridor_horizon=4, fisher_draws=1,
    )
    assert diagnostics.gradient_difference_norm == 0.0
    assert all(torch.equal(value, torch.zeros_like(value)) for value in delta.values())


def test_parameter_transport_reaches_encoder_and_decision_head():
    torch.manual_seed(7)
    actor = DirectCorridorActor(hidden_dim=8)
    reference, state, phase = transport_inputs()
    delta, diagnostics = implicit_transport_delta(
        actor,
        OracleAcrobotDynamics(),
        OracleAcrobotDynamics((7.35, 0.0, 0.5, 0.8)),
        reference, state, phase,
        rollout_steps=2, corridor_horizon=4, fisher_draws=1,
    )
    assert diagnostics.gradient_difference_norm > 0.0
    assert any(value.norm() > 0 for name, value in delta.items() if "encoder" in name)
    assert any(value.norm() > 0 for name, value in delta.items() if "head" in name)


def test_zero_transport_preserves_actor_output_exactly():
    torch.manual_seed(11)
    actor = DirectCorridorActor(hidden_dim=8)
    reference, state, phase = transport_inputs()
    corridor = future_corridor(reference, phase, 4)
    before = actor.distribution(state, corridor).mean
    zero = {
        name: torch.zeros_like(parameter)
        for name, parameter in actor.named_parameters()
        if name != "log_std"
    }
    transported = apply_parameter_delta(actor, zero)
    after = transported.distribution(state, corridor).mean
    assert torch.equal(before, after)
