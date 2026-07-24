from types import SimpleNamespace

import numpy as np
import torch

from cpbn.generic_affine_kan import (
    AffineKANContext,
    CompactInteractionKANDictionary,
)
from scripts.validate_hopper_joint_online_adaptation import (
    cognitive_effect_residual_action,
)


def constant_context(basis, drift, gain):
    coefficients = torch.zeros(
        (1 + basis.action_dim) * basis.feature_dim,
        drift.shape[-1],
    )
    coefficients[0] = drift
    for action_index in range(basis.action_dim):
        offset = (action_index + 1) * basis.feature_dim
        coefficients[offset] = gain[:, action_index]
    return AffineKANContext(coefficients)


def test_transport_is_anchored_at_nominal_action():
    basis = CompactInteractionKANDictionary(
        torch.ones(2), torch.ones(1), pair_modes=1,
    )
    drift = torch.tensor([0.2, -0.1])
    gain = torch.tensor([[2.0], [1.0]])
    context = constant_context(basis, drift, gain)
    state = torch.zeros(1, 2)
    nominal = torch.tensor([[0.7]])
    desired = context.acceleration(basis, state, nominal)

    transported = context.transport_action(
        basis,
        state,
        desired,
        nominal,
        regularization=0.1,
    )

    assert torch.allclose(transported, nominal, atol=1e-6)


def test_transport_reduces_effect_error_without_minimum_norm_jump():
    basis = CompactInteractionKANDictionary(
        torch.ones(2), torch.ones(1), pair_modes=1,
    )
    context = constant_context(
        basis,
        torch.tensor([0.0, 0.0]),
        torch.tensor([[0.5], [0.25]]),
    )
    state = torch.zeros(1, 2)
    nominal = torch.tensor([[0.8]])
    desired = torch.tensor([[0.8, 0.4]])
    before = context.acceleration(basis, state, nominal)

    transported = context.transport_action(
        basis,
        state,
        desired,
        nominal,
        regularization=0.01,
    )
    after = context.acceleration(basis, state, transported)

    assert (after - desired).norm() < (before - desired).norm()
    assert transported.item() > nominal.item()


def test_zero_control_coordinate_residual_preserves_cognitive_base_action():
    basis = CompactInteractionKANDictionary(
        torch.ones(2), torch.ones(1), pair_modes=1,
    )
    context = constant_context(
        basis,
        torch.tensor([0.2, -0.1]),
        torch.tensor([[2.0], [1.0]]),
    )

    class SourcePolicy:
        @staticmethod
        def action(observation):
            del observation
            return torch.tensor([0.35])

    args = SimpleNamespace(
        residual_scale=0.25,
        effect_metric="identity",
        metric_isotropic_floor=0.05,
        pullback_damping=0.1,
    )
    action = cognitive_effect_residual_action(
        np.zeros(2, dtype=np.float32),
        np.zeros(1, dtype=np.float32),
        SourcePolicy(),
        basis,
        context,
        context,
        args,
    )

    assert np.allclose(action, np.asarray([0.35]), atol=1e-6)
