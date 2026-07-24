import torch

from cpbn.generic_affine_kan import (
    AffineKANContext,
    CompactInteractionKANDictionary,
)


def test_rectangular_two_response_one_action_decode():
    basis = CompactInteractionKANDictionary(
        [2.4, 0.5, 3.0, 3.0], [12.0], pair_modes=2,
    )
    width = basis.feature_dim
    coefficients = torch.zeros(2 * width, 2)
    coefficients[width, 0] = 12.0
    coefficients[width, 1] = -24.0
    context = AffineKANContext(coefficients)
    state = torch.zeros(8, 4)
    desired_action = torch.linspace(-2.0, 2.0, 8)[:, None]
    virtual = context.acceleration(basis, state, desired_action)
    decoded = context.decode_action(basis, state, virtual)
    assert decoded.shape == (8, 1)
    assert torch.allclose(decoded, desired_action, atol=1e-5)
