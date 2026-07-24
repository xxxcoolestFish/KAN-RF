import torch

from cpbn.generic_affine_kan import (
    AffineKANContext,
    CompactInteractionKANDictionary,
    RecursiveAffineKANEstimator,
    fit_affine_kan_context,
)


def test_dictionary_and_gain_support_one_dimensional_action():
    torch.manual_seed(839)
    basis = CompactInteractionKANDictionary(
        [2.4, 0.5, 3.0, 3.0], [12.0],
    )
    state = torch.randn(32, 4)
    action = torch.randn(32, 1)
    coefficients = torch.randn(2 * basis.feature_dim, 2)
    context = AffineKANContext(coefficients)
    drift, gain = context.drift_and_gain(basis, state)
    assert basis.context_features(state, action).shape == (
        32, 2 * basis.feature_dim,
    )
    assert drift.shape == (32, 2)
    assert gain.shape == (32, 2, 1)


def test_batch_fit_and_recursive_update_recover_affine_dynamics():
    torch.manual_seed(853)
    basis = CompactInteractionKANDictionary(
        [2.4, 0.5, 3.0, 3.0], [12.0], pair_modes=2,
    )
    state = (torch.rand(1024, 4) * 2.0 - 1.0) * torch.tensor(
        [1.0, 0.4, 1.0, 1.0],
    )
    action = 4.0 * (torch.rand(1024, 1) * 2.0 - 1.0)
    acceleration = torch.stack((
        0.2 * state[:, 0] + 0.7 * action[:, 0],
        -0.4 * state[:, 1] - 1.1 * action[:, 0],
    ), dim=-1)
    context = fit_affine_kan_context(
        basis, state, action, acceleration, ridge=1e-3,
    )
    prediction = context.acceleration(basis, state, action)
    assert float((prediction - acceleration).square().mean().sqrt()) < 2e-3

    estimator = RecursiveAffineKANEstimator(basis, context)
    estimator.update(state, action, acceleration)
    posterior = estimator.posterior()
    assert posterior.gain_uncertainty(basis, state[:16]).shape == (16, 1, 1)


def test_recursive_estimator_forgets_old_batches_without_reset():
    torch.manual_seed(857)
    basis = CompactInteractionKANDictionary(
        [2.4, 0.5, 3.0, 3.0], [12.0], pair_modes=1,
    )
    width = 2 * basis.feature_dim
    prior = AffineKANContext(torch.zeros(width, 2))
    estimator = RecursiveAffineKANEstimator(
        basis,
        prior,
        ridge=0.2,
        forgetting_factor=0.5,
    )
    state = torch.randn(4, 4)
    action = torch.randn(4, 1)
    first_design = basis.context_features(state[:2], action[:2]).double()
    second_design = basis.context_features(state[2:], action[2:]).double()
    first_target = torch.randn(2, 2).double()
    second_target = torch.randn(2, 2).double()

    estimator.update(state[:2], action[:2], first_target)
    estimator.update(state[2:], action[2:], second_target)

    expected_precision = (
        estimator.base_precision
        + 0.25 * first_design.T @ first_design
        + second_design.T @ second_design
    )
    expected_right = (
        estimator.base_right
        + 0.25 * first_design.T @ first_target
        + second_design.T @ second_target
    )
    assert torch.allclose(estimator.precision, expected_precision)
    assert torch.allclose(estimator.right, expected_right)
