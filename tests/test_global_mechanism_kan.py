import torch

from cpbn.generic_affine_kan import AffineKANContext
from cpbn.global_mechanism_kan import (
    GlobalMechanismKANDynamics,
    RecursiveGlobalMechanismEstimator,
)


class LinearBasis:
    def context_features(self, state, action):
        return torch.cat((state, action), dim=-1)


def test_global_latent_is_recovered_from_transition_effects():
    basis = LinearBasis()
    source = AffineKANContext(torch.zeros(3, 2))
    mechanisms = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 0.5], [0.2, 0.0]],
            [[0.0, 0.5], [0.5, 0.0], [0.0, 1.0]],
        ]
    )
    model = GlobalMechanismKANDynamics(source, mechanisms)
    state = torch.tensor(
        [[1.0, 0.5], [-0.5, 1.0], [0.2, -0.7], [1.2, -0.3]]
    )
    action = torch.tensor([[0.3], [-0.2], [0.8], [-0.5]])
    expected = torch.tensor([0.7, -1.2])
    delta = model.context(expected).acceleration(basis, state, action)

    inferred = model.infer_latent(
        basis,
        state,
        action,
        delta,
        torch.ones(2),
        ridge=1e-7,
    )

    torch.testing.assert_close(inferred, expected, atol=1e-5, rtol=1e-5)


def test_recursive_estimator_recovers_the_same_global_latent():
    basis = LinearBasis()
    source = AffineKANContext(torch.zeros(3, 2))
    mechanisms = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 0.5], [0.2, 0.0]],
            [[0.0, 0.5], [0.5, 0.0], [0.0, 1.0]],
        ]
    )
    model = GlobalMechanismKANDynamics(source, mechanisms)
    estimator = RecursiveGlobalMechanismEstimator(
        model, basis, torch.ones(2), ridge=1e-7,
    )
    state = torch.tensor(
        [[1.0, 0.5], [-0.5, 1.0], [0.2, -0.7], [1.2, -0.3]]
    )
    action = torch.tensor([[0.3], [-0.2], [0.8], [-0.5]])
    expected = torch.tensor([0.7, -1.2])
    delta = model.context(expected).acceleration(basis, state, action)

    estimator.update(state[:2], action[:2], delta[:2])
    estimator.update(state[2:], action[2:], delta[2:])

    torch.testing.assert_close(
        estimator.latent(), expected, atol=1e-5, rtol=1e-5,
    )


def test_zero_weight_evidence_does_not_move_the_latent():
    basis = LinearBasis()
    source = AffineKANContext(torch.zeros(3, 2))
    model = GlobalMechanismKANDynamics(
        source,
        torch.tensor(
            [
                [[1.0, 0.0], [0.0, 0.5], [0.2, 0.0]],
                [[0.0, 0.5], [0.5, 0.0], [0.0, 1.0]],
            ]
        ),
    )
    estimator = RecursiveGlobalMechanismEstimator(
        model, basis, torch.ones(2),
    )
    state = torch.tensor([[1.0, 0.5], [-0.5, 1.0]])
    action = torch.tensor([[0.3], [-0.2]])
    delta = torch.ones(2, 2)
    before = estimator.latent().clone()

    estimator.update(
        state, action, delta, evidence_weight=0.0,
    )

    torch.testing.assert_close(estimator.latent(), before)
