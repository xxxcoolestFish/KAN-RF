import torch

from cpbn.hopper_source_twin import (
    HopperSourceAffineTwin,
    SparseComposableKANTwin,
)


def test_affine_twin_shapes_and_action_affinity():
    torch.manual_seed(3)
    model = HopperSourceAffineTwin(
        torch.ones(11),
        torch.ones(11),
        hidden_dim=16,
        depth=2,
    )
    state = torch.randn(5, 11)
    first = torch.randn(5, 3)
    second = torch.randn(5, 3)
    midpoint = model(state, 0.5 * (first + second))
    expected = 0.5 * (
        model(state, first) + model(state, second)
    )
    assert midpoint.shape == (5, 11)
    assert torch.allclose(midpoint, expected, atol=1e-6)


def test_sparse_composable_twin_shapes_and_penalty():
    model = SparseComposableKANTwin(
        torch.ones(4),
        torch.ones(4),
        action_dim=2,
        grid_size=4,
        pair_modes=2,
    )
    state = torch.randn(3, 4)
    action = torch.randn(3, 2)
    prediction = model(state, action)
    baseline, gain = model.drift_and_gain(state)
    assert prediction.shape == (3, 4)
    assert baseline.shape == (3, 4)
    assert gain.shape == (3, 4, 2)
    assert model.group_sparsity().ndim == 0


def test_affine_twin_respects_walker2d_action_dim():
    model = HopperSourceAffineTwin(
        torch.ones(17),
        torch.ones(17),
        action_dim=6,
        hidden_dim=16,
        depth=2,
    )
    state = torch.randn(5, 17)
    action = torch.randn(5, 6)
    baseline, gain = model.drift_and_gain(state)
    assert model(state, action).shape == (5, 17)
    assert baseline.shape == (5, 17)
    assert gain.shape == (5, 17, 6)
