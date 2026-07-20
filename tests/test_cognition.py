"""Checks for the manifold-preserving ProtoKAN cognition model."""

import torch

from cpbn.cognition import ProtoKANDynamics


def test_cognition_output_stays_on_state_manifold():
    torch.manual_seed(7)
    model = ProtoKANDynamics(hidden_dim=12, n_prototypes=4)
    state = torch.randn(9, 6)
    state[:, :2] = torch.nn.functional.normalize(state[:, :2], dim=-1)
    state[:, 2:4] = torch.nn.functional.normalize(state[:, 2:4], dim=-1)
    state[:, 4:] = state[:, 4:].clamp(-1.0, 1.0)
    output = model(state, torch.randn(9, 1).clamp(-1.0, 1.0))
    assert output.shape == (9, 6)
    assert torch.allclose(
        output[:, :2].square().sum(-1), torch.ones(9), atol=1e-6,
    )
    assert torch.allclose(
        output[:, 2:4].square().sum(-1), torch.ones(9), atol=1e-6,
    )
    assert bool((output[:, 4:].abs() <= 1.0).all())


def test_cognition_prediction_loss_reaches_all_parameters():
    torch.manual_seed(8)
    model = ProtoKANDynamics(hidden_dim=12, n_prototypes=4)
    state = torch.randn(6, 6)
    state[:, :2] = torch.nn.functional.normalize(state[:, :2], dim=-1)
    state[:, 2:4] = torch.nn.functional.normalize(state[:, 2:4], dim=-1)
    state[:, 4:] = state[:, 4:].clamp(-1.0, 1.0)
    action = torch.randn(6, 1).clamp(-1.0, 1.0)
    loss = model.prediction_loss(state, action, state)
    loss.backward()
    assert all(
        parameter.grad is not None and parameter.grad.isfinite().all()
        for parameter in model.parameters()
    )
