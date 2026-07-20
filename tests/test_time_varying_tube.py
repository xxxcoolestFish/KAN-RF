"""Geometry checks for the time-varying cognitive tube interface."""

import torch

from cpbn.time_varying_tube import (
    apply_tangent_error,
    tangent_coordinates,
    tangent_error,
)


def test_tangent_round_trip_for_local_errors():
    center = torch.tensor(
        [
            [1.0, 0.0, 0.0, 1.0, 0.10, -0.20],
            [-1.0, 0.0, 0.0, -1.0, -0.30, 0.25],
        ]
    )
    local_error = torch.tensor(
        [
            [0.08, -0.04, 0.03, -0.02],
            [-0.06, 0.05, 0.02, 0.04],
        ]
    )
    perturbed = apply_tangent_error(center, local_error)
    recovered = tangent_error(perturbed, center)
    assert torch.allclose(recovered, local_error, atol=1e-6)


def test_tangent_error_wraps_angular_branch_cut():
    left = torch.tensor([[torch.cos(torch.tensor(torch.pi - 0.02)),
                          torch.sin(torch.tensor(torch.pi - 0.02)),
                          1.0, 0.0, 0.0, 0.0]])
    right = torch.tensor([[torch.cos(torch.tensor(-torch.pi + 0.02)),
                           torch.sin(torch.tensor(-torch.pi + 0.02)),
                           1.0, 0.0, 0.0, 0.0]])
    error = tangent_error(left, right)
    assert torch.isclose(error[0, 0], torch.tensor(-0.04), atol=1e-5)
    assert torch.allclose(error[0, 1:], torch.zeros(3), atol=1e-6)


def test_tangent_coordinates_preserve_batch_shape():
    state = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.25, -0.50]).repeat(7, 1)
    coordinates = tangent_coordinates(state)
    assert coordinates.shape == (7, 4)
    assert torch.isfinite(coordinates).all()
