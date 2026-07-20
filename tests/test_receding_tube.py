"""Checks for short-horizon receding tube utilities."""

import torch

from cpbn.receding_tube import local_state_distance, nearest_reference_progress
from cpbn.time_varying_tube import apply_tangent_error


def test_local_state_distance_zero_for_same_state():
    state = torch.tensor([[1.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
    assert torch.allclose(local_state_distance(state, state), torch.zeros(1))


def test_nearest_progress_is_monotone():
    start = torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    errors = torch.zeros(6, 4)
    errors[:, 0] = torch.linspace(0.0, 0.25, 6)
    reference = apply_tangent_error(start.expand(6, -1), errors)
    progress = nearest_reference_progress(reference[4], reference, 2, 3)
    assert progress == 4
