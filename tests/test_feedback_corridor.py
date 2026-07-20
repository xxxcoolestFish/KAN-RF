"""Checks for time-indexed feedback corridor helpers."""

import torch

from scripts.validate_feedback_corridor_source import reference_window


def test_reference_window_pads_terminal_state():
    reference = torch.arange(30, dtype=torch.float32).reshape(5, 6)
    window = reference_window(reference, phase=3, horizon=4)
    assert window.shape == (4, 6)
    assert torch.equal(window[0], reference[4])
    assert torch.equal(window[-1], reference[4])
