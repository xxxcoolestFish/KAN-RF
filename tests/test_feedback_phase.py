"""Tests for state-feedback corridor phase alignment."""

import torch

from cpbn.feedback_phase import (
    belief_phase,
    bounded_nearest_phase,
    initialize_phase_belief,
    predict_phase_belief,
    update_phase_belief,
)
from cpbn.time_varying_tube import apply_tangent_error


def make_reference(length=20):
    angle = torch.linspace(0.0, 1.0, length)
    return torch.stack([
        torch.cos(angle), torch.sin(angle),
        torch.cos(0.5 * angle), torch.sin(0.5 * angle),
        angle / 6.0, angle / 8.0,
    ], dim=-1)


def test_phase_prediction_preserves_probability():
    phase = torch.tensor([0, 8, 19])
    belief = initialize_phase_belief(phase, 20)
    predicted = predict_phase_belief(belief)
    assert torch.allclose(predicted.sum(dim=1), torch.ones(3))
    assert int(predicted[-1].argmax()) == 19


def test_feedback_belief_follows_matching_reference_state():
    reference = make_reference()
    belief = initialize_phase_belief(torch.tensor([5]), reference.shape[0])
    state = apply_tangent_error(reference[7:8], torch.zeros(1, 4))
    posterior = update_phase_belief(
        belief, state, reference, observation_scale=0.03,
    )
    assert int(belief_phase(posterior)) == 7


def test_bounded_nearest_phase_can_stay_or_advance():
    reference = make_reference()
    phase = torch.tensor([6, 6])
    state = torch.stack([reference[6], reference[10]])
    selected = bounded_nearest_phase(
        state, reference, phase, backtrack=2, advance=6,
    )
    assert selected.tolist() == [6, 10]
