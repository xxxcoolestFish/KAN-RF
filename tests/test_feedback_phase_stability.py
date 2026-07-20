"""Numerical and monotonicity checks for feedback phase inference."""

import torch

from cpbn.feedback_phase import (
    belief_phase,
    initialize_phase_belief,
    update_phase_belief,
)


def test_monotone_log_update_stays_finite_under_extreme_mismatch():
    angle = torch.linspace(0.0, 1.0, 20)
    reference = torch.stack([
        torch.cos(angle), torch.sin(angle),
        torch.cos(0.5 * angle), torch.sin(0.5 * angle),
        angle / 6.0, angle / 8.0,
    ], dim=-1)
    lower_bound = torch.tensor([8])
    belief = initialize_phase_belief(lower_bound, reference.shape[0])
    state = reference[0:1]
    posterior = update_phase_belief(
        belief, state, reference,
        observation_scale=1e-3,
        minimum_phase=lower_bound,
    )
    assert bool(torch.isfinite(posterior).all())
    assert torch.allclose(posterior.sum(dim=1), torch.ones(1))
    assert int(belief_phase(posterior)) >= int(lower_bound)
