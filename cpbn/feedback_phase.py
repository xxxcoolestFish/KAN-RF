"""Feedback alignment between real states and a nominal state corridor."""

from __future__ import annotations

import torch

from cpbn.receding_tube import LOCAL_METRIC
from cpbn.time_varying_tube import tangent_error


DEFAULT_TRANSITIONS = ((0, 0.20), (1, 0.65), (2, 0.13), (3, 0.02))


def initialize_phase_belief(
    phase: torch.Tensor,
    route_length: int,
) -> torch.Tensor:
    """Create a batch of exact initial beliefs at known reset phases."""
    belief = torch.zeros(
        phase.shape[0], route_length, dtype=torch.float32, device=phase.device,
    )
    return belief.scatter_(1, phase.long().unsqueeze(-1), 1.0)


def predict_phase_belief(
    belief: torch.Tensor,
    transitions=DEFAULT_TRANSITIONS,
) -> torch.Tensor:
    """Apply a bounded mostly-forward Markov transition with clamped endpoints."""
    predicted = torch.zeros_like(belief)
    length = belief.shape[1]
    for offset, probability in transitions:
        if offset == 0:
            predicted += probability * belief
        elif offset > 0:
            predicted[:, offset:] += probability * belief[:, :length - offset]
            predicted[:, -1] += probability * belief[:, length - offset:].sum(dim=1)
        else:
            width = -offset
            predicted[:, :length - width] += probability * belief[:, width:]
            predicted[:, 0] += probability * belief[:, :width].sum(dim=1)
    return predicted


def route_log_likelihood(
    state: torch.Tensor,
    reference: torch.Tensor,
    observation_scale: float = 0.10,
) -> torch.Tensor:
    """Evaluate phase observations in periodic tangent coordinates."""
    error = tangent_error(state.unsqueeze(1), reference.unsqueeze(0))
    metric = LOCAL_METRIC.to(error).view(1, 1, -1)
    squared = (error * metric).square().sum(dim=-1)
    return -0.5 * squared / observation_scale**2


def update_phase_belief(
    belief: torch.Tensor,
    state: torch.Tensor,
    reference: torch.Tensor,
    observation_scale: float = 0.10,
    transitions=DEFAULT_TRANSITIONS,
    minimum_phase: torch.Tensor | None = None,
) -> torch.Tensor:
    """One differentiable predict-update step for corridor phase belief."""
    predicted = predict_phase_belief(belief, transitions)
    log_likelihood = route_log_likelihood(state, reference, observation_scale)
    log_prior = torch.where(
        predicted > 0, predicted.log(), torch.full_like(predicted, -torch.inf),
    )
    log_posterior = log_prior + log_likelihood
    if minimum_phase is not None:
        index = torch.arange(belief.shape[1], device=belief.device).unsqueeze(0)
        valid = index >= minimum_phase.long().unsqueeze(1)
        log_posterior = log_posterior.masked_fill(~valid, -torch.inf)
    return torch.softmax(log_posterior, dim=1)


def belief_phase(belief: torch.Tensor) -> torch.Tensor:
    """Return the maximum-a-posteriori route phase."""
    return belief.argmax(dim=1)


def bounded_nearest_phase(
    state: torch.Tensor,
    reference: torch.Tensor,
    phase: torch.Tensor,
    backtrack: int = 4,
    advance: int = 12,
) -> torch.Tensor:
    """Nearest-point baseline constrained by the previous route phase."""
    offsets = torch.arange(
        -backtrack, advance + 1, dtype=torch.long, device=phase.device,
    )
    candidates = (phase.unsqueeze(1) + offsets).clamp(0, reference.shape[0] - 1)
    centers = reference.to(phase.device)[candidates]
    error = tangent_error(state.unsqueeze(1), centers)
    metric = LOCAL_METRIC.to(error).view(1, 1, -1)
    distance = (error * metric).square().sum(dim=-1)
    selected = distance.argmin(dim=1)
    return candidates.gather(1, selected.unsqueeze(1)).squeeze(1)
