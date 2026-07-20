"""Corrected temporal routing entry point for the Stage 62 validation."""

from __future__ import annotations

import torch

from kanrf.temporal_causal_routing import (
    _score_gradient,
    direct_reachability_gradient,
    local_influence_graph,
    repeated_transition,
    unroll_influence_graph,
)


def temporal_reachability_route(states, state_matrices, action_matrices,
                                score_fn, temperature):
    horizon = state_matrices.shape[1]
    state_scores = torch.stack(
        [score_fn(states[:, index]) for index in range(1, horizon + 1)],
        dim=1,
    )
    weights = torch.softmax(state_scores / temperature, dim=1)
    local_score_gradients = [
        _score_gradient(score_fn, states[:, index])
        for index in range(1, horizon + 1)
    ]
    costate = weights[:, -1:] * local_score_gradients[-1]
    action_routes = [None] * horizon
    for index in reversed(range(horizon)):
        action_routes[index] = torch.bmm(
            action_matrices[:, index].transpose(1, 2),
            costate.unsqueeze(-1),
        ).squeeze(-1)
        costate = torch.bmm(
            state_matrices[:, index].transpose(1, 2),
            costate.unsqueeze(-1),
        ).squeeze(-1)
        if index > 0:
            costate = costate + weights[:, index - 1:index] * (
                local_score_gradients[index - 1]
            )
    smooth_score = temperature * torch.logsumexp(
        state_scores / temperature, dim=1,
    )
    return torch.stack(action_routes, dim=1), smooth_score, weights


__all__ = [
    "direct_reachability_gradient",
    "local_influence_graph",
    "repeated_transition",
    "temporal_reachability_route",
    "unroll_influence_graph",
]
