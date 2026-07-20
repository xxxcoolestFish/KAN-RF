"""Temporal routing utilities built from native ProtoKAN function edges."""

from __future__ import annotations

import torch

from kanrf.protokan_causal_router import linear_causal_route, trace_protokan


def tip_height(state):
    c1, s1, c2, s2 = state[:, 0], state[:, 1], state[:, 2], state[:, 3]
    return -c1 - (c1 * c2 - s1 * s2)


def tip_height_gradient(state):
    c1, s1, c2, s2 = state[:, 0], state[:, 1], state[:, 2], state[:, 3]
    zeros = torch.zeros_like(c1)
    return torch.stack([
        -1.0 - c2, s2, -c1, s1, zeros, zeros,
    ], dim=-1)


def temporal_causal_route(cognitive, initial_state, actions,
                          temperature=0.08, nonlinear_router=None):
    """Return action routes, predicted states, scores and route weights."""
    state = initial_state
    temporal_traces, states = [], []
    edge_delta = (
        nonlinear_router.delta if nonlinear_router is not None else 0.05
    )
    for index in range(actions.shape[1]):
        network_input = torch.cat([state, actions[:, index]], dim=-1)
        state, traces = trace_protokan(
            cognitive.network, network_input, edge_delta,
        )
        temporal_traces.append(traces)
        states.append(state)
    state_stack = torch.stack(states, dim=1)
    scores = torch.stack([tip_height(value) for value in states], dim=1)
    weights = torch.softmax(scores / temperature, dim=1)
    message = weights[:, -1:] * tip_height_gradient(states[-1])
    routes = [None] * actions.shape[1]
    for index in reversed(range(actions.shape[1])):
        if nonlinear_router is None:
            input_message, _ = linear_causal_route(
                temporal_traces[index], message,
            )
        else:
            input_message, _ = nonlinear_router(
                temporal_traces[index], message,
            )
        message = input_message[:, :6]
        routes[index] = input_message[:, 6:]
        if index > 0:
            message = message + weights[:, index - 1:index] * (
                tip_height_gradient(states[index - 1])
            )
    return torch.stack(routes, dim=1), state_stack, scores, weights
