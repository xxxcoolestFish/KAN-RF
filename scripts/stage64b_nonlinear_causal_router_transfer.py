"""Corrected entry point for nonlinear temporal edge-router training."""

import torch

from kanrf.protokan_causal_router import linear_causal_route, trace_protokan
from scripts import stage64_nonlinear_causal_router_transfer as experiment


def temporal_route(cognitive, initial_state, actions, temperature,
                   nonlinear_router=None):
    state = initial_state
    temporal_traces, states = [], []
    for index in range(actions.shape[1]):
        network_input = torch.cat([state, actions[:, index]], dim=-1)
        state, traces = trace_protokan(
            cognitive.network, network_input,
            nonlinear_router.delta if nonlinear_router is not None else 0.05,
        )
        temporal_traces.append(traces)
        states.append(state)
    scores = torch.stack([
        experiment.smooth_tip_height(value) for value in states
    ], dim=1)
    weights = torch.softmax(scores / temperature, dim=1)
    message = weights[:, -1:] * experiment.height_gradient(states[-1])
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
                experiment.height_gradient(states[index - 1])
            )
    return torch.stack(routes, dim=1)


experiment.temporal_route = temporal_route


if __name__ == "__main__":
    experiment.main()
