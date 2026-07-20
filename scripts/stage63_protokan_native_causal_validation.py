"""Validate native ProtoKAN function edges and temporal causal routing."""

from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F

from kanrf.protokan_causal_router import (
    ProtoKANNonlinearEdgeRouter,
    linear_causal_route,
    trace_protokan,
)
from physics_transfer.multifactor_data import _random_states
from scripts.stage15_real_outcome_replay import PRETRAIN_FACTOR
from scripts.stage23_multistep_terminal_value import (
    SimpleCognitiveKAN,
    smooth_tip_height,
)
from scripts.stage27_parameter_transport import pretrain_cognitive


def direct_input_gradient(network, inputs, output_message):
    local_inputs = inputs.detach().requires_grad_(True)
    output = network(local_inputs)
    gradient = torch.autograd.grad(
        (output * output_message).sum(), local_inputs,
    )[0]
    return output.detach(), gradient.detach()


def native_temporal_route(cognitive, initial_state, actions):
    state = initial_state
    traces = []
    for index in range(actions.shape[1]):
        network_input = torch.cat([state, actions[:, index]], dim=-1)
        state, edge_trace = trace_protokan(cognitive.network, network_input)
        traces.append(edge_trace)
    final_state = state
    local_final = final_state.detach().requires_grad_(True)
    final_score = smooth_tip_height(local_final)
    message = torch.autograd.grad(final_score.sum(), local_final)[0].detach()
    action_routes = [None] * actions.shape[1]
    for index in reversed(range(actions.shape[1])):
        input_message, _ = linear_causal_route(traces[index], message)
        message = input_message[:, :6]
        action_routes[index] = input_message[:, 6:]
    return final_state.detach(), torch.stack(action_routes, dim=1)


def direct_temporal_gradient(cognitive, initial_state, actions):
    local_actions = actions.detach().requires_grad_(True)
    state = initial_state.detach()
    for index in range(local_actions.shape[1]):
        state = cognitive(state, local_actions[:, index])
    score = smooth_tip_height(state)
    gradient = torch.autograd.grad(score.sum(), local_actions)[0]
    return state.detach(), gradient.detach()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cognitive-steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--validation-batch", type=int, default=16)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-out", type=str, default="")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    cognitive = SimpleCognitiveKAN(hidden_dim=32, n_prototypes=8)
    cognitive_fit = pretrain_cognitive(
        cognitive, args.cognitive_steps, args.batch_size, args.seed,
    )
    states = _random_states(args.validation_batch)
    actions = torch.empty(args.validation_batch, 1).uniform_(-1.0, 1.0)
    inputs = torch.cat([states, actions], dim=-1)
    output_message = torch.randn(args.validation_batch, 6)

    traced_output, traces = trace_protokan(
        cognitive.network, inputs, args.delta,
    )
    direct_output, direct_gradient = direct_input_gradient(
        cognitive.network, inputs, output_message,
    )
    native_gradient, _ = linear_causal_route(traces, output_message)
    nonlinear_router = ProtoKANNonlinearEdgeRouter(delta=args.delta)
    nonlinear_gradient, _ = nonlinear_router(traces, output_message)

    finite_errors, curvature_magnitudes = [], []
    for trace in traces:
        finite_errors.append(float((
            trace.positive_response - args.delta * trace.derivatives
        ).abs().mean()))
        curvature_magnitudes.append(float(trace.curvature.abs().mean()))

    for parameter in cognitive.parameters():
        parameter.grad = None
    edge_objective = sum(
        trace.values.square().mean()
        + trace.positive_response.square().mean()
        + trace.negative_response.square().mean()
        for trace in traces
    )
    edge_objective.backward()
    parameter_coverage = {
        name: bool(parameter.grad is not None and parameter.grad.abs().max() > 0)
        for name, parameter in cognitive.named_parameters()
    }

    temporal_actions = torch.empty(
        args.validation_batch, args.horizon, 1,
    ).uniform_(-0.5, 0.5)
    native_final, native_temporal = native_temporal_route(
        cognitive, states, temporal_actions,
    )
    direct_final, direct_temporal = direct_temporal_gradient(
        cognitive, states, temporal_actions,
    )
    temporal_cosine = F.cosine_similarity(
        native_temporal.flatten(1), direct_temporal.flatten(1), dim=1,
    )

    output = {
        "architecture": "ProtoKANNativeFunctionEdge_CausalRouter",
        "source_factor": PRETRAIN_FACTOR[0],
        "config": vars(args),
        "cognitive_fit": cognitive_fit,
        "one_step_validation": {
            "forward_max_error": float((traced_output - direct_output).abs().max()),
            "route_max_error": float((native_gradient - direct_gradient).abs().max()),
            "route_mean_cosine": float(F.cosine_similarity(
                native_gradient, direct_gradient, dim=1,
            ).mean()),
            "zero_initialized_nonlinear_router_error": float((
                nonlinear_gradient - native_gradient
            ).abs().max()),
        },
        "nonlinear_edge_evidence": {
            "mean_positive_finite_vs_linear_error_by_layer": finite_errors,
            "mean_absolute_curvature_by_layer": curvature_magnitudes,
        },
        "parameter_gradient_coverage": parameter_coverage,
        "all_cognitive_parameters_covered": all(parameter_coverage.values()),
        "temporal_validation": {
            "horizon": args.horizon,
            "final_state_max_error": float((native_final - direct_final).abs().max()),
            "route_max_error": float((native_temporal - direct_temporal).abs().max()),
            "route_mean_cosine": float(temporal_cosine.mean()),
            "route_min_cosine": float(temporal_cosine.min()),
        },
    }
    text = json.dumps(output, ensure_ascii=False, indent=2)
    print(text)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")


if __name__ == "__main__":
    main()
