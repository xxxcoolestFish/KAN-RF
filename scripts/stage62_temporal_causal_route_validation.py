"""Validate explicit temporal causal routing against full model gradients."""

from __future__ import annotations

import argparse
import json

import torch

from kanrf.temporal_causal_routing import (
    direct_reachability_gradient,
    repeated_transition,
    temporal_reachability_route,
    unroll_influence_graph,
)
from physics_transfer.variants import step
from scripts import stage51_context_cognitive_ppo as base


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-factor", type=float, nargs=4,
                        default=[13.475, 0.06, 0.90, 1.10])
    parser.add_argument("--source-json", type=str,
                        default="results/stage61d_target_cem_feasibility.json")
    parser.add_argument("--route-steps", type=int, default=12)
    parser.add_argument("--route-lr", type=float, default=0.12)
    parser.add_argument("--temperature", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--json-out", type=str, default="")
    args = parser.parse_args()

    with open(args.source_json, "r", encoding="utf-8") as handle:
        source = json.load(handle)
    heights = torch.tensor(source["best_height"])
    selected = int(heights.argmax())
    action_repeat = int(source["action_repeat"])
    actions = torch.tensor(source["best_action_blocks"])[selected:selected + 1]
    actions = actions.unsqueeze(-1)

    torch.manual_seed(args.seed)
    initial_states = base.reset_down_states(int(source["test_count"]))
    initial_state = initial_states[selected:selected + 1]
    factor = tuple(args.target_factor)

    def physical_transition(state, action):
        return step(state, action, *factor)

    transition = repeated_transition(physical_transition, action_repeat)
    score_fn = base.tip_height

    states, state_matrices, action_matrices = unroll_influence_graph(
        transition, initial_state, actions,
    )
    route, route_score, weights = temporal_reachability_route(
        states, state_matrices, action_matrices, score_fn, args.temperature,
    )
    direct, direct_score = direct_reachability_gradient(
        transition, initial_state, actions, score_fn, args.temperature,
    )
    cosine = torch.nn.functional.cosine_similarity(
        route.flatten(1), direct.flatten(1), dim=1,
    )
    max_error = (route - direct).abs().max()

    logits = torch.atanh(actions.clamp(-0.999, 0.999))
    route_history = []
    for route_step in range(args.route_steps + 1):
        routed_actions = torch.tanh(logits)
        states, state_matrices, action_matrices = unroll_influence_graph(
            transition, initial_state, routed_actions,
        )
        influence, smooth_score, _ = temporal_reachability_route(
            states, state_matrices, action_matrices, score_fn, args.temperature,
        )
        maximum_height = torch.stack(
            [score_fn(states[:, index]) for index in range(1, states.shape[1])],
            dim=1,
        ).max(dim=1).values
        route_history.append({
            "route_step": route_step,
            "smooth_score": float(smooth_score.item()),
            "maximum_height": float(maximum_height.item()),
            "success": bool(maximum_height.item() >= 1.0),
        })
        if route_step < args.route_steps:
            logit_gradient = influence * (1.0 - routed_actions.square())
            scale = logit_gradient.square().mean().sqrt().clamp_min(1e-8)
            logits = (logits + args.route_lr * logit_gradient / scale).detach()

    output = {
        "target_factor": args.target_factor,
        "source_sequence": args.source_json,
        "selected_initial_state": selected,
        "action_blocks": int(actions.shape[1]),
        "action_repeat": action_repeat,
        "graph_composition_validation": {
            "explicit_route_score": float(route_score.item()),
            "direct_autograd_score": float(direct_score.item()),
            "gradient_cosine_similarity": float(cosine.item()),
            "maximum_absolute_gradient_error": float(max_error.item()),
        },
        "route_refinement": route_history,
    }
    text = json.dumps(output, ensure_ascii=False, indent=2)
    print(text)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")


if __name__ == "__main__":
    main()
