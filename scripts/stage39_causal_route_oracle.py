"""Validate goal-directed causal routing before policy training."""

from __future__ import annotations

import argparse
import json

import torch

from physics_transfer.causal_route import (
    discounted_goal_cost,
    route_action,
    route_score,
)
from physics_transfer.multifactor_data import _random_states
from physics_transfer.variants import step
from scripts.stage15_real_outcome_replay import PRETRAIN_FACTOR
from scripts.stage23_multistep_terminal_value import (
    GOAL,
    SimpleCognitiveKAN,
    smooth_tip_height,
)
from scripts.stage27_parameter_transport import pretrain_cognitive
from scripts.stage34_real_transition_goal_loss import goal_potential


def exact_transition(state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    factor = torch.tensor(
        PRETRAIN_FACTOR[0], dtype=state.dtype, device=state.device,
    ).view(1, 4).expand(state.shape[0], -1)
    return step(state, action, factor[:, 0], factor[:, 1], factor[:, 2], factor[:, 3])


def actual_directional_score(
    transition,
    state: torch.Tensor,
    actions: torch.Tensor,
    goal: torch.Tensor,
    gamma: float,
    epsilon: float,
) -> torch.Tensor:
    plus = actions.clone(); minus = actions.clone()
    plus[:, 0, 0] += epsilon; minus[:, 0, 0] -= epsilon
    plus_cost = discounted_goal_cost(
        transition, state, plus, goal, gamma, goal_potential,
    )
    minus_cost = discounted_goal_cost(
        transition, state, minus, goal, gamma, goal_potential,
    )
    return -(plus_cost - minus_cost) / (2.0 * epsilon)


def vector_cosine(first: torch.Tensor, second: torch.Tensor) -> float:
    return float((first * second).sum().item() / (
        first.norm().item() * second.norm().item() + 1e-8
    ))


def route_direction_metrics(
    transition,
    state: torch.Tensor,
    actions: torch.Tensor,
    goal: torch.Tensor,
    gamma: float,
    epsilon: float,
) -> dict:
    predicted = route_score(
        transition, goal_potential, state, actions, goal, gamma, epsilon,
    ).detach()
    actual = actual_directional_score(
        exact_transition, state, actions, goal, gamma, epsilon,
    ).detach()
    active = actual.abs() > 1e-6
    sign_agreement = (
        (predicted[active].sign() == actual[active].sign()).float().mean()
        if active.any() else torch.zeros((), dtype=state.dtype)
    )
    return {
        "route_score_mean": float(predicted.mean().item()),
        "actual_score_mean": float(actual.mean().item()),
        "route_actual_cosine": vector_cosine(predicted, actual),
        "sign_agreement": float(sign_agreement.item()),
        "active_fraction": float(active.float().mean().item()),
    }


def route_controller(
    transition,
    states: torch.Tensor,
    goal: torch.Tensor,
    horizon: int,
    rollout_steps: int,
    epsilon: float,
    action_limit: float,
    score_scale: float,
) -> dict:
    current = states.detach().clone()
    factor = torch.tensor(
        PRETRAIN_FACTOR[0], dtype=current.dtype, device=current.device,
    ).view(1, 4).expand(states.shape[0], -1)
    maxima = torch.full((states.shape[0],), -float("inf"))
    actions = []
    for _ in range(rollout_steps):
        sequence = torch.zeros(
            current.shape[0], horizon, 1, dtype=current.dtype,
        )
        score = route_score(
            transition, goal_potential, current, sequence, goal,
            gamma=0.95, epsilon=epsilon,
        ).detach()
        action = route_action(score, action_limit, score_scale)
        actions.append(action)
        current = step(
            current, action, factor[:, 0], factor[:, 1],
            factor[:, 2], factor[:, 3],
        )
        maxima = torch.maximum(maxima, smooth_tip_height(current))
    success = maxima >= 1.0
    return {
        "success_count": int(success.sum().item()),
        "success_rate": float(success.float().mean().item()),
        "mean_max_height": float(maxima.mean().item()),
        "mean_final_goal_potential": float(goal_potential(current, goal).mean().item()),
        "mean_abs_action": float(torch.stack(actions, dim=1).abs().mean().item()),
    }


def run_seed(args: argparse.Namespace, seed: int) -> dict:
    torch.manual_seed(seed)
    cognitive = SimpleCognitiveKAN()
    cognitive_fit = pretrain_cognitive(
        cognitive, args.cognitive_steps, args.batch_size, seed,
    )
    states = _random_states(
        args.probe_count,
        generator=torch.Generator().manual_seed(args.test_seed + seed),
    )
    goal = GOAL.view(1, -1)
    output = {"seed": seed, "cognitive_fit": cognitive_fit, "horizons": {}}
    for horizon in args.horizons:
        actions = torch.zeros(args.probe_count, horizon, 1)
        output["horizons"][f"H{horizon}"] = {
            "oracle_direction": route_direction_metrics(
                exact_transition, states, actions, goal,
                args.gamma, args.epsilon,
            ),
            "cognitive_direction": route_direction_metrics(
                cognitive, states, actions, goal,
                args.gamma, args.epsilon,
            ),
            "oracle_controller": route_controller(
                exact_transition, states, goal, horizon, args.rollout_steps,
                args.epsilon, args.action_limit, args.score_scale,
            ),
            "cognitive_controller": route_controller(
                cognitive, states, goal, horizon, args.rollout_steps,
                args.epsilon, args.action_limit, args.score_scale,
            ),
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cognitive-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--probe-count", type=int, default=100)
    parser.add_argument("--test-seed", type=int, default=20260719)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--rollout-steps", type=int, default=48)
    parser.add_argument("--action-limit", type=float, default=0.9)
    parser.add_argument("--score-scale", type=float, default=0.005)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--json-out", type=str, default="")
    args = parser.parse_args()
    output = {
        "architecture": "TemporalCausalRouteOracle",
        "experiment": "oracle_vs_cognitive_route_direction_and_control",
        "source_factor": PRETRAIN_FACTOR[0],
        "config": vars(args),
        "seeds": [run_seed(args, seed) for seed in args.seeds],
    }
    text = json.dumps(output, ensure_ascii=False, indent=2)
    print(text)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")


if __name__ == "__main__":
    main()
