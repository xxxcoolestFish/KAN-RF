"""Train a nonlinear ProtoKAN edge router and test it after cognition updates."""

from __future__ import annotations

import argparse
import copy
import json

import torch
import torch.nn.functional as F

from kanrf.protokan_causal_router import (
    ProtoKANNonlinearEdgeRouter,
    linear_causal_route,
    trace_protokan,
)
from physics_transfer.multifactor_data import _random_states
from physics_transfer.transition_data import sample_transition_batch
from scripts.stage15_real_outcome_replay import PRETRAIN_FACTOR
from scripts.stage23_multistep_terminal_value import (
    SimpleCognitiveKAN,
    smooth_tip_height,
)
from scripts.stage27_parameter_transport import pretrain_cognitive


TARGET_FACTOR = (13.475, 0.06, 0.90, 1.10)


def height_gradient(state):
    c1, s1, c2, s2 = state[:, 0], state[:, 1], state[:, 2], state[:, 3]
    zeros = torch.zeros_like(c1)
    return torch.stack([
        -1.0 - c2, s2, -c1, s1, zeros, zeros,
    ], dim=-1)


def rollout_score(cognitive, initial_state, actions, temperature):
    state = initial_state
    scores = []
    for index in range(actions.shape[1]):
        state = cognitive(state, actions[:, index])
        scores.append(smooth_tip_height(state))
    scores = torch.stack(scores, dim=1)
    return temperature * torch.logsumexp(scores / temperature, dim=1)


@torch.no_grad()
def finite_action_effect(cognitive, initial_state, actions, temperature,
                         delta):
    effects = []
    for index in range(actions.shape[1]):
        positive = actions.clone(); positive[:, index] += delta
        negative = actions.clone(); negative[:, index] -= delta
        effects.append((
            rollout_score(cognitive, initial_state, positive, temperature)
            - rollout_score(cognitive, initial_state, negative, temperature)
        ) / (2.0 * delta))
    return torch.stack(effects, dim=1).unsqueeze(-1)


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
    scores = torch.stack([smooth_tip_height(value) for value in states], dim=1)
    weights = torch.softmax(scores / temperature, dim=1)
    message = weights[:, -1:] * height_gradient(states[-1])
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
            message = message + weights[:, index - 1:] * height_gradient(
                states[index - 1]
            )
    return torch.stack(routes, dim=1)


def sample_route_batch(batch_size, horizon):
    states = _random_states(batch_size)
    actions = torch.empty(batch_size, horizon, 1).uniform_(-0.6, 0.6)
    return states, actions


def route_metrics(prediction, target):
    cosine = F.cosine_similarity(
        prediction.flatten(1), target.flatten(1), dim=1,
    )
    return {
        "mse": float(F.mse_loss(prediction, target)),
        "mean_cosine": float(cosine.mean()),
        "min_cosine": float(cosine.min()),
    }


@torch.no_grad()
def evaluate_router(cognitive, router, batch_size, horizon, temperature,
                    finite_delta, seed):
    torch.manual_seed(seed)
    states, actions = sample_route_batch(batch_size, horizon)
    target = finite_action_effect(
        cognitive, states, actions, temperature, finite_delta,
    )
    linear = temporal_route(cognitive, states, actions, temperature)
    nonlinear = temporal_route(
        cognitive, states, actions, temperature, router,
    )
    return {
        "linear": route_metrics(linear, target),
        "nonlinear": route_metrics(nonlinear, target),
    }


def train_router(cognitive, router, steps, batch_size, horizon, temperature,
                 finite_delta, seed):
    for parameter in cognitive.parameters():
        parameter.requires_grad = False
    optimizer = torch.optim.Adam(router.parameters(), lr=1e-3)
    torch.manual_seed(seed)
    losses = []
    for _ in range(steps):
        states, actions = sample_route_batch(batch_size, horizon)
        target = finite_action_effect(
            cognitive, states, actions, temperature, finite_delta,
        )
        prediction = temporal_route(
            cognitive, states, actions, temperature, router,
        )
        loss = F.mse_loss(prediction, target)
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(router.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    return {
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "mean_last_20_loss": sum(losses[-20:]) / min(20, len(losses)),
    }


def adapt_cognitive(cognitive, factor, steps, batch_size, seed):
    for parameter in cognitive.parameters():
        parameter.requires_grad = True
    optimizer = torch.optim.Adam(cognitive.parameters(), lr=2e-3)
    torch.manual_seed(seed)
    losses = []
    for _ in range(steps):
        batch = sample_transition_batch(batch_size, [factor])
        prediction = cognitive(batch["state"], batch["action"])
        loss = F.smooth_l1_loss(prediction, batch["next_state"])
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(cognitive.parameters(), 5.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    for parameter in cognitive.parameters():
        parameter.requires_grad = False
    return {
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "mean_last_20_loss": sum(losses[-20:]) / min(20, len(losses)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cognitive-steps", type=int, default=300)
    parser.add_argument("--router-steps", type=int, default=200)
    parser.add_argument("--target-update-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.08)
    parser.add_argument("--finite-delta", type=float, default=0.25)
    parser.add_argument("--edge-delta", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-out", type=str, default="")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    source_cognitive = SimpleCognitiveKAN(hidden_dim=32, n_prototypes=8)
    source_fit = pretrain_cognitive(
        source_cognitive, args.cognitive_steps, args.batch_size, args.seed,
    )
    target_cognitive = copy.deepcopy(source_cognitive)
    router = ProtoKANNonlinearEdgeRouter(delta=args.edge_delta)
    source_before = evaluate_router(
        source_cognitive, router, args.eval_batch, args.horizon,
        args.temperature, args.finite_delta, args.seed + 100,
    )
    router_fit = train_router(
        source_cognitive, router, args.router_steps, args.batch_size,
        args.horizon, args.temperature, args.finite_delta, args.seed + 200,
    )
    source_after = evaluate_router(
        source_cognitive, router, args.eval_batch, args.horizon,
        args.temperature, args.finite_delta, args.seed + 300,
    )
    target_before_update = evaluate_router(
        target_cognitive, router, args.eval_batch, args.horizon,
        args.temperature, args.finite_delta, args.seed + 400,
    )
    target_fit = adapt_cognitive(
        target_cognitive, TARGET_FACTOR, args.target_update_steps,
        args.batch_size, args.seed + 500,
    )
    target_after_update = evaluate_router(
        target_cognitive, router, args.eval_batch, args.horizon,
        args.temperature, args.finite_delta, args.seed + 600,
    )

    output = {
        "architecture": "ProtoKAN_NonlinearFunctionEdge_TemporalRouter",
        "source_factor": PRETRAIN_FACTOR[0],
        "target_factor": TARGET_FACTOR,
        "config": vars(args),
        "source_cognitive_fit": source_fit,
        "source_before_router_training": source_before,
        "router_fit": router_fit,
        "source_after_router_training": source_after,
        "target_before_cognitive_update": target_before_update,
        "target_cognitive_fit": target_fit,
        "target_after_cognitive_update": target_after_update,
    }
    text = json.dumps(output, ensure_ascii=False, indent=2)
    print(text)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")


if __name__ == "__main__":
    main()
