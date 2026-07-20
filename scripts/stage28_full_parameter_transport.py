"""Validate full-dimensional (non-compressive) cognitive parameter transport."""

from __future__ import annotations

import argparse
import copy
import json
import math

import torch
import torch.nn.functional as F

from physics_transfer.full_parameter_transport import (
    FullParameterReceiverPolicy,
    FullParameterTransport,
)
from physics_transfer.multifactor_data import _random_states
from physics_transfer.transition_data import sample_transition_batch
from physics_transfer.variants import step
from scripts.stage15_real_outcome_replay import PRETRAIN_FACTOR
from scripts.stage2_lowrank_loss_ablation import FACTORS, HELDOUT
from scripts.stage23_multistep_terminal_value import (
    GOAL,
    SimpleCognitiveKAN,
    smooth_tip_height,
    task_cost,
)
from scripts.stage27_parameter_transport import pretrain_cognitive, prediction_error


def train_policy(policy, cognitive, goal, steps, batch_size, max_horizon, seed,
                 use_physics):
    for parameter in cognitive.parameters():
        parameter.requires_grad = False
    policy.transport.freeze()
    parameters = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=2e-3)
    torch.manual_seed(seed + (2800 if use_physics else 2900))
    losses, horizons = [], []
    levels = [h for h in (1, 2, 4, 8) if h <= max_horizon]
    for index in range(steps):
        horizon = levels[min(len(levels) - 1,
                             int(index / max(1, steps - 1) * len(levels)))]
        current = _random_states(batch_size)
        costs, actions = [], []
        for _ in range(horizon):
            action = policy(current, goal, use_physics=use_physics)
            next_state = cognitive(current, action)
            costs.append(task_cost(next_state, action))
            actions.append(action)
            current = next_state
        discounted = torch.stack(
            [(0.95 ** t) * value for t, value in enumerate(costs)], dim=1,
        ).sum(dim=1)
        action_stack = torch.stack(actions, dim=1)
        smooth = (
            (action_stack[:, 1:] - action_stack[:, :-1]).square().mean()
            if horizon > 1 else torch.zeros((), dtype=current.dtype)
        )
        loss = discounted.mean() + 0.02 * smooth
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 5.0)
        optimizer.step()
        losses.append(loss.item()); horizons.append(horizon)
    return {
        "first_loss": float(losses[0]),
        "last_loss": float(losses[-1]),
        "mean_last_20_loss": float(sum(losses[-20:]) / min(20, len(losses))),
        "final_horizon": horizons[-1],
    }


@torch.no_grad()
def evaluate(policy, states, factor, goal, rollout_steps, use_physics):
    current = states.detach().clone()
    factor_tensor = torch.tensor(factor, dtype=current.dtype).view(1, 4)
    factor_tensor = factor_tensor.expand(current.shape[0], -1)
    maxima = torch.full((current.shape[0],), -float("inf"))
    actions = []
    for _ in range(rollout_steps):
        action = policy(current, goal, use_physics=use_physics)
        actions.append(action)
        current = step(
            current, action, factor_tensor[:, 0], factor_tensor[:, 1],
            factor_tensor[:, 2], factor_tensor[:, 3],
        )
        maxima = torch.maximum(maxima, smooth_tip_height(current))
    success = maxima >= 1.0
    return {
        "success_count": int(success.sum().item()),
        "success_rate": float(success.float().mean().item()),
        "mean_max_height": float(maxima.mean().item()),
        "mean_abs_action": float(torch.stack(actions, 1).abs().mean().item()),
    }


def adapt_cognitive(cognitive, factor, steps, batch_size, seed):
    for parameter in cognitive.parameters():
        parameter.requires_grad = True
    optimizer = torch.optim.Adam(cognitive.parameters(), lr=1e-3)
    torch.manual_seed(seed + 2810)
    losses = []
    for _ in range(steps):
        batch = sample_transition_batch(batch_size, (factor,))
        prediction = cognitive(batch["state"], batch["action"])
        loss = F.smooth_l1_loss(prediction, batch["next_state"])
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(cognitive.parameters(), 5.0)
        optimizer.step(); losses.append(loss.item())
    return {
        "first_loss": float(losses[0]),
        "last_loss": float(losses[-1]),
        "mean_last_20_loss": float(sum(losses[-20:]) / min(20, len(losses))),
    }


def factor_distance(source, factor):
    scale = (5.0, 0.05, 0.2, 0.2)
    return float(math.sqrt(sum(((a - b) / s) ** 2
                               for a, b, s in zip(source, factor, scale))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cognitive-steps", type=int, default=150)
    parser.add_argument("--policy-steps", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--max-horizon", type=int, default=4)
    parser.add_argument("--rollout-steps", type=int, default=48)
    parser.add_argument("--test-count", type=int, default=20)
    parser.add_argument("--adapt-steps", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-seed", type=int, default=20260718)
    parser.add_argument("--json-out", type=str, default="")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    cognitive = SimpleCognitiveKAN()
    cognitive_fit = pretrain_cognitive(
        cognitive, args.cognitive_steps, args.batch_size, args.seed,
    )
    transport = FullParameterTransport(cognitive)
    full_policy = FullParameterReceiverPolicy(cognitive, transport)
    plain_policy = copy.deepcopy(full_policy)
    goal = GOAL.view(1, -1)
    full_fit = train_policy(
        full_policy, cognitive, goal, args.policy_steps, args.batch_size,
        args.max_horizon, args.seed, True,
    )
    plain_fit = train_policy(
        plain_policy, plain_policy.cognitive, goal, args.policy_steps,
        args.batch_size, args.max_horizon, args.seed, False,
    )
    generator = torch.Generator().manual_seed(args.test_seed)
    states = _random_states(args.test_count, generator=generator)
    source = PRETRAIN_FACTOR[0]
    factors = [("source", source)]
    factors += [(f"factor_{i + 1}", factor)
                for i, factor in enumerate(FACTORS[1:])]
    factors += [(f"heldout_{i + 1}", factor)
                for i, factor in enumerate(HELDOUT)]
    results = []
    for label, factor in factors:
        results.append({
            "label": label,
            "factor": factor,
            "normalized_distance_from_source": factor_distance(source, factor),
            "cognitive_prediction": prediction_error(cognitive, factor),
            "full_transport_policy": evaluate(
                full_policy, states, factor, goal, args.rollout_steps, True,
            ),
            "plain_policy": evaluate(
                plain_policy, states, factor, goal, args.rollout_steps, False,
            ),
        })
    changed = HELDOUT[0]
    adapted_cognitive = copy.deepcopy(cognitive)
    before_parameters = full_policy.transported_parameters().detach()
    adaptation_fit = adapt_cognitive(
        adapted_cognitive, changed, args.adapt_steps, args.batch_size, args.seed,
    )
    adapted_policy = copy.deepcopy(full_policy)
    adapted_policy.cognitive = adapted_cognitive
    after_parameters = adapted_policy.transported_parameters().detach()
    before_actions = full_policy(states, goal, use_physics=True)
    after_actions = adapted_policy(states, goal, use_physics=True)
    online_result = {
        "factor": changed,
        "prediction_before": prediction_error(cognitive, changed),
        "prediction_after": prediction_error(adapted_cognitive, changed),
        "adaptation_fit": adaptation_fit,
        "transported_parameter_shift_l2": float(
            torch.linalg.vector_norm(after_parameters - before_parameters).item()
        ),
        "action_shift_mean_abs": float((after_actions - before_actions).abs().mean().item()),
        "action_shift_max_abs": float((after_actions - before_actions).abs().max().item()),
        "decision_before": evaluate(
            full_policy, states, changed, goal, args.rollout_steps, True,
        ),
        "decision_after": evaluate(
            adapted_policy, states, changed, goal, args.rollout_steps, True,
        ),
    }
    output = {
        "architecture": "FullDimensionalParameterTransportPolicy",
        "training_factor": source,
        "cognitive_parameter_count": transport.theta_dim,
        "transport_dimension": transport.theta_dim,
        "compression_ratio": 1.0,
        "runtime_action_probes": "none",
        "cognitive_loss_and_decision_loss_separate": True,
        "cognitive_fit": cognitive_fit,
        "full_policy_fit": full_fit,
        "plain_policy_fit": plain_fit,
        "results": results,
        "online_parameter_refresh": online_result,
        "test_seed": args.test_seed,
    }
    text = json.dumps(output, ensure_ascii=False, indent=2)
    print(text)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")


if __name__ == "__main__":
    main()
