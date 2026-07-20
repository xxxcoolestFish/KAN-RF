"""Validate explicit parameter-sensitivity regularization."""

from __future__ import annotations

import argparse
import copy
import json
import math

import torch
import torch.nn.functional as F

from physics_transfer.full_parameter_transport import FullParameterTransport
from physics_transfer.multifactor_data import _random_states
from physics_transfer.sensitivity_policy import (
    SensitivityMandatoryPolicy,
    parameter_sensitivity_loss,
)
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


def fit_policy(policy, cognitive, goal, steps, batch_size, max_horizon, seed,
               sensitivity_weight):
    for parameter in cognitive.parameters():
        parameter.requires_grad = False
    policy.transport.freeze()
    parameters = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=2e-3)
    torch.manual_seed(seed + (3000 if sensitivity_weight else 3100))
    losses, sens_values = [], []
    levels = [h for h in (1, 2, 4, 8) if h <= max_horizon]
    for index in range(steps):
        horizon = levels[min(len(levels) - 1,
                             int(index / max(1, steps - 1) * len(levels)))]
        current = _random_states(batch_size)
        costs, actions = [], []
        for _ in range(horizon):
            action = policy(current, goal)
            next_state = cognitive(current, action)
            costs.append(task_cost(next_state, action))
            actions.append(action)
            current = next_state
        discounted = torch.stack(
            [(0.95 ** t) * value for t, value in enumerate(costs)], dim=1,
        ).sum(dim=1)
        stack = torch.stack(actions, dim=1)
        smooth = ((stack[:, 1:] - stack[:, :-1]).square().mean()
                  if horizon > 1 else torch.zeros((), dtype=current.dtype))
        task_loss = discounted.mean() + 0.02 * smooth
        if sensitivity_weight:
            sens_loss, sens_stats = parameter_sensitivity_loss(
                policy, current.detach(), goal, epsilon=0.05,
                target_response=0.01,
            )
            sens_values.append(float(sens_stats["response"].item()))
            loss = task_loss + sensitivity_weight * sens_loss
        else:
            loss = task_loss
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 5.0)
        optimizer.step(); losses.append(loss.item())
    return {
        "first_loss": float(losses[0]),
        "last_loss": float(losses[-1]),
        "mean_last_20_loss": float(sum(losses[-20:]) / min(20, len(losses))),
        "mean_training_sensitivity": float(sum(sens_values) / len(sens_values)) if sens_values else None,
    }


@torch.no_grad()
def evaluate(policy, states, factor, goal, rollout_steps):
    current = states.detach().clone()
    f = torch.tensor(factor, dtype=current.dtype).view(1, 4).expand(current.shape[0], -1)
    maxima = torch.full((current.shape[0],), -float("inf")); actions = []
    for _ in range(rollout_steps):
        action = policy(current, goal); actions.append(action)
        current = step(current, action, f[:, 0], f[:, 1], f[:, 2], f[:, 3])
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
    torch.manual_seed(seed + 3200); losses = []
    for _ in range(steps):
        batch = sample_transition_batch(batch_size, (factor,))
        loss = F.smooth_l1_loss(
            cognitive(batch["state"], batch["action"]), batch["next_state"],
        )
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(cognitive.parameters(), 5.0)
        optimizer.step(); losses.append(loss.item())
    return {
        "first_loss": float(losses[0]), "last_loss": float(losses[-1]),
        "mean_last_20_loss": float(sum(losses[-20:]) / min(20, len(losses))),
    }


def factor_distance(source, factor):
    scale = (5.0, 0.05, 0.2, 0.2)
    return float(math.sqrt(sum(((a - b) / s) ** 2
                               for a, b, s in zip(source, factor, scale))))


def action_sensitivity(policy, states, goal, seed):
    torch.manual_seed(seed)
    _, stats = parameter_sensitivity_loss(
        policy, states, goal, epsilon=0.05, target_response=0.0,
    )
    return float(stats["response"].item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cognitive-steps", type=int, default=120)
    parser.add_argument("--policy-steps", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-horizon", type=int, default=4)
    parser.add_argument("--rollout-steps", type=int, default=48)
    parser.add_argument("--test-count", type=int, default=20)
    parser.add_argument("--adapt-steps", type=int, default=60)
    parser.add_argument("--sensitivity-weight", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-seed", type=int, default=20260718)
    parser.add_argument("--json-out", type=str, default="")
    args = parser.parse_args(); torch.manual_seed(args.seed)
    cognitive = SimpleCognitiveKAN()
    cognitive_fit = pretrain_cognitive(
        cognitive, args.cognitive_steps, args.batch_size, args.seed,
    )
    transport = FullParameterTransport(cognitive)
    no_sens = SensitivityMandatoryPolicy(cognitive, transport)
    with_sens = copy.deepcopy(no_sens)
    goal = GOAL.view(1, -1)
    no_sens_fit = fit_policy(
        no_sens, cognitive, goal, args.policy_steps, args.batch_size,
        args.max_horizon, args.seed, 0.0,
    )
    with_sens_fit = fit_policy(
        with_sens, cognitive, goal, args.policy_steps, args.batch_size,
        args.max_horizon, args.seed, args.sensitivity_weight,
    )
    states = _random_states(
        args.test_count, generator=torch.Generator().manual_seed(args.test_seed),
    )
    source = PRETRAIN_FACTOR[0]
    factors = [("source", source)]
    factors += [(f"factor_{i + 1}", f) for i, f in enumerate(FACTORS[1:])]
    factors += [(f"heldout_{i + 1}", f) for i, f in enumerate(HELDOUT)]
    results = []
    for label, factor in factors:
        results.append({
            "label": label, "factor": factor,
            "normalized_distance_from_source": factor_distance(source, factor),
            "cognitive_prediction": prediction_error(cognitive, factor),
            "no_sensitivity_loss": evaluate(no_sens, states, factor, goal, args.rollout_steps),
            "with_sensitivity_loss": evaluate(with_sens, states, factor, goal, args.rollout_steps),
        })
    changed = HELDOUT[0]
    adapted_cognitive = copy.deepcopy(cognitive)
    before = with_sens.transported_parameters().detach()
    adaptation_fit = adapt_cognitive(
        adapted_cognitive, changed, args.adapt_steps, args.batch_size, args.seed,
    )
    adapted = copy.deepcopy(with_sens); adapted.cognitive = adapted_cognitive
    after = adapted.transported_parameters().detach()
    before_action = with_sens(states, goal); after_action = adapted(states, goal)
    output = {
        "architecture": "FullParameterCouplingWithSensitivityLoss",
        "training_factor": source,
        "cognitive_parameter_count": transport.theta_dim,
        "transport_dimension": transport.theta_dim,
        "compression_ratio": 1.0,
        "sensitivity_weight": args.sensitivity_weight,
        "sensitivity_definition": "bounded finite-difference response to a block perturbation",
        "cognitive_fit": cognitive_fit,
        "no_sensitivity_policy_fit": no_sens_fit,
        "with_sensitivity_policy_fit": with_sens_fit,
        "sensitivity_before_eval": action_sensitivity(no_sens, states, goal, args.seed),
        "sensitivity_after_eval": action_sensitivity(with_sens, states, goal, args.seed),
        "results": results,
        "online_parameter_refresh": {
            "factor": changed,
            "prediction_before": prediction_error(cognitive, changed),
            "prediction_after": prediction_error(adapted_cognitive, changed),
            "adaptation_fit": adaptation_fit,
            "transported_parameter_shift_l2": float(torch.linalg.vector_norm(after - before).item()),
            "action_shift_mean_abs": float((after_action - before_action).abs().mean().item()),
            "action_shift_max_abs": float((after_action - before_action).abs().max().item()),
            "decision_before": evaluate(with_sens, states, changed, goal, args.rollout_steps),
            "decision_after": evaluate(adapted, states, changed, goal, args.rollout_steps),
        },
        "test_seed": args.test_seed,
    }
    text = json.dumps(output, ensure_ascii=False, indent=2); print(text)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")


if __name__ == "__main__":
    main()
