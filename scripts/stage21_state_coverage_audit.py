"""Audit hard-state exploration coverage and operator-code variability.

This is a read-only diagnostic after one-time initialization.  It does not
update the decision network and does not query an action teacher during the
probe.  Each fixed initial state is rolled out repeatedly with different
amounts of action noise to determine whether successful trajectories are
available to the current exploration mechanism.
"""

from __future__ import annotations

import argparse
import json

import torch

from physics_transfer.multifactor_data import _random_states
from physics_transfer.transition_data import sample_transition_sequence_batch
from physics_transfer.variants import step
from scripts.stage13_online_task_loss_adaptation import RuntimeTaskDecision, initialize_decision, initial_operator, tip_height
from scripts.stage15_real_outcome_replay import PRETRAIN_FACTOR, pretrain
from scripts.stage7_single_env_decision_adaptation import operator_query


def state_metadata(states):
    theta1 = torch.atan2(states[:, 1], states[:, 0])
    theta2 = torch.atan2(states[:, 3], states[:, 2])
    return [
        {
            "index": index,
            "theta1": theta1[index].item(),
            "theta2": theta2[index].item(),
            "velocity1": states[index, 4].item(),
            "velocity2": states[index, 5].item(),
            "initial_height": tip_height(states[index:index + 1]).item(),
        }
        for index in range(states.shape[0])
    ]


def probe_state(actor, q_const, state, factor, noise, attempts, seed, steps):
    factor_tensor = torch.tensor(factor, dtype=state.dtype).view(1, 4)
    successes, maxima = [], []
    for attempt in range(attempts):
        torch.manual_seed(seed + attempt)
        current = state.detach().clone()
        maximum = -float("inf")
        for _ in range(steps):
            operator = q_const
            with torch.no_grad():
                policy_action = actor(current, operator)["action"]
            action = (policy_action + noise * torch.randn_like(policy_action)).clamp(-1.0, 1.0)
            current = step(
                current, action,
                factor_tensor[:, 0], factor_tensor[:, 1],
                factor_tensor[:, 2], factor_tensor[:, 3],
            )
            maximum = max(maximum, tip_height(current).item())
        maxima.append(maximum)
        successes.append(maximum >= 1.0)
    return {
        "success_count": sum(successes),
        "success_rate": sum(successes) / len(successes),
        "mean_max_height": sum(maxima) / len(maxima),
        "best_max_height": max(maxima),
    }


def operator_audit(cognitive, sequence_steps, seed, samples):
    torch.manual_seed(seed + 20000)
    batch = sample_transition_sequence_batch(samples, sequence_steps, PRETRAIN_FACTOR)
    with torch.no_grad():
        output = cognitive.forward_sequence(batch["state"], batch["action"], batch["next_state"])
        index = sequence_steps // 2
        operators = operator_query(
            cognitive,
            batch["state"][:, index],
            output["pre_latents"][:, index],
        )
    q_mean = operators.mean(dim=0, keepdim=True)
    centered = operators - q_mean
    norms = operators.norm(dim=1)
    cosine = (operators * q_mean).sum(dim=1) / (norms * q_mean.norm() + 1e-8)
    return {
        "samples": samples,
        "operator_dim": operators.shape[1],
        "mean_norm": norms.mean().item(),
        "mean_coordinate_std": operators.std(dim=0).mean().item(),
        "max_coordinate_std": operators.std(dim=0).max().item(),
        "mean_cosine_to_mean": cosine.mean().item(),
        "cosine_std_to_mean": cosine.std().item(),
        "mean_code": q_mean.squeeze(0).tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cognitive-steps", type=int, default=100)
    parser.add_argument("--base-steps", type=int, default=100)
    parser.add_argument("--mapper-steps", type=int, default=100)
    parser.add_argument("--sequence-steps", type=int, default=16)
    parser.add_argument("--test-count", type=int, default=20)
    parser.add_argument("--test-seed", type=int, default=20260718)
    parser.add_argument("--attempts", type=int, default=16)
    parser.add_argument("--noise-levels", type=float, nargs="+", default=[0.10, 0.20, 0.40])
    parser.add_argument("--rollout-steps", type=int, default=64)
    parser.add_argument("--operator-samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cognitive = pretrain(args.cognitive_steps, args.sequence_steps, args.seed)
    initialized = initialize_decision(
        cognitive, args.sequence_steps, args.base_steps, args.mapper_steps, args.seed
    )
    q_const = initial_operator(cognitive, args.sequence_steps, args.seed).detach()
    actor = RuntimeTaskDecision(initialized)
    generator = torch.Generator().manual_seed(args.test_seed)
    states = _random_states(args.test_count, generator=generator)
    factor = PRETRAIN_FACTOR[0]

    deterministic = [probe_state(actor, q_const, states[index:index + 1], factor, 0.0, 1,
                                 args.seed + index * 1000, args.rollout_steps)
                     for index in range(args.test_count)]
    exploration = {}
    for noise in args.noise_levels:
        exploration[str(noise)] = [
            probe_state(actor, q_const, states[index:index + 1], factor, noise, args.attempts,
                        args.seed + index * 1000 + int(noise * 10000), args.rollout_steps)
            for index in range(args.test_count)
        ]
    print(json.dumps({
        "environment": factor,
        "test_count": args.test_count,
        "test_seed": args.test_seed,
        "state_metadata": state_metadata(states),
        "deterministic_probe": deterministic,
        "exploration_probe": exploration,
        "operator_audit": operator_audit(cognitive, args.sequence_steps, args.seed, args.operator_samples),
        "note": "The probe does not update the actor and does not use an online action teacher.",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

