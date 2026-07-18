"""Stage 16 with deterministic evaluation before and after online learning."""

from __future__ import annotations

import argparse
import json

import torch

import scripts.stage16_decision_only_pretrained_env as stage16
from physics_transfer.multifactor_data import _random_states
from scripts.stage13_online_task_loss_adaptation import RuntimeTaskDecision, initial_operator
from scripts.stage13_online_task_loss_adaptation import initialize_decision
from scripts.stage15_real_outcome_replay import OutcomeReplay
from scripts.stage15_real_outcome_replay import PRETRAIN_FACTOR, pretrain


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cognitive-steps", type=int, default=100)
    parser.add_argument("--base-steps", type=int, default=100)
    parser.add_argument("--mapper-steps", type=int, default=100)
    parser.add_argument("--sequence-steps", type=int, default=16)
    parser.add_argument("--replay-capacity", type=int, default=4096)
    parser.add_argument("--replay-batch", type=int, default=128)
    parser.add_argument("--updates-per-episode", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--return-temperature", type=float, default=1.0)
    parser.add_argument("--trust-radius", type=float, default=0.05)
    parser.add_argument("--exploration-noise", type=float, default=0.10)
    parser.add_argument("--rollout-steps", type=int, default=64)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cognitive = pretrain(args.cognitive_steps, args.sequence_steps, args.seed)
    initialized = initialize_decision(
        cognitive, args.sequence_steps, args.base_steps, args.mapper_steps, args.seed
    )
    q_const = initial_operator(cognitive, args.sequence_steps, args.seed).detach()
    decision = RuntimeTaskDecision(initialized)
    references = [parameter.detach().clone() for parameter in decision.runtime_residual.parameters()]
    optimizer = torch.optim.Adam(decision.runtime_residual.parameters(), lr=5e-4)
    replay = OutcomeReplay(args.replay_capacity)

    initial = [stage16.run_episode(
        decision, q_const, replay, optimizer, references,
        PRETRAIN_FACTOR[0], args, args.seed + 4000 + i, False,
    ) for i in range(args.episodes)]
    training = [stage16.run_episode(
        decision, q_const, replay, optimizer, references,
        PRETRAIN_FACTOR[0], args, args.seed + 1000 + i, True,
    ) for i in range(args.episodes)]
    final = [stage16.run_episode(
        decision, q_const, replay, optimizer, references,
        PRETRAIN_FACTOR[0], args, args.seed + 5000 + i, False,
    ) for i in range(args.episodes)]
    print(json.dumps({
        "environment": PRETRAIN_FACTOR[0],
        "cognitive_usage": "pretraining_and_one_time_initialization_only",
        "initial_deterministic": stage16.summarize(initial),
        "online_training_with_exploration": stage16.summarize(training),
        "final_deterministic": stage16.summarize(final),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
