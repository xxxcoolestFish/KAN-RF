"""Seed-matched wrapper for the cognition-obstruction ablation."""

from __future__ import annotations

import argparse
import json

from scripts.stage42_cognitive_obstruction_ablation import train_condition
from scripts.stage15_real_outcome_replay import PRETRAIN_FACTOR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cognitive-steps", type=int, default=200)
    parser.add_argument("--cognitive-batch", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--rollout-horizon", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--minibatch", type=int, default=256)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=64)
    parser.add_argument("--test-count", type=int, default=64)
    parser.add_argument("--test-seed", type=int, default=20260719)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--json-out", type=str, default="")
    args = parser.parse_args()
    conditions = ("direct", "identity_receiver", "random_proto", "trained_proto")
    # Crucially, use identical base seeds for every condition.
    results = [
        train_condition(args, condition, seed)
        for condition in conditions
        for seed in args.seeds
    ]
    output = {
        "architecture": "CognitiveParameterObstructionAblationSeedMatched",
        "source_factor": PRETRAIN_FACTOR[0],
        "conditions": conditions,
        "config": vars(args),
        "results": results,
    }
    text = json.dumps(output, ensure_ascii=False, indent=2)
    print(text)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")


if __name__ == "__main__":
    main()
