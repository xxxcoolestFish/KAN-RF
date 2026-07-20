"""Oracle reachability check for hanging-down states in the target dynamics."""

from __future__ import annotations

import argparse
import json

import torch

from scripts import stage51_context_cognitive_ppo as base
from scripts.stage18_action_sequence_feasibility import optimize_batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-factor", type=float, nargs=4,
                        default=[13.475, 0.06, 0.90, 1.10])
    parser.add_argument("--test-count", type=int, default=16)
    parser.add_argument("--test-seed", type=int, default=20260720)
    parser.add_argument("--horizon", type=int, default=128)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--restarts", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json-out", type=str, default="")
    args = parser.parse_args()

    torch.manual_seed(args.test_seed)
    initial_states = base.reset_down_states(args.test_count)
    maximum, restart = optimize_batch(
        initial_states, tuple(args.target_factor), args.horizon, args.steps,
        args.restarts, args.seed, args.lr,
    )
    success = maximum >= 1.0
    output = {
        "target_factor": args.target_factor,
        "initial_state_distribution": "hanging_down_with_small_noise",
        "test_count": args.test_count,
        "horizon": args.horizon,
        "optimization_steps": args.steps,
        "restarts": args.restarts,
        "feasible_count_found": int(success.sum()),
        "feasible_rate_found": float(success.float().mean()),
        "mean_optimized_max_height": float(maximum.mean()),
        "optimized_max_height": maximum.tolist(),
        "best_restart": restart.tolist(),
        "note": "A found solution proves reachability; failure to find one would not prove impossibility.",
    }
    text = json.dumps(output, ensure_ascii=False, indent=2)
    print(text)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")


if __name__ == "__main__":
    main()
