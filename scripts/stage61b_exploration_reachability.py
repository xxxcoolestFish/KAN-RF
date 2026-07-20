"""Measure how often uninformed action sequences reach the target environment goal."""

from __future__ import annotations

import argparse
import json

import torch

from physics_transfer.variants import step
from scripts import stage51_context_cognitive_ppo as base


@torch.no_grad()
def random_reachability(factor, count, physical_steps, persistence, seed,
                        bang_bang=False):
    torch.manual_seed(seed)
    state = base.reset_down_states(count)
    factor_tensor = torch.tensor(factor).view(1, 4).expand(count, -1)
    maximum = torch.full((count,), -float("inf"))
    success = torch.zeros(count, dtype=torch.bool)
    action = torch.zeros(count, 1)
    for index in range(physical_steps):
        if index % persistence == 0:
            raw = torch.randn(count, 1)
            action = raw.sign() if bang_bang else torch.tanh(raw)
        state = step(
            state, action, factor_tensor[:, 0], factor_tensor[:, 1],
            factor_tensor[:, 2], factor_tensor[:, 3],
        )
        height = base.tip_height(state)
        maximum = torch.maximum(maximum, height)
        success |= height >= 1.0
    quantiles = torch.quantile(maximum, torch.tensor([0.5, 0.9, 0.99])).tolist()
    return {
        "persistence": persistence,
        "distribution": "bang_bang" if bang_bang else "tanh_normal",
        "success_count": int(success.sum()),
        "success_rate": float(success.float().mean()),
        "mean_max_height": float(maximum.mean()),
        "max_height_quantiles": quantiles,
        "maximum_height": float(maximum.max()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-factor", type=float, nargs=4,
                        default=[13.475, 0.06, 0.90, 1.10])
    parser.add_argument("--count", type=int, default=8192)
    parser.add_argument("--physical-steps", type=int, default=500)
    parser.add_argument("--persistence", type=int, nargs="+",
                        default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--json-out", type=str, default="")
    args = parser.parse_args()
    results = []
    for bang_bang in (False, True):
        for persistence in args.persistence:
            results.append(random_reachability(
                tuple(args.target_factor), args.count, args.physical_steps,
                persistence, args.seed + persistence + 1000 * int(bang_bang),
                bang_bang,
            ))
    output = {
        "target_factor": args.target_factor,
        "trajectory_count_per_setting": args.count,
        "physical_steps": args.physical_steps,
        "results": results,
    }
    text = json.dumps(output, ensure_ascii=False, indent=2)
    print(text)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")


if __name__ == "__main__":
    main()
