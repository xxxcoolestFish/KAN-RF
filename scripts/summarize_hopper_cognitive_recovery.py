"""Summarize fixed-protocol Hopper cognitive recovery curves."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_return(path, mode):
    payload = json.loads(path.read_text(encoding="utf-8"))
    return float(payload["methods"][mode]["mean_return"])


def main(args):
    results = Path(args.results_dir)
    seeds = tuple(int(value) for value in args.seeds.split(","))
    budgets = tuple(int(value) for value in args.budgets.split(","))
    thresholds = tuple(float(value) for value in args.thresholds.split(","))
    curves = {}
    for seed in seeds:
        baseline_path = results / (
            f"hopper_distilled_policy_recovery_n512_"
            f"{args.target}_seed{seed}.json"
        )
        curve = [load_return(baseline_path, "source")]
        for budget in budgets[1:]:
            path = results / (
                f"hopper_distilled_policy_recovery_n{budget}_"
                f"{args.target}_seed{seed}.json"
            )
            curve.append(load_return(path, "ungated"))
        curves[str(seed)] = curve
    matrix = np.asarray(list(curves.values()), dtype=np.float64)
    budget_array = np.asarray(budgets, dtype=np.float64)
    raw_auc = np.trapezoid(matrix, budget_array, axis=1) / budget_array[-1]
    normalized_auc = []
    for curve in matrix:
        denominator = max(curve[-1] - curve[0], 1e-8)
        normalized = (curve - curve[0]) / denominator
        normalized_auc.append(
            np.trapezoid(normalized, budget_array) / budget_array[-1],
        )
    normalized_auc = np.asarray(normalized_auc)
    first_budget = {}
    for threshold in thresholds:
        values = []
        for curve in matrix:
            indices = np.flatnonzero(curve >= threshold)
            values.append(
                int(budgets[int(indices[0])]) if indices.size else None
            )
        observed = [value for value in values if value is not None]
        first_budget[str(threshold)] = {
            "by_seed": dict(zip((str(seed) for seed in seeds), values)),
            "reached_fraction": len(observed) / len(values),
            "mean_budget_when_reached": (
                float(np.mean(observed)) if observed else None
            ),
        }
    output = {
        "experiment": "HopperCognitiveRecoveryThreeSeedSummary",
        "target": args.target,
        "seeds": list(seeds),
        "budgets": list(budgets),
        "returns_by_seed": curves,
        "aggregate": {
            str(budget): {
                "mean_return": float(matrix[:, index].mean()),
                "std_return": float(matrix[:, index].std()),
            }
            for index, budget in enumerate(budgets)
        },
        "raw_return_auc": {
            "by_seed": dict(zip(
                (str(seed) for seed in seeds),
                (float(value) for value in raw_auc),
            )),
            "mean": float(raw_auc.mean()),
            "std": float(raw_auc.std()),
        },
        "normalized_recovery_auc": {
            "by_seed": dict(zip(
                (str(seed) for seed in seeds),
                (float(value) for value in normalized_auc),
            )),
            "mean": float(normalized_auc.mean()),
            "std": float(normalized_auc.std()),
        },
        "first_budget_at_return": first_budget,
        "monotonic_by_seed": {
            str(seed): bool(np.all(np.diff(matrix[index]) >= 0.0))
            for index, seed in enumerate(seeds)
        },
        "all_curves_monotonic": bool(np.all(np.diff(matrix, axis=1) >= 0.0)),
        "protocol": {
            "target_reward_used_for_policy_update": False,
            "target_physical_parameters_visible": False,
            "source_actor_frozen": True,
            "evaluation_episodes_per_point": 3,
        },
    }
    Path(args.json_out).write_text(
        json.dumps(output, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, indent=2), flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--target", default="combo_medium")
    parser.add_argument("--seeds", default="1811,1813,1817")
    parser.add_argument("--budgets", default="0,256,512,1024,2048")
    parser.add_argument("--thresholds", default="360,380,400")
    parser.add_argument(
        "--json-out",
        default="results/hopper_cognitive_recovery_three_seed_summary.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
