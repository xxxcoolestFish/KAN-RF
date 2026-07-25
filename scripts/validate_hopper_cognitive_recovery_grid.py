"""Fast fixed-protocol recovery grid for one hidden Hopper physics shift."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.diagnose_hopper_pullback_effect import (
    fit_distilled_source_counterfactual_context,
    load_source_twin,
)
from scripts.prescreen_hopper_physics_shifts import ENVS, SHIFTS
from scripts.validate_hopper_joint_online_adaptation import (
    FrozenSourcePolicy,
    load_cognition,
)
from scripts.validate_hopper_support_gated_policy import evaluate_policy


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    budgets = tuple(
        int(value) for value in args.budgets.split(",")
    )
    if not budgets or budgets[0] != 0:
        raise ValueError("budgets must start with zero")
    if any(
        following <= previous
        for previous, following in zip(budgets, budgets[1:])
    ):
        raise ValueError("budgets must be strictly increasing")

    source_policy = FrozenSourcePolicy(
        args.source_model, args.source_norm, device, args.seed, env=args.env,
    )
    source_twin = load_source_twin(args.source_twin_checkpoint, device)
    basis, source_context, _, delta_scale = load_cognition(args, device)
    results = {
        "0": evaluate_policy(
            "source",
            source_policy,
            None,
            basis,
            source_context,
            None,
            delta_scale,
            args,
            device,
        ),
    }
    print({
        "target": args.target,
        "budget": 0,
        "mean_return": results["0"]["mean_return"],
    }, flush=True)
    original_warmup = args.cognition_warmup
    drift_trust_radius = getattr(args, "drift_trust_radius", None)
    drift_trust_calibrated = drift_trust_radius is not None
    drift_trust_log = {}
    for budget in budgets[1:]:
        args.cognition_warmup = budget
        target_context, _ = fit_distilled_source_counterfactual_context(
            source_policy,
            basis,
            source_context,
            args,
            device,
            source_twin,
            drift_trust_radius=(
                drift_trust_radius if drift_trust_calibrated else None
            ),
        )
        if not drift_trust_calibrated:
            drift_trust_radius = getattr(
                target_context, "drift_unconstrained_norm", None,
            )
            drift_trust_log["calibration_budget"] = budget
            drift_trust_log["calibrated_radius"] = drift_trust_radius
            drift_trust_calibrated = True
        results[str(budget)] = evaluate_policy(
            "ungated",
            source_policy,
            None,
            basis,
            source_context,
            target_context,
            delta_scale,
            args,
            device,
        )
        drift_trust_log[str(budget)] = {
            "drift_norm": getattr(
                target_context, "paired_source_drift_delta_norm", None,
            ),
            "trust_active": getattr(
                target_context, "drift_trust_active", None,
            ),
            "lagrange_multiplier": getattr(
                target_context, "drift_lagrange_multiplier", None,
            ),
            "unconstrained_norm": getattr(
                target_context, "drift_unconstrained_norm", None,
            ),
        }
        print({
            "target": args.target,
            "budget": budget,
            "mean_return": results[str(budget)]["mean_return"],
        }, flush=True)
    args.cognition_warmup = original_warmup
    baseline = results["0"]["mean_return"]
    final = results[str(budgets[-1])]["mean_return"]
    output = {
        "experiment": "HopperCognitiveRecoveryPhysicsGrid",
        "env": args.env,
        "target": args.target,
        "hidden_shift_not_visible_to_learner": True,
        "source_actor_frozen": True,
        "target_reward_used_for_policy_update": False,
        "source_simulator_queried_during_target_evaluation": False,
        "budgets": list(budgets),
        "results": results,
        "improvement_at_final_budget": float(final - baseline),
        "relative_final_return": float(final / max(baseline, 1e-8)),
        "monotonic_recovery": bool(all(
            results[str(following)]["mean_return"]
            >= results[str(previous)]["mean_return"]
            for previous, following in zip(budgets, budgets[1:])
        )),
        "drift_trust_region": drift_trust_log,
        "config": vars(args),
    }
    Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json_out).write_text(
        json.dumps(output, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, indent=2), flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1811)
    parser.add_argument(
        "--env", choices=tuple(ENVS), default="hopper",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--target",
        choices=tuple(name for name in SHIFTS if name != "source"),
        required=True,
    )
    parser.add_argument("--budgets", default="0,512,2048")
    parser.add_argument("--evaluation-episodes", type=int, default=3)
    parser.add_argument("--cognition-warmup", type=int, default=0)
    parser.add_argument("--warmup-noise", type=float, default=0.3)
    parser.add_argument("--transform-ridge", type=float, default=10.0)
    parser.add_argument("--diagonal-transform", action="store_true")
    parser.add_argument("--drift-ridge", type=float, default=100.0)
    parser.add_argument("--drift-trust-radius", type=float, default=None)
    parser.add_argument("--drift-spectral-eta", type=float, default=0.0)
    parser.add_argument("--drift-spectral-beta", type=float, default=1.0)
    parser.add_argument("--drift-smooth-lambda", type=float, default=0.0)
    parser.add_argument(
        "--drift-spectral-mode", choices=("max", "mean"), default="max",
    )
    parser.add_argument("--pullback-damping", type=float, default=0.05)
    parser.add_argument(
        "--source-model",
        default="results/hopper_source_sb3_ppo_continued_seed1811.zip",
    )
    parser.add_argument(
        "--source-norm",
        default="results/hopper_source_sb3_vecnorm_continued_seed1811.pkl",
    )
    parser.add_argument(
        "--source-twin-checkpoint",
        default="results/hopper_source_affine_twin_cloud_seed1811.pt",
    )
    parser.add_argument(
        "--cognition-checkpoint",
        default="results/hopper_source_control_sobolev_calibrated_seed1811.pt",
    )
    parser.add_argument(
        "--json-out",
        default="results/hopper_cognitive_recovery_grid.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
