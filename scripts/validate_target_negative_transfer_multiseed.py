"""Matched-randomness multi-seed validation of target negative transfer."""

from __future__ import annotations

import argparse
import copy
import json
import statistics
from argparse import Namespace
from pathlib import Path

import torch

from cpbn import OracleAcrobotDynamics
from cpbn.corridor_policy import DirectCorridorActor
from cpbn.time_varying_tube import plan_continuous_cem_route
from scripts import validate_direct_corridor_actor as fixed
from scripts import validate_target_online_adaptation as online
from scripts.validate_target_selective_reset import selective_reset_actor


CONDITIONS = ("source", "head_reset", "scratch")


def summarize_seed_results(per_seed):
    """Aggregate condition success while retaining training-seed variation."""
    summary = {}
    for condition in CONDITIONS:
        records = [item["conditions"][condition] for item in per_seed]
        rates = [record["final"]["success_rate"] for record in records]
        successes = [record["final"]["success_count"] for record in records]
        counts = [record["final"]["evaluation_count"] for record in records]
        summary[condition] = {
            "training_seed_success_rates": rates,
            "mean_success_rate": statistics.mean(rates),
            "population_std_success_rate": statistics.pstdev(rates),
            "pooled_success_count": sum(successes),
            "pooled_evaluation_count": sum(counts),
            "pooled_success_rate": sum(successes) / sum(counts),
        }
    summary["head_reset_minus_source"] = {
        "per_training_seed": [
            item["conditions"]["head_reset"]["final"]["success_rate"]
            - item["conditions"]["source"]["final"]["success_rate"]
            for item in per_seed
        ],
    }
    deltas = summary["head_reset_minus_source"]["per_training_seed"]
    summary["head_reset_minus_source"]["mean"] = statistics.mean(deltas)
    return summary


def _training_config(args):
    with open(args.baseline_json, "r", encoding="utf-8") as handle:
        baseline = json.load(handle)
    config = Namespace(**baseline["config"])
    if args.source_checkpoint is not None:
        config.checkpoint = args.source_checkpoint
    for name in (
        "iterations", "num_envs", "rollout_horizon", "ppo_epochs",
        "minibatch", "eval_every", "eval_count", "num_test_seeds",
        "final_count", "evaluation_steps", "reference_population",
        "reference_elite", "reference_iterations",
    ):
        value = getattr(args, name)
        if value is not None:
            setattr(config, name, value)
    return config


def _fresh_actor(model_config, seed):
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        return DirectCorridorActor(
            model_config.hidden_dim, model_config.log_std_init,
        )


def run(args):
    training = _training_config(args)
    checkpoint = torch.load(
        training.checkpoint, map_location="cpu", weights_only=False,
    )
    model_config = Namespace(**checkpoint["config"])
    source_actor = DirectCorridorActor(
        model_config.hidden_dim, model_config.log_std_init,
    )
    source_actor.load_state_dict(checkpoint["actor"])

    dynamics = OracleAcrobotDynamics(online.HEAVY_INERTIA_FACTOR)
    construction = plan_continuous_cem_route(
        dynamics,
        segment_count=training.reference_segments,
        segment_steps=training.segment_steps,
        population=training.reference_population,
        elite_count=training.reference_elite,
        iterations=training.reference_iterations,
        seed=training.reference_seed,
    )
    reference = construction.states.detach().clone()

    original_collector = online.collect_feedback_rollout
    online.collect_feedback_rollout = fixed.collect_rollout
    per_seed = []
    try:
        for training_seed in args.training_seeds:
            training.seed = training_seed
            head_actor, reset_names = selective_reset_actor(
                source_actor, "head", 320000 + training_seed,
            )
            initial = {
                "source": copy.deepcopy(source_actor),
                "head_reset": head_actor,
                "scratch": _fresh_actor(
                    model_config, 310000 + training_seed,
                ),
            }
            seed_record = {
                "training_seed": training_seed,
                "matched_rollout_seed_offset": args.seed_offset,
                "matched_torch_seed": 330000 + training_seed,
                "head_reset_parameter_names": list(reset_names),
                "conditions": {},
            }
            for condition in CONDITIONS:
                # Critic initialization, stochastic actions, rollout resets,
                # minibatch ordering, and final evaluation states are matched.
                torch.manual_seed(330000 + training_seed)
                _, result = online.train_condition(
                    f"{condition}_training_seed_{training_seed}",
                    initial[condition], dynamics, reference, model_config,
                    training, training.adapt_actor_lr, args.seed_offset,
                )
                seed_record["conditions"][condition] = result
            per_seed.append(seed_record)
    finally:
        online.collect_feedback_rollout = original_collector

    return {
        "target_factor": online.HEAVY_INERTIA_FACTOR,
        "target_reference": {
            "state_count": int(reference.shape[0]),
            "maximum_height": construction.diagnostics.maximum_height,
            "success_step": construction.diagnostics.success_step,
            "actions_exposed_to_actor": False,
        },
        "training_phase_mode": "fixed_clock",
        "evaluation_phase_mode": "bounded_nearest",
        "randomness_control": {
            "matched_within_training_seed": True,
            "matched_items": [
                "critic initialization",
                "policy sampling stream",
                "rollout initial-state stream",
                "PPO minibatch ordering",
                "evaluation initial states",
            ],
            "intentional_difference": "actor initialization only",
        },
        "per_training_seed": per_seed,
        "summary": summarize_seed_results(per_seed),
        "training_config": vars(training),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline-json",
        default="results/target_fixed_route_training_equal_lr_seed0.json",
    )
    parser.add_argument("--source-checkpoint")
    parser.add_argument(
        "--training-seeds", nargs="+", type=int, default=[0, 1, 2],
    )
    parser.add_argument("--seed-offset", type=int, default=900000)
    for name in (
        "iterations", "num_envs", "rollout_horizon", "ppo_epochs",
        "minibatch", "eval_every", "eval_count", "num_test_seeds",
        "final_count", "evaluation_steps", "reference_population",
        "reference_elite", "reference_iterations",
    ):
        parser.add_argument("--" + name.replace("_", "-"), type=int)
    parser.add_argument(
        "--json-out",
        default="results/target_negative_transfer_multiseed.json",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output = {
        "experiment": "HeavyInertiaTargetNegativeTransferMultiSeed",
        "config": vars(args),
        "result": run(args),
    }
    text = json.dumps(output, ensure_ascii=False, indent=2)
    Path(args.json_out).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
