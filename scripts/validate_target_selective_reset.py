"""Locate source-policy negative transfer with selective parameter resets."""

from __future__ import annotations

import argparse
import copy
import json
from argparse import Namespace
from pathlib import Path

import torch

from cpbn import OracleAcrobotDynamics
from cpbn.corridor_policy import DirectCorridorActor
from cpbn.time_varying_tube import plan_continuous_cem_route
from scripts import validate_direct_corridor_actor as fixed
from scripts import validate_target_online_adaptation as online


RESET_BLOCKS = {
    "head": (
        "head.0.weight",
        "head.0.bias",
        "head.2.weight",
        "head.2.bias",
    ),
    "recurrent": (
        "encoder.gru.weight_hh_l0",
        "encoder.gru.bias_hh_l0",
    ),
}


def selective_reset_actor(
    source_actor: DirectCorridorActor,
    mode: str,
    seed: int,
):
    """Return a source Actor with only the requested block reinitialized."""
    if mode not in {"head", "recurrent", "recurrent_and_head"}:
        raise ValueError(f"unknown reset mode: {mode}")
    hidden_dim = source_actor.head[0].in_features
    log_std = float(source_actor.log_std.detach().item())
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        fresh_actor = DirectCorridorActor(hidden_dim, log_std)
    actor = copy.deepcopy(source_actor)
    reset_names = []
    if mode in {"head", "recurrent_and_head"}:
        reset_names.extend(RESET_BLOCKS["head"])
    if mode in {"recurrent", "recurrent_and_head"}:
        reset_names.extend(RESET_BLOCKS["recurrent"])
    target = dict(actor.named_parameters())
    fresh = dict(fresh_actor.named_parameters())
    with torch.no_grad():
        for name in reset_names:
            target[name].copy_(fresh[name])
    return actor, tuple(reset_names)


def _override_training_config(config: Namespace, cli: Namespace):
    for name in (
        "iterations",
        "num_envs",
        "rollout_horizon",
        "ppo_epochs",
        "minibatch",
        "eval_every",
        "eval_count",
        "num_test_seeds",
        "final_count",
        "evaluation_steps",
        "reference_population",
        "reference_elite",
        "reference_iterations",
    ):
        value = getattr(cli, name)
        if value is not None:
            setattr(config, name, value)
    config.seed = cli.seed
    if cli.source_checkpoint is not None:
        config.checkpoint = cli.source_checkpoint
    return config


def _load_baseline(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        baseline = json.load(handle)
    if baseline.get("experiment") != "HeavyInertiaTargetFixedRouteTrainingValidation":
        raise ValueError("baseline JSON is not a fixed-route target experiment")
    return baseline


def run(args):
    baseline = _load_baseline(args.baseline_json)
    training = _override_training_config(
        Namespace(**baseline["config"]), args,
    )
    torch.manual_seed(training.seed)
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

    initialized = {}
    reset_metadata = {}
    for index, mode in enumerate(args.modes):
        actor, reset_names = selective_reset_actor(
            source_actor, mode, training.seed + 701 + index,
        )
        initialized[mode] = actor
        named = dict(actor.named_parameters())
        reset_metadata[mode] = {
            "parameter_names": list(reset_names),
            "parameter_count": sum(named[name].numel() for name in reset_names),
            "initial_displacement": online.parameter_displacement(
                source_actor, actor,
            ),
            "initial_evaluation": online.aggregate_actor(
                actor, dynamics, reference, model_config, training,
                training.final_count, 600000 + index * 100000,
            ),
        }

    original_collector = online.collect_feedback_rollout
    online.collect_feedback_rollout = fixed.collect_rollout
    trained = {}
    conditions = {}
    try:
        for index, mode in enumerate(args.modes):
            actor, result = online.train_condition(
                f"reset_{mode}_target_route_fixed",
                initialized[mode], dynamics, reference, model_config, training,
                training.adapt_actor_lr, 700000 + index * 100000,
            )
            trained[mode] = actor
            conditions[mode] = result
            reset_metadata[mode]["trained_displacement_from_source"] = (
                online.parameter_displacement(source_actor, actor)
            )
            reset_metadata[mode]["training_displacement_from_reset"] = (
                online.parameter_displacement(initialized[mode], actor)
            )
    finally:
        online.collect_feedback_rollout = original_collector

    baseline_summary = baseline["result"]["summary"]
    rates = {
        mode: conditions[mode]["final"]["success_rate"]
        for mode in args.modes
    }
    source_rate = baseline_summary["source_initialized_success_rate"]
    scratch_rate = baseline_summary["scratch_success_rate"]
    best_mode = max(rates, key=rates.get)
    result = {
        "target_factor": online.HEAVY_INERTIA_FACTOR,
        "target_reference": {
            "state_count": int(reference.shape[0]),
            "maximum_height": construction.diagnostics.maximum_height,
            "success_step": construction.diagnostics.success_step,
            "actions_exposed_to_actor": False,
        },
        "training_phase_mode": "fixed_clock",
        "evaluation_phase_mode": "bounded_nearest",
        "baseline": {
            "json": args.baseline_json,
            "full_source_finetune_success_rate": source_rate,
            "scratch_success_rate": scratch_rate,
        },
        "reset_metadata": reset_metadata,
        "conditions": conditions,
        "summary": {
            "success_rates": rates,
            "best_reset_mode": best_mode,
            "best_reset_success_rate": rates[best_mode],
            "gain_over_full_source_finetune": rates[best_mode] - source_rate,
            "remaining_gap_to_scratch": scratch_rate - rates[best_mode],
        },
        "training_config": vars(training),
    }
    if args.checkpoint_out:
        torch.save({
            "actors": {
                mode: actor.state_dict() for mode, actor in trained.items()
            },
            "target_reference": reference,
            "reset_metadata": reset_metadata,
            "training_config": vars(training),
        }, args.checkpoint_out)
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline-json",
        default="results/target_fixed_route_training_equal_lr_seed0.json",
    )
    parser.add_argument("--source-checkpoint")
    parser.add_argument(
        "--modes", nargs="+", default=["head", "recurrent"],
        choices=["head", "recurrent", "recurrent_and_head"],
    )
    parser.add_argument("--seed", type=int, default=0)
    for name in (
        "iterations", "num_envs", "rollout_horizon", "ppo_epochs",
        "minibatch", "eval_every", "eval_count", "num_test_seeds",
        "final_count", "evaluation_steps", "reference_population",
        "reference_elite", "reference_iterations",
    ):
        parser.add_argument("--" + name.replace("_", "-"), type=int)
    parser.add_argument(
        "--checkpoint-out", default="results/target_selective_reset_seed0.pt",
    )
    parser.add_argument(
        "--json-out", default="results/target_selective_reset_seed0.json",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output = {
        "experiment": "HeavyInertiaTargetSelectiveResetValidation",
        "config": vars(args),
        "result": run(args),
    }
    text = json.dumps(output, ensure_ascii=False, indent=2)
    Path(args.json_out).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
