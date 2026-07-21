"""Train target-route Actors with the proven fixed-clock PPO protocol."""

from __future__ import annotations

import sys
import json
from argparse import Namespace

import torch

from cpbn import OracleAcrobotDynamics
from cpbn.corridor_policy import DirectCorridorActor
from cpbn.time_varying_tube import plan_continuous_cem_route
from scripts import validate_direct_corridor_actor as fixed
from scripts import validate_target_online_adaptation as online


def run(args):
    torch.manual_seed(args.seed)
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False,
    )
    config = Namespace(**checkpoint["config"])
    source_actor = DirectCorridorActor(config.hidden_dim, config.log_std_init)
    source_actor.load_state_dict(checkpoint["actor"])
    scratch_actor = DirectCorridorActor(config.hidden_dim, config.log_std_init)
    target_dynamics = OracleAcrobotDynamics(online.HEAVY_INERTIA_FACTOR)
    construction = plan_continuous_cem_route(
        target_dynamics,
        segment_count=args.reference_segments,
        segment_steps=args.segment_steps,
        population=args.reference_population,
        elite_count=args.reference_elite,
        iterations=args.reference_iterations,
        seed=args.reference_seed,
    )
    reference = construction.states.detach().clone()
    original_collector = online.collect_feedback_rollout
    online.collect_feedback_rollout = fixed.collect_rollout
    try:
        source_adapted, source_result = online.train_condition(
            "source_actor_target_route_fixed", source_actor,
            target_dynamics, reference, config, args,
            args.adapt_actor_lr, 400000,
        )
        scratch_adapted, scratch_result = online.train_condition(
            "scratch_actor_target_route_fixed", scratch_actor,
            target_dynamics, reference, config, args,
            args.scratch_actor_lr, 500000,
        )
    finally:
        online.collect_feedback_rollout = original_collector
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
        "conditions": {
            "source_actor_target_route_fixed": source_result,
            "scratch_actor_target_route_fixed": scratch_result,
        },
        "source_initialized_parameter_displacement": online.parameter_displacement(
            source_actor, source_adapted,
        ),
    }
    result["summary"] = {
        "source_initialized_success_rate": source_result["final"][
            "success_rate"
        ],
        "scratch_success_rate": scratch_result["final"]["success_rate"],
        "target_decision_learning_is_solved": max(
            source_result["final"]["success_rate"],
            scratch_result["final"]["success_rate"],
        ) >= 0.80,
    }
    if args.checkpoint_out:
        torch.save({
            "source_initialized_actor": source_adapted.state_dict(),
            "scratch_actor": scratch_adapted.state_dict(),
            "target_reference": reference,
            "config": vars(args),
        }, args.checkpoint_out)
    return result


def parse_args():
    args = online.parse_args()
    defaults = {
        "reference_segments": 30,
        "reference_population": 4096,
        "reference_elite": 256,
        "reference_iterations": 18,
        "reference_seed": 0,
        "evaluation_steps": 750,
        "iterations": 60,
        "num_envs": 64,
        "rollout_horizon": 96,
        "ppo_epochs": 2,
        "minibatch": 512,
        "eval_every": 10,
        "eval_count": 16,
        "num_test_seeds": 3,
        "final_count": 32,
        "checkpoint_out": "results/target_fixed_route_training_seed0.pt",
        "json_out": "results/target_fixed_route_training_seed0.json",
    }
    command_line = set(sys.argv[1:])
    for name, value in defaults.items():
        option = "--" + name.replace("_", "-")
        if option not in command_line:
            setattr(args, name, value)
    return args


def main():
    args = parse_args()
    output = {
        "experiment": "HeavyInertiaTargetFixedRouteTrainingValidation",
        "config": vars(args),
        "result": run(args),
    }
    text = json.dumps(output, ensure_ascii=False, indent=2)
    with open(args.json_out, "w", encoding="utf-8") as handle:
        handle.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
