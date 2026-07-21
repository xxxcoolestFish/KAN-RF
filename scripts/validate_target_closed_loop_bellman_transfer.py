"""Evaluate a frozen closed-loop Bellman Actor on one fixed target corridor."""

from __future__ import annotations

import argparse
import json
from argparse import Namespace
from pathlib import Path

import torch

from cpbn import OracleAcrobotDynamics, tip_height
from cpbn.closed_loop_bellman import ClosedLoopBellmanActor
from cpbn.time_varying_tube import plan_continuous_cem_route
from scripts import validate_oracle_bellman_adjoint_actor as adjoint
from scripts import validate_oracle_policy_transport as transport
from scripts import validate_target_online_adaptation as online


WRONG_STRONG_GRAVITY_FACTOR = transport.TARGET_FACTORS["strong_gravity"]


def actor_from_config(config):
    return ClosedLoopBellmanActor(
        hidden_dim=config.hidden_dim,
        corridor_horizon=config.corridor_horizon,
        action_bins=config.action_bins,
        backup_depth=config.backup_depth,
        macro_steps=config.macro_steps,
        temperature=config.temperature,
        gamma=config.gamma,
        progress_weight=config.progress_reward,
        progress_clip=config.progress_clip,
        inside_reward=config.inside_reward,
        success_reward=config.success_reward,
        action_penalty=config.action_penalty,
        corridor_radius=config.corridor_radius,
    )


def load_target_reference(args, target_dynamics):
    checkpoint_path = Path(args.target_route_checkpoint)
    if checkpoint_path.exists():
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False,
        )
        if "target_reference" not in checkpoint:
            raise KeyError("target route checkpoint has no target_reference")
        return checkpoint["target_reference"].detach().clone(), {
            "origin": str(checkpoint_path),
            "recomputed": False,
        }
    construction = plan_continuous_cem_route(
        target_dynamics,
        segment_count=args.reference_segments,
        segment_steps=args.segment_steps,
        population=args.reference_population,
        elite_count=args.reference_elite,
        iterations=args.reference_iterations,
        seed=args.reference_seed,
    )
    return construction.states.detach().clone(), {
        "origin": "recomputed_oracle_cem",
        "recomputed": True,
    }


def actor_parameter_vector(actor):
    return torch.cat([
        parameter.detach().flatten() for parameter in actor.parameters()
    ])


def aggregate_condition(
    actor, internal_cognition, real_dynamics, reference, config, args,
):
    bound = adjoint.BoundCognitionActor(actor, internal_cognition)
    return online.aggregate_actor(
        bound, real_dynamics, reference, config, args,
        args.final_count, args.seed_offset,
    )


def run(args):
    torch.manual_seed(args.seed)
    actor_checkpoint = torch.load(
        args.actor_checkpoint, map_location="cpu", weights_only=False,
    )
    config = Namespace(**actor_checkpoint["config"])
    actor = actor_from_config(config)
    actor.load_state_dict(actor_checkpoint["actor"])
    actor.eval()
    parameters_before = actor_parameter_vector(actor)

    target_dynamics = OracleAcrobotDynamics(online.HEAVY_INERTIA_FACTOR)
    source_cognition = OracleAcrobotDynamics()
    correct_target_cognition = OracleAcrobotDynamics(
        online.HEAVY_INERTIA_FACTOR,
    )
    wrong_cognition = OracleAcrobotDynamics(WRONG_STRONG_GRAVITY_FACTOR)
    target_reference, route_origin = load_target_reference(
        args, target_dynamics,
    )

    conditions = {
        "correct_target_cognition": aggregate_condition(
            actor, correct_target_cognition, target_dynamics,
            target_reference, config, args,
        ),
        "frozen_source_cognition": aggregate_condition(
            actor, source_cognition, target_dynamics,
            target_reference, config, args,
        ),
        "wrong_strong_gravity_cognition": aggregate_condition(
            actor, wrong_cognition, target_dynamics,
            target_reference, config, args,
        ),
    }
    sensitivity_target_vs_source = adjoint.cognition_action_sensitivity(
        actor, correct_target_cognition, source_cognition,
        target_reference, args, 512, args.test_seed + 300000,
    )
    sensitivity_target_vs_wrong = adjoint.cognition_action_sensitivity(
        actor, correct_target_cognition, wrong_cognition,
        target_reference, args, 512, args.test_seed + 400000,
    )
    parameters_after = actor_parameter_vector(actor)
    if not torch.equal(parameters_before, parameters_after):
        raise RuntimeError("frozen Actor parameters changed during evaluation")

    correct_rate = conditions["correct_target_cognition"]["success_rate"]
    source_rate = conditions["frozen_source_cognition"]["success_rate"]
    wrong_rate = conditions[
        "wrong_strong_gravity_cognition"
    ]["success_rate"]
    height = tip_height(target_reference)
    success = torch.nonzero(height >= 1.0, as_tuple=False).flatten()
    return {
        "actor": {
            "checkpoint": args.actor_checkpoint,
            "best_iteration": actor_checkpoint.get("best_iteration", 35),
            "frozen": True,
            "parameter_count": int(parameters_before.numel()),
            "parameters_bitwise_unchanged": True,
        },
        "real_environment_factor": online.HEAVY_INERTIA_FACTOR,
        "target_reference": {
            **route_origin,
            "state_count": int(target_reference.shape[0]),
            "maximum_height": float(height.max()),
            "success_step": int(success[0]) if success.numel() else -1,
            "actions_exposed_to_actor": False,
            "shared_by_all_conditions": True,
        },
        "matched_protocol": {
            "same_real_environment": True,
            "same_target_reference": True,
            "same_actor_parameters": True,
            "same_initial_state_seeds": True,
            "only_internal_cognition_changes": True,
        },
        "conditions": conditions,
        "correct_minus_frozen_source_success_rate": (
            correct_rate - source_rate
        ),
        "correct_minus_wrong_success_rate": correct_rate - wrong_rate,
        "action_sensitivity": {
            "target_vs_source": sensitivity_target_vs_source,
            "target_vs_wrong": sensitivity_target_vs_wrong,
        },
        "correct_cognition_gate_passed": (
            correct_rate >= source_rate + args.minimum_correct_gain
            and correct_rate >= wrong_rate + args.minimum_correct_gain
        ),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--actor-checkpoint",
        default="results/oracle_closed_loop_bellman_actor_seed0.pt",
    )
    parser.add_argument(
        "--target-route-checkpoint",
        default="results/target_fixed_route_training_seed0.pt",
    )
    parser.add_argument("--reference-segments", type=int, default=30)
    parser.add_argument("--segment-steps", type=int, default=24)
    parser.add_argument("--reference-population", type=int, default=4096)
    parser.add_argument("--reference-elite", type=int, default=256)
    parser.add_argument("--reference-iterations", type=int, default=18)
    parser.add_argument("--reference-seed", type=int, default=0)
    parser.add_argument("--corridor-horizon", type=int, default=12)
    parser.add_argument("--initial-noise", type=float, default=0.025)
    parser.add_argument("--phase-backtrack", type=int, default=4)
    parser.add_argument("--phase-advance", type=int, default=12)
    parser.add_argument("--evaluation-steps", type=int, default=750)
    parser.add_argument("--num-test-seeds", type=int, default=3)
    parser.add_argument("--final-count", type=int, default=32)
    parser.add_argument("--minimum-correct-gain", type=float, default=0.02)
    parser.add_argument("--test-seed", type=int, default=20261701)
    parser.add_argument("--seed-offset", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--json-out",
        default="results/target_closed_loop_bellman_transfer_seed0.json",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output = {
        "experiment": "TargetClosedLoopBellmanCognitionTransfer",
        "config": vars(args),
        "result": run(args),
    }
    text = json.dumps(output, ensure_ascii=False, indent=2)
    Path(args.json_out).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
