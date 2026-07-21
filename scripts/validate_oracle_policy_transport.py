"""Validate implicit policy-parameter transport under Oracle physics shifts."""

from __future__ import annotations

import argparse
import json
from argparse import Namespace
from dataclasses import asdict

import torch

from cpbn import OracleAcrobotDynamics
from cpbn.corridor_policy import DirectCorridorActor, future_corridor
from cpbn.policy_transport import (
    apply_parameter_delta,
    closed_loop_corridor_objective,
    implicit_transport_delta,
)
from scripts.validate_direct_corridor_actor import perturbed_reference
from scripts.validate_feedback_phase_actor import aggregate, evaluate_mode


TARGET_FACTORS = {
    "weak_actuator": (7.35, 0.0, 0.50, 0.80),
    "heavy_inertia": (7.35, 0.0, 0.80, 1.20),
    "strong_gravity": (9.00, 0.0, 0.80, 0.80),
    "combined_shift": (9.00, 0.05, 0.55, 1.10),
}


def calibration_batch(reference, count, rollout_steps, noise, seed):
    generator = torch.Generator().manual_seed(seed)
    maximum_phase = reference.shape[0] - rollout_steps - 1
    phase = torch.randint(maximum_phase, (count,), generator=generator)
    state = perturbed_reference(reference, phase, generator, noise)
    return state, phase


@torch.no_grad()
def model_objective(
    actor, dynamics, reference, state, phase, args,
):
    return float(closed_loop_corridor_objective(
        actor, dynamics, reference, state, phase,
        args.rollout_steps, args.corridor_horizon,
        args.action_penalty,
    ))


def select_transport_scale(
    actor, delta, model, reference, state, phase, args,
):
    candidates = []
    for scale in args.line_search_scales:
        candidate = apply_parameter_delta(actor, delta, scale)
        objective = model_objective(
            candidate, model, reference, state, phase, args,
        )
        candidates.append({"scale": scale, "objective": objective})
    best = min(candidates, key=lambda item: item["objective"])
    return apply_parameter_delta(actor, delta, best["scale"]), best, candidates


@torch.no_grad()
def mean_action_change(actor, transported, reference, state, phase, horizon):
    corridor = future_corridor(reference, phase, horizon)
    before = torch.tanh(actor.distribution(state, corridor).mean)
    after = torch.tanh(transported.distribution(state, corridor).mean)
    return float((before - after).abs().mean())


def evaluate_actor(actor, dynamics, reference, config, args):
    cli = Namespace(
        count=args.evaluation_count,
        steps=args.evaluation_steps,
        nearest_backtrack=args.nearest_backtrack,
        nearest_advance=args.nearest_advance,
        observation_scale=0.10,
    )
    records = []
    for index in range(args.num_seeds):
        record = evaluate_mode(
            actor, dynamics, reference, config, cli, "nearest",
            args.test_seed + index * 1009,
        )
        record["evaluation_count"] = args.evaluation_count
        records.append(record)
    return aggregate(records)


def run(args):
    torch.manual_seed(args.seed)
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False,
    )
    config = Namespace(**checkpoint["config"])
    reference = checkpoint["reference"].detach().clone()
    actor = DirectCorridorActor(config.hidden_dim, config.log_std_init)
    actor.load_state_dict(checkpoint["actor"])
    actor.eval()
    source_model = OracleAcrobotDynamics()
    zero_delta = {
        name: torch.zeros_like(parameter)
        for name, parameter in actor.named_parameters()
        if name != "log_std"
    }
    inherited_actor = apply_parameter_delta(actor, zero_delta)
    source_base = evaluate_actor(
        actor, source_model, reference, config, args,
    )
    source_inherited = evaluate_actor(
        inherited_actor, source_model, reference, config, args,
    )

    train_state, train_phase = calibration_batch(
        reference, args.calibration_count, args.rollout_steps,
        args.calibration_noise, args.seed + 100,
    )
    validation_state, validation_phase = calibration_batch(
        reference, args.validation_count, args.rollout_steps,
        args.calibration_noise, args.seed + 200,
    )
    target_names = list(TARGET_FACTORS)
    targets = {}
    for index, name in enumerate(target_names):
        target_model = OracleAcrobotDynamics(TARGET_FACTORS[name])
        wrong_name = target_names[(index + 1) % len(target_names)]
        wrong_model = OracleAcrobotDynamics(TARGET_FACTORS[wrong_name])

        correct_delta, correct_diagnostics = implicit_transport_delta(
            actor, source_model, target_model, reference,
            train_state, train_phase,
            args.rollout_steps, args.corridor_horizon,
            args.fisher_draws, args.damping, args.trust_radius,
            args.seed + 1000 + index,
        )
        correct_actor, correct_best, correct_line = select_transport_scale(
            actor, correct_delta, target_model, reference,
            validation_state, validation_phase, args,
        )
        wrong_delta, wrong_diagnostics = implicit_transport_delta(
            actor, source_model, wrong_model, reference,
            train_state, train_phase,
            args.rollout_steps, args.corridor_horizon,
            args.fisher_draws, args.damping, args.trust_radius,
            args.seed + 2000 + index,
        )
        wrong_actor, wrong_best, wrong_line = select_transport_scale(
            actor, wrong_delta, wrong_model, reference,
            validation_state, validation_phase, args,
        )
        no_transport = evaluate_actor(
            actor, target_model, reference, config, args,
        )
        correct_transport = evaluate_actor(
            correct_actor, target_model, reference, config, args,
        )
        wrong_transport = evaluate_actor(
            wrong_actor, target_model, reference, config, args,
        )
        targets[name] = {
            "factor": TARGET_FACTORS[name],
            "wrong_cognition_factor": wrong_name,
            "no_transport": no_transport,
            "correct_transport": correct_transport,
            "wrong_transport": wrong_transport,
            "correct_transport_diagnostics": asdict(correct_diagnostics),
            "wrong_transport_diagnostics": asdict(wrong_diagnostics),
            "correct_line_search": correct_line,
            "wrong_line_search": wrong_line,
            "selected_correct_scale": correct_best["scale"],
            "selected_wrong_scale": wrong_best["scale"],
            "correct_mean_action_change": mean_action_change(
                actor, correct_actor, reference,
                validation_state, validation_phase, args.corridor_horizon,
            ),
            "wrong_mean_action_change": mean_action_change(
                actor, wrong_actor, reference,
                validation_state, validation_phase, args.corridor_horizon,
            ),
        }
        print(json.dumps({
            "target": name,
            "no_transport": no_transport["success_rate"],
            "correct_transport": correct_transport["success_rate"],
            "wrong_transport": wrong_transport["success_rate"],
            "correct_scale": correct_best["scale"],
        }), flush=True)

    improved = sum(
        item["correct_transport"]["success_rate"]
        > item["no_transport"]["success_rate"]
        for item in targets.values()
    )
    correct_beats_wrong = sum(
        item["correct_transport"]["success_rate"]
        > item["wrong_transport"]["success_rate"]
        for item in targets.values()
    )
    return {
        "source_inheritance": {
            "base": source_base,
            "zero_transport": source_inherited,
            "success_counts_equal": (
                source_base["success_count"]
                == source_inherited["success_count"]
            ),
        },
        "targets": targets,
        "summary": {
            "target_count": len(targets),
            "correct_transport_improved_targets": improved,
            "correct_transport_beat_wrong_targets": correct_beats_wrong,
            "source_inheritance_passed": (
                source_base["success_count"]
                == source_inherited["success_count"]
                and source_base["success_rate"] >= 0.95
            ),
            "oracle_transport_passed": (
                improved >= 3 and correct_beats_wrong >= 3
            ),
        },
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", default="results/direct_corridor_actor_strong_seed0.pt",
    )
    parser.add_argument("--corridor-horizon", type=int, default=12)
    parser.add_argument("--rollout-steps", type=int, default=6)
    parser.add_argument("--action-penalty", type=float, default=0.002)
    parser.add_argument("--calibration-count", type=int, default=128)
    parser.add_argument("--validation-count", type=int, default=128)
    parser.add_argument("--calibration-noise", type=float, default=0.025)
    parser.add_argument("--fisher-draws", type=int, default=4)
    parser.add_argument("--damping", type=float, default=0.03)
    parser.add_argument("--trust-radius", type=float, default=0.20)
    parser.add_argument(
        "--line-search-scales", type=float, nargs="+",
        default=[0.0, 0.25, 0.5, 1.0],
    )
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--evaluation-count", type=int, default=32)
    parser.add_argument("--evaluation-steps", type=int, default=500)
    parser.add_argument("--nearest-backtrack", type=int, default=4)
    parser.add_argument("--nearest-advance", type=int, default=12)
    parser.add_argument("--test-seed", type=int, default=20261101)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--json-out", default="results/oracle_policy_transport_seed0.json",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output = {
        "experiment": "OracleImplicitPolicyTransportValidation",
        "config": vars(args),
        "result": run(args),
    }
    text = json.dumps(output, ensure_ascii=False, indent=2)
    with open(args.json_out, "w", encoding="utf-8") as handle:
        handle.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
