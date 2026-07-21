"""Compare CPIT parameter transport with real target PPO adaptation."""

from __future__ import annotations

import argparse
import json
import math
from argparse import Namespace

import torch

from cpbn import OracleAcrobotDynamics
from cpbn.corridor_policy import DirectCorridorActor, future_corridor
from cpbn.policy_transport import (
    apply_parameter_delta,
    implicit_transport_delta,
)
from scripts.validate_direct_corridor_actor import perturbed_reference
from scripts.validate_oracle_policy_transport import calibration_batch


HEAVY_INERTIA_FACTOR = (7.35, 0.0, 0.80, 1.20)


def cosine(left, right):
    denominator = left.norm() * right.norm()
    if float(denominator) <= 1e-12:
        return 0.0
    return float(torch.dot(left, right) / denominator)


def flatten_blocks(blocks, names):
    return torch.cat([blocks[name].reshape(-1) for name in names])


@torch.no_grad()
def action_shift_alignment(
    source_actor, adapted_actor, transported_actor,
    reference, count, horizon, noise, seed,
):
    generator = torch.Generator().manual_seed(seed)
    phase = torch.randint(reference.shape[0] - horizon - 1, (count,), generator=generator)
    state = perturbed_reference(reference, phase, generator, noise)
    corridor = future_corridor(reference, phase, horizon)
    base = torch.tanh(source_actor.distribution(state, corridor).mean)
    actual = torch.tanh(adapted_actor.distribution(state, corridor).mean) - base
    predicted = torch.tanh(transported_actor.distribution(state, corridor).mean) - base
    return {
        "actual_mean_absolute_change": float(actual.abs().mean()),
        "predicted_mean_absolute_change": float(predicted.abs().mean()),
        "cosine": cosine(actual.reshape(-1), predicted.reshape(-1)),
        "sign_agreement": float(
            ((actual * predicted) > 0).float().mean()
        ),
    }


def run(args):
    source_checkpoint = torch.load(
        args.source_checkpoint, map_location="cpu", weights_only=False,
    )
    adapted_checkpoint = torch.load(
        args.adapted_checkpoint, map_location="cpu", weights_only=False,
    )
    config = Namespace(**source_checkpoint["config"])
    source_actor = DirectCorridorActor(config.hidden_dim, config.log_std_init)
    source_actor.load_state_dict(source_checkpoint["actor"])
    adapted_actor = DirectCorridorActor(config.hidden_dim, config.log_std_init)
    adapted_actor.load_state_dict(
        adapted_checkpoint["source_initialized_actor"],
    )
    successful_actor = DirectCorridorActor(config.hidden_dim, config.log_std_init)
    successful_actor.load_state_dict(
        adapted_checkpoint["scratch_actor"],
    )
    source_reference = source_checkpoint["reference"].detach().clone()
    target_reference = adapted_checkpoint["target_reference"].detach().clone()
    state, phase = calibration_batch(
        source_reference, args.calibration_count, args.rollout_steps,
        args.noise, args.seed + 100,
    )
    predicted_delta, transport_diagnostics = implicit_transport_delta(
        source_actor,
        OracleAcrobotDynamics(),
        OracleAcrobotDynamics(HEAVY_INERTIA_FACTOR),
        source_reference, state, phase,
        args.rollout_steps, args.corridor_horizon,
        args.fisher_draws, args.damping, args.trust_radius,
        args.seed + 1001,
    )
    source_parameters = dict(source_actor.named_parameters())
    adapted_parameters = dict(adapted_actor.named_parameters())
    names = [name for name in predicted_delta]
    actual_delta = {
        name: adapted_parameters[name].detach() - source_parameters[name].detach()
        for name in names
    }
    actual_vector = flatten_blocks(actual_delta, names)
    predicted_vector = flatten_blocks(predicted_delta, names)
    scaled_delta = {
        name: args.transport_scale * value
        for name, value in predicted_delta.items()
    }
    transported_actor = apply_parameter_delta(source_actor, scaled_delta)
    layer_alignment = {}
    for name in names:
        actual = actual_delta[name].reshape(-1)
        predicted = predicted_delta[name].reshape(-1)
        layer_alignment[name] = {
            "actual_norm": float(actual.norm()),
            "predicted_norm": float(predicted.norm()),
            "cosine": cosine(actual, predicted),
        }
    return {
        "target_factor": HEAVY_INERTIA_FACTOR,
        "actual_online_ppo_update": {
            "norm": float(actual_vector.norm()),
            "relative_to_source_parameter_norm": float(
                actual_vector.norm() / math.sqrt(sum(
                    float(source_parameters[name].detach().square().sum())
                    for name in names
                ))
            ),
        },
        "cpit_update": {
            "unscaled_norm": float(predicted_vector.norm()),
            "evaluation_scale": args.transport_scale,
            "scaled_norm": float(args.transport_scale * predicted_vector.norm()),
            "diagnostics": transport_diagnostics.__dict__,
        },
        "parameter_space_alignment": {
            "global_cosine": cosine(actual_vector, predicted_vector),
            "layer_alignment": layer_alignment,
        },
        "source_route_action_alignment": action_shift_alignment(
            source_actor, adapted_actor, transported_actor,
            source_reference, args.action_count, args.corridor_horizon,
            args.noise, args.seed + 200,
        ),
        "target_route_action_alignment": action_shift_alignment(
            source_actor, adapted_actor, transported_actor,
            target_reference, args.action_count, args.corridor_horizon,
            args.noise, args.seed + 300,
        ),
        "successful_target_policy_action_alignment": action_shift_alignment(
            source_actor, successful_actor, transported_actor,
            target_reference, args.action_count, args.corridor_horizon,
            args.noise, args.seed + 400,
        ),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-checkpoint", default="results/direct_corridor_actor_strong_seed0.pt",
    )
    parser.add_argument(
        "--adapted-checkpoint",
        default="results/target_fixed_route_training_equal_lr_seed0.pt",
    )
    parser.add_argument("--rollout-steps", type=int, default=24)
    parser.add_argument("--corridor-horizon", type=int, default=12)
    parser.add_argument("--calibration-count", type=int, default=32)
    parser.add_argument("--action-count", type=int, default=512)
    parser.add_argument("--fisher-draws", type=int, default=2)
    parser.add_argument("--damping", type=float, default=0.03)
    parser.add_argument("--trust-radius", type=float, default=0.20)
    parser.add_argument("--transport-scale", type=float, default=0.125)
    parser.add_argument("--noise", type=float, default=0.025)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--json-out", default="results/policy_transport_alignment_seed0.json",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output = {
        "experiment": "PolicyTransportVsOnlinePPOAlignment",
        "config": vars(args),
        "result": run(args),
    }
    text = json.dumps(output, ensure_ascii=False, indent=2)
    with open(args.json_out, "w", encoding="utf-8") as handle:
        handle.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
