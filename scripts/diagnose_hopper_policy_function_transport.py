"""Diagnose function-space transport from cognitive mechanism coordinates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from cpbn.global_mechanism_kan import (
    GlobalMechanismKANDynamics,
    RecursiveGlobalMechanismEstimator,
)
from scripts.diagnose_hopper_global_physics_context import collect_transitions
from scripts.prescreen_hopper_physics_shifts import SHIFTS, make_shifted_env
from scripts.validate_hopper_joint_online_adaptation import (
    FrozenSourcePolicy,
    load_cognition,
)


@torch.no_grad()
def evaluate(policies, coefficients, target, args):
    environment = make_shifted_env(
        SHIFTS[target], args.seed + 10000,
    )()
    returns, lengths = [], []
    healthy = 0
    action_delta = []
    for episode in range(args.evaluation_episodes):
        observation, _ = environment.reset(
            seed=args.seed + 10000 + episode,
        )
        total = 0.0
        length = 0
        while True:
            source_action = policies[0].action(observation)
            action = source_action.clone()
            for coefficient, policy in zip(
                coefficients, policies[1:],
            ):
                action.add_(
                    coefficient
                    * (policy.action(observation) - source_action)
                )
            action = action.clamp(-1.0, 1.0)
            action_delta.extend(
                torch.abs(action - source_action).cpu().tolist()
            )
            observation, reward, terminated, truncated, _ = (
                environment.step(action.cpu().numpy())
            )
            total += float(reward)
            length += 1
            if terminated or truncated:
                healthy += int(truncated and not terminated)
                break
        returns.append(total)
        lengths.append(length)
    environment.close()
    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_episode_length": float(np.mean(lengths)),
        "healthy_completion_rate": healthy / args.evaluation_episodes,
        "action_delta_abs_mean": float(np.mean(action_delta)),
        "action_delta_abs_p95": float(np.quantile(action_delta, 0.95)),
    }


def main(args):
    torch.manual_seed(args.seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu",
    )
    print({"stage": "setup", "device": str(device)}, flush=True)
    source = FrozenSourcePolicy(
        args.source_model, args.source_norm, device, args.seed,
    )
    historical = [
        FrozenSourcePolicy(model, norm, device, args.seed + index + 1)
        for index, (model, norm) in enumerate(
            zip(args.mechanism_actor_models, args.mechanism_actor_norms)
        )
    ]
    basis, source_context, _, delta_scale = load_cognition(args, device)
    payload = torch.load(
        args.mechanism_checkpoint,
        map_location=device,
        weights_only=True,
    )
    mechanism_model = GlobalMechanismKANDynamics(
        source_context, payload["mechanisms"].to(device),
    )
    estimator = RecursiveGlobalMechanismEstimator(
        mechanism_model,
        basis,
        delta_scale,
        ridge=args.mechanism_latent_ridge,
    )
    adaptation = collect_transitions(
        source,
        args.target,
        args.cognition_warmup,
        args,
        device,
        97000,
    )
    estimator.update(
        adaptation["state"],
        adaptation["innovation"],
        adaptation["delta"],
    )
    scale = payload["latent_scale"].to(device)
    target_coordinate = estimator.latent() / scale
    training_coordinates = payload["training_latents"].to(device) / scale
    coefficients = torch.linalg.lstsq(
        training_coordinates.T,
        target_coordinate,
    ).solution
    transported = evaluate(
        [source, *historical], coefficients, args.target, args,
    )
    frozen = evaluate(
        [source, *historical],
        torch.zeros_like(coefficients),
        args.target,
        args,
    )
    output = {
        "experiment": "HopperPolicyFunctionTransportDiagnostic",
        "seed": args.seed,
        "device": str(device),
        "target": args.target,
        "physical_parameters_visible_to_learner": False,
        "target_coordinate": target_coordinate.cpu().tolist(),
        "historical_policy_coefficients": coefficients.cpu().tolist(),
        "frozen_source": frozen,
        "function_transport": transported,
        "improvement": (
            transported["mean_return"] - frozen["mean_return"]
        ),
        "config": vars(args),
    }
    Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json_out).write_text(
        json.dumps(output, indent=2) + "\n",
        encoding="utf-8",
    )
    print(output, flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1811)
    parser.add_argument(
        "--target",
        choices=("payload_150", "combo_mild", "combo_medium"),
        default="combo_medium",
    )
    parser.add_argument("--cognition-warmup", type=int, default=512)
    parser.add_argument("--exploration-noise", type=float, default=0.2)
    parser.add_argument("--evaluation-episodes", type=int, default=5)
    parser.add_argument("--mechanism-latent-ridge", type=float, default=1e-2)
    parser.add_argument(
        "--source-model",
        default="results/hopper_source_sb3_ppo_continued_seed1811.zip",
    )
    parser.add_argument(
        "--source-norm",
        default="results/hopper_source_sb3_vecnorm_continued_seed1811.pkl",
    )
    parser.add_argument(
        "--cognition-checkpoint",
        default="results/hopper_source_centered_protokan_seed1811.pt",
    )
    parser.add_argument(
        "--mechanism-checkpoint",
        default="results/hopper_global_mechanism_latent_seed1811.pt",
    )
    parser.add_argument(
        "--mechanism-actor-models",
        nargs="+",
        default=(
            "results/hopper_payload125_mechanism_actor_frozennorm_80k_seed1811.zip",
            "results/hopper_friction070_mechanism_actor_frozennorm_80k_seed1811.zip",
            "results/hopper_actuator080_mechanism_actor_frozennorm_80k_seed1811.zip",
        ),
    )
    parser.add_argument(
        "--mechanism-actor-norms",
        nargs="+",
        default=(
            "results/hopper_payload125_mechanism_actor_frozennorm_16k_norm_seed1811.pkl",
            "results/hopper_friction070_mechanism_actor_frozennorm_16k_norm_seed1811.pkl",
            "results/hopper_actuator080_mechanism_actor_frozennorm_16k_norm_seed1811.pkl",
        ),
    )
    parser.add_argument(
        "--json-out",
        default="results/hopper_policy_function_transport_seed1811.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
