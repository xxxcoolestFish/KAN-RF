"""Closed-loop evaluation of support- and identifiability-gated pullback."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from cpbn.hopper_source_twin import JointStateSupportCalibrator
from scripts.diagnose_hopper_pullback_effect import (
    fit_distilled_source_counterfactual_context,
    load_source_twin,
)
from scripts.diagnose_hopper_source_support_confidence import (
    collect_source_calibration,
)
from scripts.prescreen_hopper_physics_shifts import SHIFTS, make_shifted_env
from scripts.validate_hopper_joint_online_adaptation import (
    FrozenSourcePolicy,
    cognitive_action_and_features,
    load_cognition,
)


@torch.no_grad()
def cognitive_gate(
    observation,
    calibrator,
    basis,
    source_context,
    target_context,
    delta_scale,
    device,
):
    state = torch.as_tensor(
        observation, dtype=torch.float32, device=device,
    ).unsqueeze(0)
    support = calibrator.score(state)
    zero = torch.zeros(1, basis.action_dim, device=device)
    source_effect = source_context.acceleration(basis, state, zero)
    target_effect = target_context.acceleration(basis, state, zero)
    predicted_shift = (
        (target_effect - source_effect) / delta_scale
    ).norm(dim=-1)
    local_error = support["local_source_error"].clamp_min(1e-6)
    ratio = predicted_shift / local_error
    coverage = support["coverage_ratio"]
    gate = (
        (coverage <= 1.0).float()
        * (1.0 - ratio.clamp_min(1.0).reciprocal())
    )
    return float(gate.item()), float(coverage.item()), float(ratio.item())


@torch.no_grad()
def evaluate_policy(
    mode,
    source_policy,
    calibrator,
    basis,
    source_context,
    target_context,
    delta_scale,
    args,
    device,
):
    returns = []
    lengths = []
    gates = []
    coverage = []
    signal_ratios = []
    action_changes = []
    for episode in range(args.evaluation_episodes):
        environment = make_shifted_env(
            SHIFTS[args.target], args.seed + 10000 + episode,
        )()
        observation, _ = environment.reset(
            seed=args.seed + 10000 + episode,
        )
        total = 0.0
        length = 0
        while True:
            source_action = source_policy.action(
                observation,
            ).cpu().numpy()
            if mode == "source":
                action = source_action
                gate = 0.0
                state_coverage = 0.0
                signal_ratio = 0.0
            else:
                transported, _ = cognitive_action_and_features(
                    observation,
                    source_policy,
                    basis,
                    source_context,
                    target_context,
                    delta_scale,
                    args.pullback_damping,
                    "identity",
                    0.05,
                )
                if mode == "ungated":
                    gate = 1.0
                    state_coverage = 0.0
                    signal_ratio = 0.0
                else:
                    gate, state_coverage, signal_ratio = cognitive_gate(
                        observation,
                        calibrator,
                        basis,
                        source_context,
                        target_context,
                        delta_scale,
                        device,
                    )
                action = np.clip(
                    source_action + gate * (transported - source_action),
                    -1.0,
                    1.0,
                )
            following, reward, terminated, truncated, _ = environment.step(
                action,
            )
            total += reward
            length += 1
            gates.append(gate)
            coverage.append(state_coverage)
            signal_ratios.append(signal_ratio)
            action_changes.append(float(np.linalg.norm(
                action - source_action,
            )))
            observation = following
            if terminated or truncated:
                break
        environment.close()
        returns.append(total)
        lengths.append(length)
    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_episode_length": float(np.mean(lengths)),
        "gate_mean": float(np.mean(gates)),
        "gate_active_fraction": float((np.asarray(gates) > 1e-3).mean()),
        "coverage_ratio_mean": float(np.mean(coverage)),
        "signal_to_source_error_mean": float(np.mean(signal_ratios)),
        "action_change_l2_mean": float(np.mean(action_changes)),
        "episode_returns": [float(value) for value in returns],
    }


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    source_policy = FrozenSourcePolicy(
        args.source_model, args.source_norm, device, args.seed,
    )
    source_twin = load_source_twin(args.source_twin_checkpoint, device)
    modes = tuple(
        value.strip()
        for value in args.evaluation_modes.split(",")
        if value.strip()
    )
    allowed_modes = {"source", "ungated", "identifiable_margin"}
    if not modes or any(mode not in allowed_modes for mode in modes):
        raise ValueError(
            "evaluation modes must be a comma-separated subset of "
            "source,ungated,identifiable_margin"
        )
    calibrator = None
    if "identifiable_margin" in modes:
        reference_state, reference_error = collect_source_calibration(
            source_policy,
            source_twin,
            count=args.source_calibration_states,
            seed=args.seed + 100,
            perturb_probability=0.0,
            perturb_scale=0.0,
            state_support=args.state_support,
            trajectory_noise=args.trajectory_noise,
            device=device,
        )
        calibrator = JointStateSupportCalibrator(
            torch.as_tensor(reference_state, device=device),
            torch.as_tensor(reference_error, device=device),
            source_twin.state_scale,
            neighbors=args.neighbors,
            covariance_ridge=args.covariance_ridge,
            chunk_size=args.chunk_size,
        )
    basis, source_context, _, delta_scale = load_cognition(args, device)
    target_context, _ = fit_distilled_source_counterfactual_context(
        source_policy,
        basis,
        source_context,
        args,
        device,
    )
    methods = {
        mode: evaluate_policy(
            mode,
            source_policy,
            calibrator,
            basis,
            source_context,
            target_context,
            delta_scale,
            args,
            device,
        )
        for mode in modes
    }
    output = {
        "experiment": "HopperSupportGatedClosedLoopPolicy",
        "target": args.target,
        "source_stage_calibration_only": True,
        "source_simulator_queried_during_target_evaluation": False,
        "target_physical_parameters_visible": False,
        "target_reward_used_for_policy_update": False,
        "methods": methods,
        "config": vars(args),
    }
    if {"source", "identifiable_margin"} <= methods.keys():
        output["gated_improvement_over_source"] = (
            methods["identifiable_margin"]["mean_return"]
            - methods["source"]["mean_return"]
        )
    if {"ungated", "identifiable_margin"} <= methods.keys():
        output["gated_improvement_over_ungated"] = (
            methods["identifiable_margin"]["mean_return"]
            - methods["ungated"]["mean_return"]
        )
    Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json_out).write_text(
        json.dumps(output, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, indent=2), flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1811)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--target", choices=tuple(SHIFTS), default="combo_medium")
    parser.add_argument("--source-calibration-states", type=int, default=2048)
    parser.add_argument("--neighbors", type=int, default=16)
    parser.add_argument("--covariance-ridge", type=float, default=0.05)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--state-support", type=float, default=1.5)
    parser.add_argument("--trajectory-noise", type=float, default=0.3)
    parser.add_argument("--cognition-warmup", type=int, default=2048)
    parser.add_argument("--warmup-noise", type=float, default=0.3)
    parser.add_argument("--transform-ridge", type=float, default=10.0)
    parser.add_argument("--drift-ridge", type=float, default=100.0)
    parser.add_argument("--pullback-damping", type=float, default=0.05)
    parser.add_argument("--evaluation-episodes", type=int, default=10)
    parser.add_argument(
        "--evaluation-modes",
        default="source,ungated,identifiable_margin",
    )
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
        default="results/hopper_support_gated_policy_combo_medium_seed1811.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
