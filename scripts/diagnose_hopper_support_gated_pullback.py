"""Test source-only support gating on distilled Hopper pullback actions."""

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
    one_step_from_observation,
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


def summarize(error, baseline, action_change, gate):
    error = np.asarray(error)
    baseline = np.asarray(baseline)
    return {
        "effect_error_mean": float(error.mean()),
        "relative_to_baseline": float(
            error.mean() / max(baseline.mean(), 1e-8)
        ),
        "improves_fraction": float((error < baseline).mean()),
        "action_change_l2_mean": float(np.mean(action_change)),
        "gate_mean": float(np.mean(gate)),
        "gate_active_fraction": float((np.asarray(gate) > 1e-3).mean()),
    }


@torch.no_grad()
def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    source_policy = FrozenSourcePolicy(
        args.source_model, args.source_norm, device, args.seed,
    )
    source_twin = load_source_twin(args.source_twin_checkpoint, device)
    reference_state, reference_error = collect_source_calibration(
        source_policy,
        source_twin,
        count=args.source_calibration_states,
        seed=args.seed + 100,
        perturb_probability=args.state_perturb_probability,
        perturb_scale=args.state_perturb_scale,
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
    target_context, states = fit_distilled_source_counterfactual_context(
        source_policy,
        basis,
        source_context,
        args,
        device,
    )
    states = np.asarray(states[-args.states:], dtype=np.float32)
    support = calibrator.score(
        torch.as_tensor(states, device=device),
    )
    coverage = support["coverage_ratio"].cpu().numpy()
    uncertainty = support["uncertainty"].cpu().numpy()
    local_source_error = support["local_source_error"].cpu().numpy()
    state_tensor = torch.as_tensor(states, device=device)
    zero_innovation = torch.zeros(
        states.shape[0], basis.action_dim, device=device,
    )
    source_effect = source_context.acceleration(
        basis, state_tensor, zero_innovation,
    )
    target_effect = target_context.acceleration(
        basis, state_tensor, zero_innovation,
    )
    predicted_shift = (
        (target_effect - source_effect) / delta_scale
    ).norm(dim=-1).cpu().numpy()
    signal_to_source_error = predicted_shift / np.maximum(
        local_source_error, 1e-6,
    )
    gates = {
        "ungated": np.ones_like(coverage),
        "hard_source_q95": (coverage <= 1.0).astype(np.float32),
        "soft_source_q95": np.exp(
            -2.0 * np.maximum(coverage - 1.0, 0.0),
        ),
        "calibrated_risk": np.minimum(
            np.exp(1.0 - uncertainty), 1.0,
        ),
        "identifiable_hard": (
            (coverage <= 1.0)
            & (signal_to_source_error >= 1.0)
        ).astype(np.float32),
        "identifiable_margin": (
            (coverage <= 1.0).astype(np.float32)
            * np.clip(
                1.0 - 1.0 / np.maximum(signal_to_source_error, 1.0),
                0.0,
                1.0,
            )
        ),
    }

    source_environment = make_shifted_env(
        SHIFTS["source"], args.seed + 7000,
    )()
    target_environment = make_shifted_env(
        SHIFTS[args.target], args.seed + 7001,
    )()
    source_environment.reset(seed=args.seed + 7000)
    target_environment.reset(seed=args.seed + 7001)
    scale = delta_scale.detach().cpu().numpy().clip(1e-6)
    baseline_errors = []
    gated_errors = {name: [] for name in gates}
    action_changes = {name: [] for name in gates}
    for index, observation in enumerate(states):
        source_action = source_policy.action(observation).cpu().numpy()
        transported_action, _ = cognitive_action_and_features(
            observation,
            source_policy,
            basis,
            source_context,
            target_context,
            delta_scale,
            args.pullback_damping,
            args.effect_metric,
            args.metric_isotropic_floor,
        )
        desired = one_step_from_observation(
            source_environment, observation, source_action,
        )
        baseline_effect = one_step_from_observation(
            target_environment, observation, source_action,
        )
        baseline_errors.append(
            np.linalg.norm((baseline_effect - desired) / scale),
        )
        action_delta = transported_action - source_action
        for name, values in gates.items():
            gated_action = np.clip(
                source_action + values[index] * action_delta,
                -1.0,
                1.0,
            )
            gated_effect = one_step_from_observation(
                target_environment, observation, gated_action,
            )
            gated_errors[name].append(
                np.linalg.norm((gated_effect - desired) / scale),
            )
            action_changes[name].append(
                np.linalg.norm(gated_action - source_action),
            )
    source_environment.close()
    target_environment.close()
    baseline_errors = np.asarray(baseline_errors)
    methods = {
        name: summarize(
            gated_errors[name],
            baseline_errors,
            action_changes[name],
            gates[name],
        )
        for name in gates
    }
    output = {
        "experiment": "HopperSourceSupportGatedDistilledPullback",
        "target": args.target,
        "source_only_gate": True,
        "target_physical_parameters_visible_to_gate": False,
        "source_simulator_used_for_evaluation_only": True,
        "states": int(states.shape[0]),
        "baseline_effect_error_mean": float(baseline_errors.mean()),
        "support_statistics": {
            "coverage_ratio_mean": float(coverage.mean()),
            "coverage_ratio_q50": float(np.quantile(coverage, 0.50)),
            "coverage_ratio_q95": float(np.quantile(coverage, 0.95)),
            "uncertainty_mean": float(uncertainty.mean()),
            "uncertainty_q95": float(np.quantile(uncertainty, 0.95)),
            "predicted_shift_mean": float(predicted_shift.mean()),
            "local_source_error_mean": float(local_source_error.mean()),
            "signal_to_source_error_q50": float(np.quantile(
                signal_to_source_error, 0.50,
            )),
            "signal_to_source_error_q95": float(np.quantile(
                signal_to_source_error, 0.95,
            )),
        },
        "methods": methods,
        "negative_transfer_avoided": {
            name: bool(
                value["effect_error_mean"]
                < methods["ungated"]["effect_error_mean"]
            )
            for name, value in methods.items()
            if name != "ungated"
        },
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
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--target", choices=tuple(SHIFTS), default="combo_medium")
    parser.add_argument("--states", type=int, default=128)
    parser.add_argument("--source-calibration-states", type=int, default=2048)
    parser.add_argument("--neighbors", type=int, default=16)
    parser.add_argument("--covariance-ridge", type=float, default=0.05)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--state-perturb-probability", type=float, default=0.5)
    parser.add_argument("--state-perturb-scale", type=float, default=0.2)
    parser.add_argument("--state-support", type=float, default=1.5)
    parser.add_argument("--trajectory-noise", type=float, default=0.3)
    parser.add_argument("--cognition-warmup", type=int, default=2048)
    parser.add_argument("--warmup-noise", type=float, default=0.3)
    parser.add_argument("--transform-ridge", type=float, default=10.0)
    parser.add_argument("--drift-ridge", type=float, default=100.0)
    parser.add_argument("--pullback-damping", type=float, default=0.05)
    parser.add_argument(
        "--effect-metric", choices=("identity", "critic"), default="identity",
    )
    parser.add_argument("--metric-isotropic-floor", type=float, default=0.05)
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
        default="results/hopper_support_gated_pullback_combo_medium_seed1811.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
