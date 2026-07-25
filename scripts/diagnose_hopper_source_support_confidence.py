"""Gate 1: can source-only joint support predict Hopper twin error?"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from cpbn.hopper_source_twin import JointStateSupportCalibrator
from scripts.diagnose_hopper_pullback_effect import load_source_twin
from scripts.prescreen_hopper_physics_shifts import (
    ENVS,
    SHIFTS,
    make_shifted_env,
)
from scripts.validate_hopper_joint_online_adaptation import FrozenSourcePolicy


def set_observation_state(environment, observation):
    unwrapped = environment.unwrapped
    qpos = unwrapped.data.qpos.copy()
    qvel = unwrapped.data.qvel.copy()
    qpos[0] = 0.0
    qpos[1:] = observation[: qpos.shape[0] - 1]
    qvel[:] = observation[qpos.shape[0] - 1:]
    unwrapped.set_state(qpos, qvel)


def one_step_from_observation(environment, observation, action):
    set_observation_state(environment, observation)
    following, _, _, _, _ = environment.step(action)
    return following - observation


@torch.no_grad()
def source_twin_error(twin, observation, true_delta, device):
    state = torch.as_tensor(
        observation, dtype=torch.float32, device=device,
    ).unsqueeze(0)
    innovation = torch.zeros(
        1, twin.action_dim, dtype=torch.float32, device=device,
    )
    prediction = twin(state, innovation)[0].cpu().numpy()
    scale = twin.delta_scale.detach().cpu().numpy().clip(1e-6)
    return float(np.linalg.norm((prediction - true_delta) / scale))


@torch.no_grad()
def collect_source_calibration(
    source_policy,
    source_twin,
    *,
    count,
    seed,
    perturb_probability,
    perturb_scale,
    state_support,
    trajectory_noise,
    device,
    env="hopper",
):
    trajectory = make_shifted_env(SHIFTS["source"], seed, env)()
    query = make_shifted_env(SHIFTS["source"], seed + 1, env)()
    observation, _ = trajectory.reset(seed=seed)
    query.reset(seed=seed + 1)
    generator = np.random.default_rng(seed + 2)
    state_scale = source_twin.state_scale.detach().cpu().numpy()
    states = []
    errors = []
    for _ in range(count):
        query_observation = observation.copy()
        if generator.random() < perturb_probability:
            query_observation = np.clip(
                query_observation
                + perturb_scale
                * state_scale
                * generator.standard_normal(query_observation.shape),
                -state_support * state_scale,
                state_support * state_scale,
            ).astype(np.float32)
        nominal = source_policy.action(
            query_observation,
        ).cpu().numpy()
        true_delta = one_step_from_observation(
            query, query_observation, nominal,
        )
        states.append(query_observation)
        errors.append(source_twin_error(
            source_twin,
            query_observation,
            true_delta,
            device,
        ))

        trajectory_nominal = source_policy.action(
            observation,
        ).cpu().numpy()
        action = np.clip(
            trajectory_nominal
            + trajectory_noise
            * generator.standard_normal(trajectory_nominal.shape),
            -1.0,
            1.0,
        )
        following, _, terminated, truncated, _ = trajectory.step(action)
        observation = (
            trajectory.reset()[0]
            if terminated or truncated
            else following
        )
    trajectory.close()
    query.close()
    return (
        np.asarray(states, dtype=np.float32),
        np.asarray(errors, dtype=np.float32),
    )


@torch.no_grad()
def collect_target_diagnostic(
    source_policy,
    source_twin,
    *,
    target,
    count,
    seed,
    exploration_noise,
    device,
    env="hopper",
):
    target_environment = make_shifted_env(SHIFTS[target], seed, env)()
    source_query = make_shifted_env(SHIFTS["source"], seed + 1, env)()
    observation, _ = target_environment.reset(seed=seed)
    source_query.reset(seed=seed + 1)
    generator = np.random.default_rng(seed + 2)
    states = []
    errors = []
    for _ in range(count):
        nominal_tensor = source_policy.action(observation)
        nominal = nominal_tensor.cpu().numpy()
        true_source_delta = one_step_from_observation(
            source_query, observation, nominal,
        )
        states.append(observation.copy())
        errors.append(source_twin_error(
            source_twin,
            observation,
            true_source_delta,
            device,
        ))

        amplitude = np.minimum(
            exploration_noise,
            np.maximum(1.0 - np.abs(nominal), 0.0),
        )
        action = nominal + amplitude * generator.choice(
            (-1.0, 1.0), size=nominal.shape,
        )
        following, _, terminated, truncated, _ = target_environment.step(
            action,
        )
        observation = (
            target_environment.reset()[0]
            if terminated or truncated
            else following
        )
    target_environment.close()
    source_query.close()
    return (
        np.asarray(states, dtype=np.float32),
        np.asarray(errors, dtype=np.float32),
    )


def rank_correlation(first, second):
    first = np.asarray(first)
    second = np.asarray(second)
    first_rank = np.argsort(np.argsort(first)).astype(np.float64)
    second_rank = np.argsort(np.argsort(second)).astype(np.float64)
    first_rank -= first_rank.mean()
    second_rank -= second_rank.mean()
    denominator = np.linalg.norm(first_rank) * np.linalg.norm(second_rank)
    return float(np.vdot(first_rank, second_rank) / max(denominator, 1e-12))


def binary_auc(label, score):
    label = np.asarray(label, dtype=bool)
    score = np.asarray(score, dtype=np.float64)
    positive = score[label]
    negative = score[~label]
    if positive.size == 0 or negative.size == 0:
        return None
    comparison = (
        (positive[:, None] > negative[None, :]).mean()
        + 0.5 * (positive[:, None] == negative[None, :]).mean()
    )
    return float(comparison)


def selected_group(error, uncertainty, fraction):
    count = max(1, int(round(len(error) * fraction)))
    order = np.argsort(uncertainty)
    selected = error[order[:count]]
    rejected = error[order[-count:]]
    return {
        "fraction": float(fraction),
        "selected_count": int(count),
        "high_confidence_error_mean": float(selected.mean()),
        "low_confidence_error_mean": float(rejected.mean()),
        "error_ratio": float(
            selected.mean() / max(rejected.mean(), 1e-8)
        ),
    }


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    source_policy = FrozenSourcePolicy(
        args.source_model,
        args.source_norm,
        device,
        args.seed,
        env=args.env,
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
        env=args.env,
    )
    calibrator = JointStateSupportCalibrator(
        torch.as_tensor(reference_state, device=device),
        torch.as_tensor(reference_error, device=device),
        source_twin.state_scale,
        neighbors=args.neighbors,
        covariance_ridge=args.covariance_ridge,
        chunk_size=args.chunk_size,
    )
    target_state, target_error = collect_target_diagnostic(
        source_policy,
        source_twin,
        target=args.target,
        count=args.target_states,
        seed=args.seed + 1000,
        exploration_noise=args.exploration_noise,
        device=device,
        env=args.env,
    )
    score = calibrator.score(
        torch.as_tensor(target_state, device=device),
    )
    score = {
        key: value.detach().cpu().numpy()
        for key, value in score.items()
    }
    normalized_coordinate_max = np.max(
        np.abs(target_state)
        / source_twin.state_scale.detach().cpu().numpy()[None, :],
        axis=-1,
    )
    source_error_q95 = float(np.quantile(reference_error, 0.95))
    target_inlier = target_error <= source_error_q95
    group_curves = [
        selected_group(target_error, score["uncertainty"], fraction)
        for fraction in (0.10, 0.25, 0.50)
    ]
    high_quartile = group_curves[1]
    uncertainty_correlation = rank_correlation(
        score["uncertainty"], target_error,
    )
    output = {
        "experiment": "HopperSourceOnlyJointSupportConfidenceGate",
        "env": args.env,
        "target": args.target,
        "source_only_confidence": True,
        "target_physical_parameters_visible_to_confidence": False,
        "source_simulator_used_for_diagnostic_labels_only": True,
        "source_calibration": {
            "states": int(reference_state.shape[0]),
            "counterfactual_error_mean": float(reference_error.mean()),
            "counterfactual_error_q95": source_error_q95,
            "knn_distance_q95": float(calibrator.distance_scale),
        },
        "target_diagnostic": {
            "states": int(target_state.shape[0]),
            "counterfactual_error_mean": float(target_error.mean()),
            "counterfactual_error_q50": float(np.quantile(
                target_error, 0.50,
            )),
            "counterfactual_error_q95": float(np.quantile(
                target_error, 0.95,
            )),
            "source_calibrated_inlier_fraction": float(target_inlier.mean()),
        },
        "rank_correlations_with_true_error": {
            "joint_coverage": rank_correlation(
                score["coverage_ratio"], target_error,
            ),
            "local_source_error": rank_correlation(
                score["local_error_ratio"], target_error,
            ),
            "calibrated_uncertainty": uncertainty_correlation,
            "coordinatewise_max": rank_correlation(
                normalized_coordinate_max, target_error,
            ),
        },
        "low_error_discrimination_auc": {
            "joint_confidence": binary_auc(
                target_inlier, score["confidence"],
            ),
            "coordinatewise_range": binary_auc(
                target_inlier, -normalized_coordinate_max,
            ),
        },
        "selection_curves": group_curves,
        "gate_thresholds": {
            "uncertainty_error_spearman_min": 0.30,
            "high_to_low_quartile_error_ratio_max": 0.80,
        },
        "gate_pass": bool(
            uncertainty_correlation >= 0.30
            and high_quartile["error_ratio"] <= 0.80
        ),
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
    parser.add_argument("--target", choices=tuple(SHIFTS), default="combo_medium")
    parser.add_argument("--source-calibration-states", type=int, default=4096)
    parser.add_argument("--target-states", type=int, default=512)
    parser.add_argument("--neighbors", type=int, default=16)
    parser.add_argument("--covariance-ridge", type=float, default=0.05)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--state-perturb-probability", type=float, default=0.5)
    parser.add_argument("--state-perturb-scale", type=float, default=0.2)
    parser.add_argument("--state-support", type=float, default=1.5)
    parser.add_argument("--trajectory-noise", type=float, default=0.3)
    parser.add_argument("--exploration-noise", type=float, default=0.3)
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
        "--json-out",
        default="results/hopper_source_support_confidence_combo_medium_seed1811.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
