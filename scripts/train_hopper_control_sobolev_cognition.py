"""Train source-only Hopper KAN cognition with black-box Jacobian probes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from cpbn.generic_affine_kan import (
    AffineKANContext,
    CompactInteractionKANDictionary,
)
from scripts.prescreen_hopper_physics_shifts import (
    ENVS,
    SHIFTS,
    make_shifted_env,
)
from scripts.validate_hopper_joint_online_adaptation import FrozenSourcePolicy


def snapshot(environment):
    unwrapped = environment.unwrapped
    return unwrapped.data.qpos.copy(), unwrapped.data.qvel.copy()


def restore(environment, state):
    environment.unwrapped.set_state(state[0], state[1])


def probe_delta(environment, state, observation, action):
    restore(environment, state)
    following, _, _, _, _ = environment.step(action)
    return following - observation


@torch.no_grad()
def collect_probes(
    source_policy,
    count,
    epsilon,
    seed,
    trajectory_noise=0.0,
    env="hopper",
):
    trajectory = make_shifted_env(SHIFTS["source"], seed, env)()
    probe = make_shifted_env(SHIFTS["source"], seed + 1, env)()
    observation, _ = trajectory.reset(seed=seed)
    probe.reset(seed=seed + 1)
    generator = np.random.default_rng(seed + 2)
    states = []
    baselines = []
    jacobians = []
    innovations = []
    while len(states) < count:
        simulator_state = snapshot(trajectory)
        nominal = source_policy.action(observation).cpu().numpy()
        baseline = probe_delta(
            probe, simulator_state, observation, nominal,
        )
        columns = []
        for index in range(nominal.shape[0]):
            plus = nominal.copy()
            minus = nominal.copy()
            plus[index] = min(1.0, plus[index] + epsilon)
            minus[index] = max(-1.0, minus[index] - epsilon)
            denominator = plus[index] - minus[index]
            plus_delta = probe_delta(
                probe, simulator_state, observation, plus,
            )
            minus_delta = probe_delta(
                probe, simulator_state, observation, minus,
            )
            columns.append(
                (plus_delta - minus_delta) / max(denominator, 1e-6),
            )
            plus_innovation = plus - nominal
            minus_innovation = minus - nominal
            innovations.extend((plus_innovation, minus_innovation))
        states.append(observation.copy())
        baselines.append(baseline)
        jacobians.append(np.stack(columns, axis=-1))
        trajectory_action = np.clip(
            nominal
            + trajectory_noise
            * generator.standard_normal(nominal.shape),
            -1.0,
            1.0,
        )
        following, _, terminated, truncated, _ = trajectory.step(
            trajectory_action,
        )
        observation = (
            trajectory.reset()[0]
            if terminated or truncated
            else following
        )
    trajectory.close()
    probe.close()
    return (
        np.asarray(states, dtype=np.float32),
        np.asarray(baselines, dtype=np.float32),
        np.asarray(jacobians, dtype=np.float32),
        np.asarray(innovations, dtype=np.float32),
    )


@torch.no_grad()
def collect_policy_baselines(
    source_policy,
    count,
    seed,
    trajectory_noise=0.0,
    env="hopper",
):
    """Probe nominal effects while a noisy source trajectory broadens support."""
    environment = make_shifted_env(SHIFTS["source"], seed, env)()
    probe = make_shifted_env(SHIFTS["source"], seed + 1, env)()
    observation, _ = environment.reset(seed=seed)
    probe.reset(seed=seed + 1)
    generator = np.random.default_rng(seed + 2)
    states = []
    baselines = []
    for _ in range(count):
        action = source_policy.action(observation).cpu().numpy()
        simulator_state = snapshot(environment)
        baseline = probe_delta(
            probe, simulator_state, observation, action,
        )
        trajectory_action = np.clip(
            action
            + trajectory_noise
            * generator.standard_normal(action.shape),
            -1.0,
            1.0,
        )
        following, _, terminated, truncated, _ = environment.step(
            trajectory_action,
        )
        states.append(observation.copy())
        baselines.append(baseline)
        observation = (
            environment.reset()[0]
            if terminated or truncated
            else following
        )
    environment.close()
    probe.close()
    return (
        np.asarray(states, dtype=np.float32),
        np.asarray(baselines, dtype=np.float32),
    )


def fit_context(
    basis,
    baseline_states,
    baselines,
    gain_states,
    jacobians,
    ridge,
):
    baseline_feature = basis(baseline_states)
    identity = torch.eye(
        baseline_feature.shape[-1],
        dtype=baseline_feature.dtype,
        device=baseline_feature.device,
    )
    normal = baseline_feature.T @ baseline_feature + ridge * identity
    baseline_coefficient = torch.linalg.solve(
        normal, baseline_feature.T @ baselines,
    )
    gain_feature = basis(gain_states)
    gain_normal = gain_feature.T @ gain_feature + ridge * identity
    gain_coefficients = [
        torch.linalg.solve(
            gain_normal,
            gain_feature.T @ jacobians[:, :, index],
        ) * basis.action_scale[index]
        for index in range(basis.action_dim)
    ]
    return AffineKANContext(torch.cat(
        (baseline_coefficient, *gain_coefficients),
        dim=0,
    ))


@torch.no_grad()
def metrics(context, basis, states, baselines, jacobians, delta_scale):
    prediction = context.acceleration(
        basis,
        states,
        torch.zeros(
            states.shape[0], basis.action_dim, device=states.device,
        ),
    )
    _, gain = context.drift_and_gain(basis, states)
    baseline_error = (
        (prediction - baselines) / delta_scale
    ).square().mean().sqrt()
    scaled_gain = gain / delta_scale[None, :, None]
    scaled_target = jacobians / delta_scale[None, :, None]
    relative = (
        (scaled_gain - scaled_target).square().sum(dim=(1, 2)).sqrt()
        / scaled_target.square().sum(dim=(1, 2)).sqrt().clamp_min(1e-8)
    )
    cosine = torch.nn.functional.cosine_similarity(
        scaled_gain.flatten(1),
        scaled_target.flatten(1),
        dim=-1,
    )
    return {
        "baseline_normalized_rmse": float(baseline_error),
        "jacobian_relative_error_mean": float(relative.mean()),
        "jacobian_cosine_mean": float(cosine.mean()),
        "jacobian_positive_cosine_fraction": float(
            (cosine > 0.0).float().mean(),
        ),
    }


def main(args):
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    old = torch.load(
        args.template_checkpoint,
        map_location=device,
        weights_only=True,
    )
    action_dim = int(old.get("action_dim", 3))
    basis = CompactInteractionKANDictionary(
        old["state_scale"],
        torch.ones(action_dim, device=device),
        pair_modes=int(old["pair_modes"]),
    ).to(device)
    basis.policy_centered = True
    source_policy = FrozenSourcePolicy(
        args.source_model, args.source_norm, device, args.seed, env=args.env,
    )
    arrays = collect_probes(
        source_policy,
        args.train_states + args.validation_states,
        args.probe_epsilon,
        args.seed + 100,
        args.trajectory_noise,
        env=args.env,
    )
    state = torch.as_tensor(arrays[0], device=device)
    baseline = torch.as_tensor(arrays[1], device=device)
    jacobian = torch.as_tensor(arrays[2], device=device)
    split = args.train_states
    baseline_arrays = collect_policy_baselines(
        source_policy,
        args.baseline_states + args.baseline_validation_states,
        args.seed + 500,
        args.baseline_trajectory_noise,
        env=args.env,
    )
    baseline_state = torch.as_tensor(baseline_arrays[0], device=device)
    baseline_target = torch.as_tensor(baseline_arrays[1], device=device)
    baseline_split = args.baseline_states
    context = fit_context(
        basis,
        baseline_state[:baseline_split],
        baseline_target[:baseline_split],
        state[:split],
        jacobian[:split],
        args.fit_ridge,
    )
    delta_scale = baseline_target[:baseline_split].square().mean(
        dim=0,
    ).sqrt().clamp_min(1e-4)
    train_metrics = metrics(
        context,
        basis,
        state[:split],
        baseline[:split],
        jacobian[:split],
        delta_scale,
    )
    validation_metrics = metrics(
        context,
        basis,
        state[split:],
        baseline[split:],
        jacobian[split:],
        delta_scale,
    )
    baseline_validation_prediction = context.acceleration(
        basis,
        baseline_state[baseline_split:],
        torch.zeros(
            args.baseline_validation_states,
            basis.action_dim,
            device=device,
        ),
    )
    baseline_validation_rmse = (
        (
            baseline_validation_prediction
            - baseline_target[baseline_split:]
        )
        / delta_scale
    ).square().mean().sqrt()
    validation_metrics["independent_baseline_normalized_rmse"] = float(
        baseline_validation_rmse,
    )
    _, calibration_gain = context.drift_and_gain(
        basis, baseline_state[baseline_split:],
    )
    calibration_residual = (
        baseline_target[baseline_split:]
        - baseline_validation_prediction
    )
    calibration_normal = (
        calibration_gain.transpose(-1, -2) @ calibration_gain
    ).sum(dim=0)
    calibration_right = (
        calibration_gain.transpose(-1, -2)
        @ calibration_residual.unsqueeze(-1)
    ).sum(dim=0).squeeze(-1)
    source_controllable_bias = torch.linalg.solve(
        calibration_normal
        + args.calibration_ridge
        * torch.eye(
            basis.action_dim,
            dtype=state.dtype,
            device=device,
        ),
        calibration_right,
    )
    validation_metrics["source_controllable_bias_norm"] = float(
        source_controllable_bias.norm(),
    )

    innovation = torch.as_tensor(arrays[3], device=device)
    repeated_state = state[:split].repeat_interleave(
        2 * basis.action_dim, dim=0,
    )
    design = basis.context_features(
        repeated_state, innovation[: repeated_state.shape[0]],
    ).to(torch.float64)
    dimension = design.shape[-1]
    precision = (
        design.T @ design
        + args.cognition_ridge
        * torch.eye(dimension, dtype=torch.float64, device=device)
    )
    right = precision @ context.coefficients.to(torch.float64)
    payload = {
        "state_scale": basis.state_scale.detach().cpu(),
        "delta_scale": delta_scale.detach().cpu(),
        "source_coefficients": context.coefficients.detach().cpu(),
        "estimator_precision": precision.detach().cpu(),
        "estimator_right": right.detach().cpu(),
        "estimator_base_precision": precision.detach().cpu(),
        "estimator_base_right": right.detach().cpu(),
        "pair_modes": int(old["pair_modes"]),
        "action_dim": action_dim,
        "forgetting_factor": args.forgetting_factor,
        "cognition_ridge": args.cognition_ridge,
        "source_fit_ridge": args.fit_ridge,
        "policy_centered": True,
        "exploration_noise": args.probe_epsilon,
        "control_sobolev_source_training": True,
        "source_controllable_bias": (
            source_controllable_bias.detach().cpu()
        ),
    }
    output = {
        "experiment": "HopperSourceControlSobolevCognition",
        "env": args.env,
        "source_only": True,
        "physical_parameters_visible": False,
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "config": vars(args),
    }
    Path(args.checkpoint_out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.checkpoint_out)
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
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--train-states", type=int, default=1024)
    parser.add_argument("--validation-states", type=int, default=256)
    parser.add_argument("--baseline-states", type=int, default=8192)
    parser.add_argument(
        "--baseline-validation-states", type=int, default=1024,
    )
    parser.add_argument(
        "--baseline-trajectory-noise",
        type=float,
        default=0.0,
        help="Source-only state exploration used for nominal-effect labels.",
    )
    parser.add_argument("--probe-epsilon", type=float, default=0.1)
    parser.add_argument(
        "--trajectory-noise",
        type=float,
        default=0.0,
        help="Source-only behavior noise used to broaden probed state support.",
    )
    parser.add_argument("--fit-ridge", type=float, default=0.01)
    parser.add_argument("--cognition-ridge", type=float, default=0.1)
    parser.add_argument("--calibration-ridge", type=float, default=1.0)
    parser.add_argument("--forgetting-factor", type=float, default=0.9999)
    parser.add_argument(
        "--source-model",
        default="results/hopper_source_sb3_ppo_continued_seed1811.zip",
    )
    parser.add_argument(
        "--source-norm",
        default="results/hopper_source_sb3_vecnorm_continued_seed1811.pkl",
    )
    parser.add_argument(
        "--template-checkpoint",
        default="results/hopper_source_centered_protokan_seed1811.pt",
    )
    parser.add_argument(
        "--checkpoint-out",
        default="results/hopper_source_control_sobolev_seed1811.pt",
    )
    parser.add_argument(
        "--json-out",
        default="results/hopper_source_control_sobolev_seed1811.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
