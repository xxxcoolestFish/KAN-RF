"""Diagnose finite-difference probe Jacobian quality on Walker2d.

Hypothesis: if eps variants disagree strongly, probe labels (not model
capacity) cap the gain-fit cosine.  Also reports source-action saturation,
which biases clipped probes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.diagnose_hopper_pullback_effect import load_source_twin
from scripts.prescreen_hopper_physics_shifts import SHIFTS, make_shifted_env
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
def main(args):
    device = torch.device(args.device)
    source_policy = FrozenSourcePolicy(
        args.source_model, args.source_norm, device, args.seed, env=args.env,
    )
    trajectory = make_shifted_env(SHIFTS["source"], args.seed, args.env)()
    probe = make_shifted_env(SHIFTS["source"], args.seed + 1, args.env)()
    observation, _ = trajectory.reset(seed=args.seed)
    probe.reset(seed=args.seed + 1)

    states = []
    saturations = []
    while len(states) < args.states:
        nominal = source_policy.action(observation).cpu().numpy()
        saturations.append(np.abs(nominal))
        simulator_state = snapshot(trajectory)
        per_eps = {}
        for epsilon in (0.02, 0.05, 0.1):
            columns = []
            for index in range(nominal.shape[0]):
                plus = nominal.copy()
                minus = nominal.copy()
                plus[index] = min(1.0, plus[index] + epsilon)
                minus[index] = max(-1.0, minus[index] - epsilon)
                denominator = max(plus[index] - minus[index], 1e-6)
                plus_delta = probe_delta(
                    probe, simulator_state, observation, plus,
                )
                minus_delta = probe_delta(
                    probe, simulator_state, observation, minus,
                )
                columns.append((plus_delta - minus_delta) / denominator)
            per_eps[str(epsilon)] = np.stack(columns, axis=-1)
        states.append((observation.copy(), per_eps))
        following, _, terminated, truncated, _ = trajectory.step(nominal)
        observation = (
            trajectory.reset()[0] if terminated or truncated else following
        )
    trajectory.close()
    probe.close()

    saturations = np.asarray(saturations)
    eps_keys = ("0.02", "0.05", "0.1")
    jacobians = {
        key: np.stack([entry[1][key] for entry in states])
        for key in eps_keys
    }
    norms = {
        key: float(np.linalg.norm(value, axis=(1, 2)).mean())
        for key, value in jacobians.items()
    }
    cosine = {}
    for first in eps_keys:
        for second in eps_keys:
            if first < second:
                flat_first = jacobians[first].reshape(len(states), -1)
                flat_second = jacobians[second].reshape(len(states), -1)
                value = np.sum(flat_first * flat_second, axis=1) / np.maximum(
                    np.linalg.norm(flat_first, axis=1)
                    * np.linalg.norm(flat_second, axis=1),
                    1e-8,
                )
                cosine[f"{first}_vs_{second}"] = float(value.mean())
    twin = load_source_twin(args.source_twin_checkpoint, device)
    twin_cosine = {}
    state_tensor = torch.as_tensor(
        np.stack([entry[0] for entry in states]),
        dtype=torch.float32,
        device=device,
    )
    _, predicted_gain = twin.drift_and_gain(state_tensor)
    predicted = predicted_gain.cpu().numpy()
    for key in eps_keys:
        target = jacobians[key]
        relative = (
            np.linalg.norm(predicted - target, axis=(1, 2))
            / np.maximum(np.linalg.norm(target, axis=(1, 2)), 1e-8)
        )
        twin_cosine[f"twin_vs_eps{key}_relative_error"] = float(
            relative.mean()
        )
    output = {
        "experiment": "Walker2dProbeJacobianDiagnostic",
        "states": len(states),
        "action_saturation_fraction": {
            threshold: float((saturations > float(threshold)).mean())
            for threshold in ("0.5", "0.8", "0.9", "0.99")
        },
        "jacobian_norm_by_eps": norms,
        "eps_agreement_cosine": cosine,
        **twin_cosine,
        "config": vars(args),
    }
    Path(args.json_out).write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(output, indent=2), flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1811)
    parser.add_argument("--env", default="walker2d")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--states", type=int, default=64)
    parser.add_argument(
        "--source-model",
        default="results/walker2d_source_sb3_ppo_phase3_seed1811.zip",
    )
    parser.add_argument(
        "--source-norm",
        default="results/walker2d_source_sb3_vecnorm_phase3_seed1811.pkl",
    )
    parser.add_argument(
        "--source-twin-checkpoint",
        default="results/walker2d_source_affine_twin_cloud_seed1811.pt",
    )
    parser.add_argument(
        "--json-out",
        default="results/_walker2d_probe_jacobian_diagnostic.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
