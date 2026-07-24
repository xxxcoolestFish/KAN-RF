"""Test whether hidden physics residuals exceed calibrated source model bias."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.prescreen_hopper_physics_shifts import SHIFTS, make_shifted_env
from scripts.validate_hopper_joint_online_adaptation import (
    FrozenSourcePolicy,
    load_cognition,
)


@torch.no_grad()
def collect_scores(
    environment_name,
    source_policy,
    basis,
    source_context,
    delta_scale,
    args,
    device,
):
    environment = make_shifted_env(
        SHIFTS[environment_name],
        args.seed + 1000,
    )()
    observation, _ = environment.reset(seed=args.seed + 1000)
    generator = torch.Generator(device=device).manual_seed(args.seed + 3)
    full_scores = []
    control_scores = []
    residuals = []
    for _ in range(args.transitions):
        state = torch.as_tensor(
            observation, dtype=torch.float32, device=device,
        ).unsqueeze(0)
        nominal = source_policy.action(observation)
        action = (
            nominal
            + args.exploration_noise
            * torch.randn(
                nominal.shape,
                device=device,
                generator=generator,
            )
        ).clamp(-1.0, 1.0)
        following, _, terminated, truncated, _ = environment.step(
            action.cpu().numpy(),
        )
        regressor_action = (
            action - nominal
            if getattr(basis, "policy_centered", False)
            else action
        ).unsqueeze(0)
        prediction = source_context.acceleration(
            basis, state, regressor_action,
        )[0]
        observed = torch.as_tensor(
            following - observation,
            dtype=torch.float32,
            device=device,
        )
        residual = (observed - prediction) / delta_scale
        _, gain = source_context.drift_and_gain(basis, state)
        scaled_gain = gain[0] / delta_scale[:, None]
        coordinate = torch.linalg.lstsq(
            scaled_gain, residual.unsqueeze(-1),
        ).solution.squeeze(-1)
        full_scores.append(float(residual.norm()))
        control_scores.append(float(coordinate.norm()))
        residuals.append(residual.cpu().numpy())
        observation = (
            environment.reset()[0]
            if terminated or truncated
            else following
        )
    environment.close()
    return {
        "full": np.asarray(full_scores),
        "control": np.asarray(control_scores),
        "residual": np.asarray(residuals),
    }


def auc(source, target):
    comparison = target[:, None] - source[None, :]
    return float(
        (
            (comparison > 0.0).astype(np.float64)
            + 0.5 * (comparison == 0.0).astype(np.float64)
        ).mean(),
    )


def comparison(source, target):
    threshold = float(np.quantile(source, 0.95))
    return {
        "source_mean": float(source.mean()),
        "source_std": float(source.std()),
        "source_q95": threshold,
        "target_mean": float(target.mean()),
        "target_std": float(target.std()),
        "target_above_source_q95": float((target > threshold).mean()),
        "separation_auc": auc(source, target),
    }


def main(args):
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    source_policy = FrozenSourcePolicy(
        args.source_model, args.source_norm, device, args.seed,
    )
    basis, source_context, _, delta_scale = load_cognition(
        args, device,
    )
    source = collect_scores(
        "source",
        source_policy,
        basis,
        source_context,
        delta_scale,
        args,
        device,
    )
    target = collect_scores(
        args.target,
        source_policy,
        basis,
        source_context,
        delta_scale,
        args,
        device,
    )
    per_dimension_source_q95 = np.quantile(
        np.abs(source["residual"]), 0.95, axis=0,
    )
    output = {
        "experiment": "HopperSourceBiasTargetShiftSeparability",
        "source_only_model": True,
        "target_parameters_visible_to_learner": False,
        "target_label_used_for_diagnostic_only": True,
        "full_residual": comparison(source["full"], target["full"]),
        "control_projected_residual": comparison(
            source["control"], target["control"],
        ),
        "per_dimension_source_abs_q95": per_dimension_source_q95.tolist(),
        "target_dimension_exceedance": (
            np.abs(target["residual"]) > per_dimension_source_q95
        ).mean(axis=0).tolist(),
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
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--target", default="combo_mild")
    parser.add_argument("--transitions", type=int, default=512)
    parser.add_argument("--exploration-noise", type=float, default=0.2)
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
        default="results/hopper_source_control_sobolev_seed1811.pt",
    )
    parser.add_argument(
        "--json-out",
        default="results/hopper_residual_separability_seed1811.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
