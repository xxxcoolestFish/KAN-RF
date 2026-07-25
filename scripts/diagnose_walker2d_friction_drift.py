"""Diagnose Walker2d friction_070 early-gain-then-collapse across budgets.

Tracks, for each nested warmup budget:
- the fitted 6x6 control transform T (spectrum, distance from identity);
- the drift correction magnitude |e_0 - b_t| on fresh evaluation states;
- the transported action correction magnitude;
- held-out target prediction RMSE (estimation quality);
- the closed-loop return at the same budget.

The pattern of these curves distinguishes recursive-estimation drift from
transport infeasibility.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.diagnose_hopper_pullback_effect import (
    fit_distilled_source_counterfactual_context,
    load_source_twin,
)
from scripts.prescreen_hopper_physics_shifts import SHIFTS, make_shifted_env
from scripts.validate_hopper_joint_online_adaptation import (
    FrozenSourcePolicy,
    load_cognition,
)
from scripts.validate_hopper_support_gated_policy import evaluate_policy


@torch.no_grad()
def collect_target_transitions(source_policy, args, device, count, seed_offset):
    environment = make_shifted_env(
        SHIFTS[args.target], args.seed + seed_offset, args.env,
    )()
    observation, _ = environment.reset(seed=args.seed + seed_offset)
    generator = torch.Generator(device=device).manual_seed(args.seed + 77)
    rows = []
    for _ in range(count):
        nominal = source_policy.action(observation)
        amplitude = torch.minimum(
            torch.full_like(nominal, args.warmup_noise),
            (1.0 - nominal.abs()).clamp_min(0.0),
        )
        sign = (
            2
            * torch.randint(
                0, 2, nominal.shape, device=device, generator=generator,
            )
            - 1
        )
        action = nominal + amplitude * sign
        following, _, terminated, truncated, _ = environment.step(
            action.cpu().numpy(),
        )
        rows.append(
            (
                observation.copy(),
                (action - nominal).cpu().numpy(),
                (following - observation).copy(),
            )
        )
        observation = (
            environment.reset()[0] if terminated or truncated else following
        )
    environment.close()
    return rows


@torch.no_grad()
def holdout_rmse(context, basis, rows, delta_scale, device):
    state = torch.as_tensor(
        np.asarray([row[0] for row in rows]), dtype=torch.float32,
        device=device,
    )
    innovation = torch.as_tensor(
        np.asarray([row[1] for row in rows]), dtype=torch.float32,
        device=device,
    )
    delta = torch.as_tensor(
        np.asarray([row[2] for row in rows]), dtype=torch.float32,
        device=device,
    )
    prediction = context.acceleration(basis, state, innovation)
    return float(((prediction - delta) / delta_scale).square().mean().sqrt())


@torch.no_grad()
def evaluation_states(source_policy, args, count):
    environment = make_shifted_env(
        SHIFTS[args.target], args.seed + 10000, args.env,
    )()
    observation, _ = environment.reset(seed=args.seed + 10000)
    states = []
    for _ in range(count):
        action = source_policy.action(observation).cpu().numpy()
        observation, _, terminated, truncated, _ = environment.step(action)
        states.append(observation.copy())
        if terminated or truncated:
            observation, _ = environment.reset()
    environment.close()
    return np.asarray(states, dtype=np.float32)


@torch.no_grad()
def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    source_policy = FrozenSourcePolicy(
        args.source_model, args.source_norm, device, args.seed, env=args.env,
    )
    source_twin = load_source_twin(args.source_twin_checkpoint, device)
    basis, source_context, _, delta_scale = load_cognition(args, device)
    holdout = collect_target_transitions(
        source_policy, args, device, args.holdout, 900,
    )
    eval_states = torch.as_tensor(
        evaluation_states(source_policy, args, args.eval_states),
        dtype=torch.float32,
        device=device,
    )
    zero = torch.zeros(
        eval_states.shape[0], basis.action_dim, device=device,
    )
    source_effect = source_context.acceleration(basis, eval_states, zero)
    budgets = tuple(int(value) for value in args.budgets.split(","))
    records = []
    original_warmup = args.cognition_warmup
    for budget in budgets:
        args.cognition_warmup = budget
        target_context, _ = fit_distilled_source_counterfactual_context(
            source_policy, basis, source_context, args, device, source_twin,
        )
        transform = target_context.estimated_transform.to(device)
        target_effect = target_context.acceleration(
            basis, eval_states, zero,
        )
        effect_gap = (
            (source_effect - target_effect) / delta_scale
        ).norm(dim=-1)
        correction = target_context.transport_action(
            basis,
            eval_states,
            source_effect,
            zero,
            regularization=args.pullback_damping,
        )
        correction_norm = correction.norm(dim=-1)
        closed_loop = evaluate_policy(
            "ungated",
            source_policy,
            None,
            basis,
            source_context,
            target_context,
            delta_scale,
            args,
            device,
        )
        record = {
            "budget": budget,
            "closed_loop_return": closed_loop["mean_return"],
            "closed_loop_length": closed_loop["mean_episode_length"],
            "transform_singular_values": [
                float(value)
                for value in torch.linalg.svdvals(transform).cpu()
            ],
            "transform_delta_frobenius": float(
                (
                    transform
                    - torch.eye(
                        basis.action_dim, device=device,
                    )
                ).norm()
            ),
            "transform_design_condition": float(
                target_context.transform_design_condition
            ),
            "drift_delta_norm": float(
                target_context.paired_source_drift_delta_norm
            ),
            "effect_gap_rmse_eval_states": float(
                effect_gap.square().mean().sqrt()
            ),
            "correction_norm_mean": float(correction_norm.mean()),
            "correction_norm_p95": float(
                correction_norm.quantile(0.95)
            ),
            "correction_per_action_mean_abs": [
                float(value)
                for value in correction.abs().mean(dim=0).cpu()
            ],
            "holdout_target_rmse": holdout_rmse(
                target_context, basis, holdout, delta_scale, device,
            ),
        }
        records.append(record)
        print(record, flush=True)
    args.cognition_warmup = original_warmup
    output = {
        "experiment": "Walker2dFrictionDriftDiagnosis",
        "env": args.env,
        "target": args.target,
        "budgets_nested_warmup_prefix": True,
        "records": records,
        "config": vars(args),
    }
    Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json_out).write_text(
        json.dumps(output, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1811)
    parser.add_argument("--env", default="walker2d")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--target", default="friction_070")
    parser.add_argument("--budgets", default="256,512,1024,2048")
    parser.add_argument("--cognition-warmup", type=int, default=0)
    parser.add_argument("--warmup-noise", type=float, default=0.3)
    parser.add_argument("--transform-ridge", type=float, default=10.0)
    parser.add_argument("--drift-ridge", type=float, default=100.0)
    parser.add_argument("--pullback-damping", type=float, default=0.05)
    parser.add_argument("--holdout", type=int, default=512)
    parser.add_argument("--eval-states", type=int, default=256)
    parser.add_argument("--evaluation-episodes", type=int, default=2)
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
        default="results/walker2d_source_affine_twin_cloud_v2_seed1811.pt",
    )
    parser.add_argument(
        "--cognition-checkpoint",
        default="results/walker2d_source_control_sobolev_pm2_calibrated_seed1811.pt",
    )
    parser.add_argument(
        "--json-out",
        default="results/walker2d_friction_drift_diagnosis_seed1811.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
