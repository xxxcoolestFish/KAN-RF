"""Policy-centered ProtoKAN cognition for identifiable Hopper action effects."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from cpbn.generic_affine_kan import (
    CompactInteractionKANDictionary,
    RecursiveAffineKANEstimator,
    fit_affine_kan_context,
)
from scripts.prescreen_hopper_physics_shifts import (
    ENVS,
    SHIFTS,
    make_shifted_env,
)
from scripts.validate_hopper_joint_online_adaptation import FrozenSourcePolicy


@torch.no_grad()
def collect(source_policy, shift, count, args, device, generator, seed_offset):
    environment = make_shifted_env(shift, args.seed + seed_offset, args.env)()
    observation, _ = environment.reset(seed=args.seed + seed_offset)
    states, innovations, deltas = [], [], []
    for _ in range(count):
        nominal = source_policy.action(observation)
        innovation = args.exploration_noise * torch.randn(
            nominal.shape, device=device, generator=generator,
        )
        action = (nominal + innovation).clamp(-1.0, 1.0)
        innovation = action - nominal
        following, _, terminated, truncated, _ = environment.step(
            action.cpu().numpy(),
        )
        states.append(
            torch.as_tensor(observation, dtype=torch.float32, device=device),
        )
        innovations.append(innovation)
        deltas.append(
            torch.as_tensor(
                following - observation,
                dtype=torch.float32,
                device=device,
            ),
        )
        if terminated or truncated:
            observation, _ = environment.reset()
        else:
            observation = following
    environment.close()
    return (
        torch.stack(states),
        torch.stack(innovations),
        torch.stack(deltas),
    )


def normalized_rmse(context, basis, data, delta_scale):
    prediction = context.acceleration(basis, data[0], data[1])
    return float(
        ((prediction - data[2]) / delta_scale).square().mean().sqrt(),
    )


def main(args):
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    generator = torch.Generator(device=device).manual_seed(args.seed + 1)
    source_policy = FrozenSourcePolicy(
        args.model, args.norm, device, args.seed, env=args.env,
    )
    action_dim = int(source_policy.model.action_space.shape[0])
    source = collect(
        source_policy,
        SHIFTS["source"],
        args.source_samples + args.holdout_samples,
        args,
        device,
        generator,
        100,
    )
    source_train = tuple(value[:args.source_samples] for value in source)
    source_holdout = tuple(value[args.source_samples:] for value in source)
    state_scale = source_train[0].abs().quantile(
        0.99, dim=0,
    ).clamp_min(0.1)
    delta_scale = source_train[2].square().mean(
        dim=0,
    ).sqrt().clamp_min(1e-3)
    basis = CompactInteractionKANDictionary(
        state_scale,
        torch.ones(action_dim, device=device),
        pair_modes=args.pair_modes,
    ).to(device)
    source_context = fit_affine_kan_context(
        basis,
        *source_train,
        ridge=args.source_fit_ridge,
    )
    estimator = RecursiveAffineKANEstimator(
        basis,
        source_context,
        ridge=args.cognition_ridge,
        forgetting_factor=args.forgetting_factor,
    )
    for start in range(0, args.source_samples, args.batch_size):
        estimator.update(
            source_train[0][start:start + args.batch_size],
            source_train[1][start:start + args.batch_size],
            source_train[2][start:start + args.batch_size],
        )
    Path(args.checkpoint_out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_scale": state_scale.detach().cpu(),
            "delta_scale": delta_scale.detach().cpu(),
            "source_coefficients": source_context.coefficients.detach().cpu(),
            "estimator_precision": estimator.precision.detach().cpu(),
            "estimator_right": estimator.right.detach().cpu(),
            "estimator_base_precision": estimator.base_precision.detach().cpu(),
            "estimator_base_right": estimator.base_right.detach().cpu(),
            "pair_modes": args.pair_modes,
            "action_dim": action_dim,
            "forgetting_factor": args.forgetting_factor,
            "cognition_ridge": args.cognition_ridge,
            "source_fit_ridge": args.source_fit_ridge,
            "policy_centered": True,
            "exploration_noise": args.exploration_noise,
        },
        args.checkpoint_out,
    )
    target_learning = collect(
        source_policy,
        SHIFTS[args.target],
        max(args.budgets),
        args,
        device,
        generator,
        200,
    )
    target_holdout = collect(
        source_policy,
        SHIFTS[args.target],
        args.holdout_samples,
        args,
        device,
        generator,
        300,
    )
    budgets = sorted(set(args.budgets) | {0})
    records = []

    def record(transitions):
        context = estimator.context()
        item = {
            "transitions": transitions,
            "source_normalized_rmse": normalized_rmse(
                context, basis, source_holdout, delta_scale,
            ),
            "target_normalized_rmse": normalized_rmse(
                context, basis, target_holdout, delta_scale,
            ),
        }
        records.append(item)
        print(item, flush=True)

    record(0)
    transitions = 0
    while transitions < max(budgets):
        stop = min(transitions + args.batch_size, max(budgets))
        estimator.update(
            target_learning[0][transitions:stop],
            target_learning[1][transitions:stop],
            target_learning[2][transitions:stop],
        )
        transitions = stop
        if transitions in budgets:
            record(transitions)
    output = {
        "experiment": "HopperPolicyCenteredProtoKANCognition",
        "env": args.env,
        "seed": args.seed,
        "device": str(device),
        "target": args.target,
        "physical_parameters_visible_to_learner": False,
        "reward_used_for_cognition": False,
        "policy_centered": True,
        "state_dim": basis.state_dim,
        "action_dim": action_dim,
        "feature_dim": basis.feature_dim,
        "context_feature_dim": (1 + basis.action_dim) * basis.feature_dim,
        "config": vars(args),
        "records": records,
    }
    Path(args.json_out).write_text(
        json.dumps(output, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1811)
    parser.add_argument(
        "--env", choices=tuple(ENVS), default="hopper",
    )
    parser.add_argument("--target", default="combo_mild")
    parser.add_argument("--source-samples", type=int, default=32768)
    parser.add_argument("--holdout-samples", type=int, default=4096)
    parser.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        default=(0, 256, 1024, 4096, 16384),
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--pair-modes", type=int, default=1)
    parser.add_argument("--exploration-noise", type=float, default=0.20)
    parser.add_argument("--source-fit-ridge", type=float, default=0.01)
    parser.add_argument("--cognition-ridge", type=float, default=0.1)
    parser.add_argument("--forgetting-factor", type=float, default=0.9999)
    parser.add_argument(
        "--model",
        default="results/hopper_source_sb3_ppo_continued_seed1811.zip",
    )
    parser.add_argument(
        "--norm",
        default="results/hopper_source_sb3_vecnorm_continued_seed1811.pkl",
    )
    parser.add_argument(
        "--checkpoint-out",
        default="results/hopper_source_centered_protokan_seed1811.pt",
    )
    parser.add_argument(
        "--json-out",
        default="results/hopper_centered_protokan_combo_mild_seed1811.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
