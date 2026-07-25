"""Distill a broad-support source Hopper simulator into an affine neural twin."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from cpbn.hopper_source_twin import (
    HopperSourceAffineTwin,
    SparseComposableKANTwin,
)
from scripts.prescreen_hopper_physics_shifts import (
    ENVS,
    SHIFTS,
    make_shifted_env,
)
from scripts.train_hopper_control_sobolev_cognition import collect_probes
from scripts.validate_hopper_joint_online_adaptation import FrozenSourcePolicy


def set_observation_state(environment, observation):
    unwrapped = environment.unwrapped
    qpos = unwrapped.data.qpos.copy()
    qvel = unwrapped.data.qvel.copy()
    qpos[0] = 0.0
    qpos[1:] = observation[: qpos.shape[0] - 1]
    qvel[:] = observation[qpos.shape[0] - 1:]
    unwrapped.set_state(qpos, qvel)


@torch.no_grad()
def collect_transitions(
    source_policy,
    count,
    seed,
    noise_levels,
    state_scale,
    perturb_probability,
    perturb_scale,
    state_support,
    env="hopper",
):
    trajectory = make_shifted_env(SHIFTS["source"], seed, env)()
    query = make_shifted_env(SHIFTS["source"], seed + 1, env)()
    observation, _ = trajectory.reset(seed=seed)
    query.reset(seed=seed + 1)
    generator = np.random.default_rng(seed + 1)
    states = []
    innovations = []
    deltas = []
    for index in range(count):
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
        nominal = source_policy.action(query_observation).cpu().numpy()
        noise = noise_levels[index % len(noise_levels)]
        amplitude = np.minimum(noise, np.maximum(1.0 - np.abs(nominal), 0.0))
        innovation = amplitude * generator.choice(
            (-1.0, 1.0), size=nominal.shape,
        )
        action = nominal + innovation
        set_observation_state(query, query_observation)
        query_following, _, _, _, _ = query.step(action)
        trajectory_nominal = source_policy.action(observation).cpu().numpy()
        trajectory_noise = noise_levels[
            (index + 1) % len(noise_levels)
        ]
        trajectory_action = np.clip(
            trajectory_nominal
            + trajectory_noise
            * generator.standard_normal(trajectory_nominal.shape),
            -1.0,
            1.0,
        )
        following, _, terminated, truncated, _ = trajectory.step(
            trajectory_action,
        )
        states.append(query_observation)
        innovations.append(innovation)
        deltas.append(query_following - query_observation)
        observation = (
            trajectory.reset()[0]
            if terminated or truncated
            else following
        )
    trajectory.close()
    query.close()
    return tuple(
        np.asarray(value, dtype=np.float32)
        for value in (states, innovations, deltas)
    )


def batch_indices(count, batch_size, device):
    return torch.randint(
        0, count, (batch_size,), device=device,
    )


@torch.no_grad()
def metrics(model, transition, probe):
    state, innovation, delta = transition
    prediction = model(state, innovation)
    transition_rmse = (
        (prediction - delta) / model.delta_scale
    ).square().mean().sqrt()
    probe_state, baseline, jacobian = probe
    predicted_baseline, predicted_gain = model.drift_and_gain(probe_state)
    baseline_rmse = (
        (predicted_baseline - baseline) / model.delta_scale
    ).square().mean().sqrt()
    scaled_gain = predicted_gain / model.delta_scale[None, :, None]
    scaled_jacobian = jacobian / model.delta_scale[None, :, None]
    relative = (
        (scaled_gain - scaled_jacobian)
        .square().sum(dim=(1, 2)).sqrt()
        / scaled_jacobian.square().sum(
            dim=(1, 2),
        ).sqrt().clamp_min(1e-8)
    )
    cosine = nn.functional.cosine_similarity(
        scaled_gain.flatten(1),
        scaled_jacobian.flatten(1),
        dim=-1,
    )
    return {
        "transition_normalized_rmse": float(transition_rmse),
        "baseline_normalized_rmse": float(baseline_rmse),
        "jacobian_relative_error_mean": float(relative.mean()),
        "jacobian_cosine_mean": float(cosine.mean()),
        "jacobian_positive_cosine_fraction": float(
            (cosine > 0.0).float().mean(),
        ),
    }


def tensor_tuple(arrays, device):
    return tuple(torch.as_tensor(array, device=device) for array in arrays)


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    source_policy = FrozenSourcePolicy(
        args.source_model,
        args.source_norm,
        torch.device("cpu"),
        args.seed,
        env=args.env,
    )
    template = torch.load(
        args.template_checkpoint,
        map_location="cpu",
        weights_only=True,
    )
    noise_levels = tuple(
        float(value) for value in args.noise_levels.split(",")
    )
    state_scale = np.asarray(
        template["state_scale"], dtype=np.float32,
    )
    train_transition_np = collect_transitions(
        source_policy,
        args.train_transitions,
        args.seed + 100,
        noise_levels,
        state_scale,
        args.state_perturb_probability,
        args.state_perturb_scale,
        args.state_support,
        env=args.env,
    )
    validation_transition_np = collect_transitions(
        source_policy,
        args.validation_transitions,
        args.seed + 200,
        noise_levels,
        state_scale,
        args.state_perturb_probability,
        args.state_perturb_scale,
        args.state_support,
        env=args.env,
    )
    train_probe_np = collect_probes(
        source_policy,
        args.train_probes,
        args.probe_epsilon,
        args.seed + 300,
        args.probe_trajectory_noise,
        env=args.env,
    )[:3]
    validation_probe_np = collect_probes(
        source_policy,
        args.validation_probes,
        args.probe_epsilon,
        args.seed + 400,
        args.probe_trajectory_noise,
        env=args.env,
    )[:3]
    delta_scale = torch.as_tensor(
        train_transition_np[2],
    ).square().mean(dim=0).sqrt().clamp_min(1e-4)
    if args.model_type == "sparse_kan":
        model = SparseComposableKANTwin(
            template["state_scale"],
            delta_scale,
            grid_size=args.grid_size,
            spline_order=args.spline_order,
            pair_modes=args.pair_modes,
            support=args.state_support,
        ).to(device)
    else:
        model = HopperSourceAffineTwin(
            template["state_scale"],
            delta_scale,
            hidden_dim=args.hidden_dim,
            depth=args.depth,
        ).to(device)
    train_transition = tensor_tuple(train_transition_np, device)
    validation_transition = tensor_tuple(
        validation_transition_np, device,
    )
    train_probe = tensor_tuple(train_probe_np, device)
    validation_probe = tensor_tuple(validation_probe_np, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    for step in range(1, args.gradient_steps + 1):
        transition_index = batch_indices(
            args.train_transitions, args.batch_size, device,
        )
        probe_index = batch_indices(
            args.train_probes, args.probe_batch_size, device,
        )
        state = train_transition[0][transition_index]
        innovation = train_transition[1][transition_index]
        target = train_transition[2][transition_index]
        prediction = model(state, innovation)
        transition_loss = (
            (prediction - target) / model.delta_scale
        ).square().mean()
        probe_state = train_probe[0][probe_index]
        target_baseline = train_probe[1][probe_index]
        target_gain = train_probe[2][probe_index]
        baseline, gain = model.drift_and_gain(probe_state)
        baseline_loss = (
            (baseline - target_baseline) / model.delta_scale
        ).square().mean()
        gain_loss = (
            (
                gain - target_gain
            ) / model.delta_scale[None, :, None]
        ).square().mean()
        loss = (
            transition_loss
            + args.baseline_weight * baseline_loss
            + args.jacobian_weight * gain_loss
        )
        sparsity_loss = torch.zeros((), device=device)
        if hasattr(model, "group_sparsity"):
            sparsity_loss = model.group_sparsity()
            loss = loss + args.group_sparsity_weight * sparsity_loss
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if step % args.report_every == 0 or step == args.gradient_steps:
            print({
                "step": step,
                "loss": float(loss.detach()),
                "transition_loss": float(transition_loss.detach()),
                "baseline_loss": float(baseline_loss.detach()),
                "gain_loss": float(gain_loss.detach()),
                "sparsity_loss": float(sparsity_loss.detach()),
            }, flush=True)
    train_metrics = metrics(model, train_transition, train_probe)
    validation_metrics = metrics(
        model, validation_transition, validation_probe,
    )
    payload = {
        "model_state": {
            key: value.detach().cpu()
            for key, value in model.state_dict().items()
        },
        "state_scale": model.state_scale.detach().cpu(),
        "delta_scale": model.delta_scale.detach().cpu(),
        "model_type": args.model_type,
        "hidden_dim": args.hidden_dim,
        "depth": args.depth,
        "grid_size": args.grid_size,
        "spline_order": args.spline_order,
        "pair_modes": args.pair_modes,
        "support": args.state_support,
        "action_dim": model.action_dim,
        "policy_centered": True,
        "source_only": True,
        "control_sobolev_training": True,
    }
    if hasattr(model, "active_group_fraction"):
        output_active_group_fraction = model.active_group_fraction()
    else:
        output_active_group_fraction = None
    output = {
        "experiment": "HopperSourceAffineNeuralTwin",
        "env": args.env,
        "source_only": True,
        "physical_parameters_visible": False,
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "active_group_fraction": output_active_group_fraction,
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
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--train-transitions", type=int, default=30000)
    parser.add_argument("--validation-transitions", type=int, default=4096)
    parser.add_argument("--train-probes", type=int, default=4096)
    parser.add_argument("--validation-probes", type=int, default=512)
    parser.add_argument("--noise-levels", default="0.0,0.15,0.3,0.5")
    parser.add_argument(
        "--state-perturb-probability", type=float, default=0.5,
    )
    parser.add_argument("--state-perturb-scale", type=float, default=0.2)
    parser.add_argument("--probe-epsilon", type=float, default=0.1)
    parser.add_argument("--probe-trajectory-noise", type=float, default=0.3)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument(
        "--model-type",
        choices=("dense_mlp", "sparse_kan"),
        default="dense_mlp",
    )
    parser.add_argument("--grid-size", type=int, default=6)
    parser.add_argument("--spline-order", type=int, default=2)
    parser.add_argument("--pair-modes", type=int, default=4)
    parser.add_argument("--gradient-steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--probe-batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--baseline-weight", type=float, default=1.0)
    parser.add_argument("--jacobian-weight", type=float, default=0.2)
    parser.add_argument("--group-sparsity-weight", type=float, default=1e-4)
    parser.add_argument("--state-support", type=float, default=1.5)
    parser.add_argument("--report-every", type=int, default=500)
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
        default="results/hopper_source_control_sobolev_calibrated_seed1811.pt",
    )
    parser.add_argument(
        "--checkpoint-out",
        default="results/hopper_source_affine_twin_seed1811.pt",
    )
    parser.add_argument(
        "--json-out",
        default="results/hopper_source_affine_twin_seed1811.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
