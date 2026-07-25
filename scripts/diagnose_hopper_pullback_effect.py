"""Counterfactual test of whether Hopper cognition pullback improves effects."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from cpbn.hopper_source_twin import (
    HopperSourceAffineTwin,
    SparseComposableKANTwin,
)
from cpbn.generic_affine_kan import AffineKANContext
from scripts.prescreen_hopper_physics_shifts import (
    ENVS,
    SHIFTS,
    make_shifted_env,
)
from scripts.validate_hopper_joint_online_adaptation import (
    FrozenSourcePolicy,
    cognitive_action_and_features,
    cognition_warmup,
    load_cognition,
)


@torch.no_grad()
def fit_stein_centered_context(
    source_policy,
    basis,
    args,
    device,
):
    environment = make_shifted_env(
        SHIFTS[args.target], args.seed + 200, args.env,
    )()
    observation, _ = environment.reset(seed=args.seed + 200)
    generator = torch.Generator(device=device).manual_seed(args.seed + 3)
    states = []
    innovations = []
    deltas = []
    for _ in range(args.cognition_warmup):
        nominal = source_policy.action(observation)
        if args.warmup_exploration == "symmetric":
            amplitude = torch.minimum(
                torch.full_like(nominal, args.warmup_noise),
                (1.0 - nominal.abs()).clamp_min(0.0),
            )
            sign = (
                2.0
                * torch.randint(
                    0,
                    2,
                    nominal.shape,
                    device=device,
                    generator=generator,
                )
                - 1.0
            )
            action = nominal + amplitude * sign
        else:
            action = (
                nominal
                + args.warmup_noise
                * torch.randn(
                    nominal.shape,
                    device=device,
                    generator=generator,
                )
            ).clamp(-1.0, 1.0)
        following, _, terminated, truncated, _ = environment.step(
            action.cpu().numpy(),
        )
        states.append(observation.copy())
        innovations.append((action - nominal).cpu().numpy())
        deltas.append(following - observation)
        observation = (
            environment.reset()[0]
            if terminated or truncated
            else following
        )
    environment.close()
    state = torch.as_tensor(
        np.asarray(states), dtype=torch.float32, device=device,
    )
    innovation = torch.as_tensor(
        np.asarray(innovations), dtype=torch.float32, device=device,
    )
    delta = torch.as_tensor(
        np.asarray(deltas), dtype=torch.float32, device=device,
    )
    feature = basis(state)
    identity = torch.eye(
        feature.shape[-1], dtype=feature.dtype, device=device,
    )
    normal = feature.T @ feature + args.stein_ridge * identity
    baseline_coefficient = torch.linalg.solve(
        normal, feature.T @ delta,
    )
    residual = delta - feature @ baseline_coefficient
    variance = innovation.square().mean(dim=0).clamp_min(1e-5)
    gain_coefficients = []
    for index in range(basis.action_dim):
        pseudo_gain = (
            residual
            * innovation[:, index:index + 1]
            / variance[index]
        )
        coefficient = torch.linalg.solve(
            normal, feature.T @ pseudo_gain,
        )
        gain_coefficients.append(
            coefficient * basis.action_scale[index],
        )
    coefficients = torch.cat(
        (baseline_coefficient, *gain_coefficients),
        dim=0,
    )
    return AffineKANContext(coefficients), np.asarray(states)


@torch.no_grad()
def fit_global_control_transform(
    source_policy,
    basis,
    source_context,
    args,
    device,
):
    environment = make_shifted_env(
        SHIFTS[args.target], args.seed + 200, args.env,
    )()
    observation, _ = environment.reset(seed=args.seed + 200)
    generator = torch.Generator(device=device).manual_seed(args.seed + 3)
    states = []
    innovations = []
    deltas = []
    for _ in range(args.cognition_warmup):
        nominal = source_policy.action(observation)
        action = (
            nominal
            + args.warmup_noise
            * torch.randn(
                nominal.shape,
                device=device,
                generator=generator,
            )
        ).clamp(-1.0, 1.0)
        following, _, terminated, truncated, _ = environment.step(
            action.cpu().numpy(),
        )
        states.append(observation.copy())
        innovations.append((action - nominal).cpu().numpy())
        deltas.append(following - observation)
        observation = (
            environment.reset()[0]
            if terminated or truncated
            else following
        )
    environment.close()
    state = torch.as_tensor(
        np.asarray(states), dtype=torch.float32, device=device,
    )
    innovation = torch.as_tensor(
        np.asarray(innovations), dtype=torch.float32, device=device,
    )
    delta = torch.as_tensor(
        np.asarray(deltas), dtype=torch.float32, device=device,
    )
    feature = basis(state)
    _, source_gain = source_context.drift_and_gain(basis, state)
    width = basis.feature_dim
    source_blocks = source_context.coefficients.reshape(
        1 + basis.action_dim, width, -1,
    )
    source_baseline = feature @ source_blocks[0]
    identity = torch.eye(width, dtype=feature.dtype, device=device)
    baseline_normal = (
        feature.T @ feature + args.stein_ridge * identity
    )
    transform = torch.eye(
        basis.action_dim, dtype=feature.dtype, device=device,
    )
    transform_design = torch.stack(
        [
            source_gain[:, :, source_index]
            * innovation[:, target_index:target_index + 1]
            for source_index in range(basis.action_dim)
            for target_index in range(basis.action_dim)
        ],
        dim=-1,
    ).reshape(-1, basis.action_dim ** 2)
    transform_identity = torch.eye(
        basis.action_dim ** 2,
        dtype=feature.dtype,
        device=device,
    )
    for _ in range(args.transform_iterations):
        control_effect = torch.einsum(
            "noi,ij,nj->no",
            source_gain,
            transform,
            innovation,
        )
        baseline_delta_coefficient = torch.linalg.solve(
            baseline_normal,
            feature.T @ (
                delta - source_baseline - control_effect
            ),
        )
        residual = (
            delta
            - source_baseline
            - feature @ baseline_delta_coefficient
        ).reshape(-1)
        vector = torch.linalg.solve(
            transform_design.T @ transform_design
            + args.transform_ridge * transform_identity,
            transform_design.T @ residual
            + args.transform_ridge
            * torch.eye(
                basis.action_dim,
                dtype=feature.dtype,
                device=device,
            ).reshape(-1),
        )
        transform = vector.reshape(
            basis.action_dim, basis.action_dim,
        )
    baseline_coefficient = (
        source_blocks[0] + baseline_delta_coefficient
    )
    transformed_gain = [
        sum(
            source_blocks[source_index + 1]
            * transform[source_index, target_index]
            for source_index in range(basis.action_dim)
        )
        for target_index in range(basis.action_dim)
    ]
    coefficients = torch.cat(
        (baseline_coefficient, *transformed_gain),
        dim=0,
    )
    context = AffineKANContext(coefficients)
    context.estimated_transform = transform.detach().cpu()
    context.transform_design_condition = float(
        torch.linalg.cond(
            transform_design.T @ transform_design
            + args.transform_ridge * transform_identity,
        ),
    )
    return context, np.asarray(states)


@torch.no_grad()
def fit_orthogonal_control_transform(
    source_policy,
    basis,
    source_context,
    args,
    device,
):
    """Identify only target control-gain change with exogenous innovations.

    The source-policy baseline is a nuisance term.  Since the recorded action
    innovation is zero-mean and independent of the current state, its
    population cross-moment with that nuisance is zero.  We therefore estimate
    a compact 3x3 gain transform directly and never fit target drift here.
    """
    environment = make_shifted_env(
        SHIFTS[args.target], args.seed + 200, args.env,
    )()
    observation, _ = environment.reset(seed=args.seed + 200)
    generator = torch.Generator(device=device).manual_seed(args.seed + 3)
    states = []
    innovations = []
    deltas = []
    for _ in range(args.cognition_warmup):
        nominal = source_policy.action(observation)
        if args.warmup_exploration == "symmetric":
            amplitude = torch.minimum(
                torch.full_like(nominal, args.warmup_noise),
                (1.0 - nominal.abs()).clamp_min(0.0),
            )
            sign = (
                2.0
                * torch.randint(
                    0,
                    2,
                    nominal.shape,
                    device=device,
                    generator=generator,
                )
                - 1.0
            )
            action = nominal + amplitude * sign
        else:
            action = (
                nominal
                + args.warmup_noise
                * torch.randn(
                    nominal.shape,
                    device=device,
                    generator=generator,
                )
            ).clamp(-1.0, 1.0)
        following, _, terminated, truncated, _ = environment.step(
            action.cpu().numpy(),
        )
        states.append(observation.copy())
        innovations.append((action - nominal).cpu().numpy())
        deltas.append(following - observation)
        observation = (
            environment.reset()[0]
            if terminated or truncated
            else following
        )
    environment.close()
    state = torch.as_tensor(
        np.asarray(states), dtype=torch.float32, device=device,
    )
    innovation = torch.as_tensor(
        np.asarray(innovations), dtype=torch.float32, device=device,
    )
    delta = torch.as_tensor(
        np.asarray(deltas), dtype=torch.float32, device=device,
    )
    feature = basis(state)
    source_baseline, source_gain = source_context.drift_and_gain(
        basis, state,
    )
    source_control = (
        source_gain @ innovation.unsqueeze(-1)
    ).squeeze(-1)
    residual = (delta - source_baseline - source_control).reshape(-1)
    design = torch.stack(
        [
            source_gain[:, :, source_index]
            * innovation[:, target_index:target_index + 1]
            for source_index in range(basis.action_dim)
            for target_index in range(basis.action_dim)
        ],
        dim=-1,
    ).reshape(-1, basis.action_dim ** 2)
    identity = torch.eye(
        design.shape[-1], dtype=design.dtype, device=device,
    )
    transform_delta = torch.linalg.solve(
        design.T @ design + args.transform_ridge * identity,
        design.T @ residual,
    ).reshape(basis.action_dim, basis.action_dim)
    transform = (
        torch.eye(
            basis.action_dim, dtype=design.dtype, device=device,
        )
        + transform_delta
    )
    width = basis.feature_dim
    source_blocks = source_context.coefficients.reshape(
        1 + basis.action_dim, width, -1,
    )
    transformed_gain = [
        sum(
            source_blocks[source_index + 1]
            * transform[source_index, target_index]
            for source_index in range(basis.action_dim)
        )
        for target_index in range(basis.action_dim)
    ]
    baseline_coefficient = source_blocks[0]
    drift_delta_norm = 0.0
    if args.cognition_mode == "orthogonal_full":
        transformed_control = torch.einsum(
            "noi,ij,nj->no",
            source_gain,
            transform,
            innovation,
        )
        drift_target = delta - source_baseline - transformed_control
        drift_identity = torch.eye(
            feature.shape[-1], dtype=feature.dtype, device=device,
        )
        drift_delta = torch.linalg.solve(
            feature.T @ feature + args.drift_ridge * drift_identity,
            feature.T @ drift_target,
        )
        baseline_coefficient = baseline_coefficient + drift_delta
        drift_delta_norm = float(drift_delta.norm())
    elif args.cognition_mode == "orthogonal_constant_drift":
        target_gain = torch.einsum(
            "noi,ij->noj",
            source_gain,
            transform,
        )
        transformed_control = (
            target_gain @ innovation.unsqueeze(-1)
        ).squeeze(-1)
        drift_target = delta - source_baseline - transformed_control
        control_normal = (
            target_gain.transpose(-1, -2) @ target_gain
        ).sum(dim=0)
        control_right = (
            target_gain.transpose(-1, -2)
            @ drift_target.unsqueeze(-1)
        ).sum(dim=0).squeeze(-1)
        control_coordinate = torch.linalg.solve(
            control_normal
            + args.drift_ridge
            * torch.eye(
                basis.action_dim,
                dtype=feature.dtype,
                device=device,
            ),
            control_right,
        )
        control_coordinate = control_coordinate - getattr(
            source_context,
            "source_controllable_bias",
            torch.zeros_like(control_coordinate),
        )
        baseline_coefficient = baseline_coefficient + sum(
            transformed_gain[index]
            * control_coordinate[index]
            / basis.action_scale[index]
            for index in range(basis.action_dim)
        )
        drift_delta_norm = float(control_coordinate.norm())
    elif args.cognition_mode == "orthogonal_rank1_drift":
        target_gain = torch.einsum(
            "noi,ij->noj",
            source_gain,
            transform,
        )
        transformed_control = (
            target_gain @ innovation.unsqueeze(-1)
        ).squeeze(-1)
        drift_target = delta - source_baseline - transformed_control
        coordinate_system = (
            target_gain.transpose(-1, -2) @ target_gain
            + args.drift_coordinate_damping
            * torch.eye(
                basis.action_dim,
                dtype=feature.dtype,
                device=device,
            )
        )
        raw_coordinate = torch.linalg.solve(
            coordinate_system,
            target_gain.transpose(-1, -2)
            @ drift_target.unsqueeze(-1),
        ).squeeze(-1)
        raw_coordinate = raw_coordinate - getattr(
            source_context,
            "source_controllable_bias",
            torch.zeros(
                basis.action_dim,
                dtype=feature.dtype,
                device=device,
            ),
        )
        additive_width = 1 + basis.state_dim * basis.contrast_dim
        additive_feature = feature[:, :additive_width]
        additive_identity = torch.eye(
            additive_width, dtype=feature.dtype, device=device,
        )
        coordinate_coefficient = torch.linalg.solve(
            additive_feature.T @ additive_feature
            + args.drift_ridge * additive_identity,
            additive_feature.T @ raw_coordinate,
        )
        smoothed_coordinate = (
            additive_feature @ coordinate_coefficient
        )
        coordinate_mean = smoothed_coordinate.mean(dim=0, keepdim=True)
        centered_coordinate = smoothed_coordinate - coordinate_mean
        _, _, right_h = torch.linalg.svd(
            centered_coordinate, full_matrices=False,
        )
        direction = right_h[0]
        rank1_coordinate = (
            coordinate_mean
            + (centered_coordinate @ direction[:, None]) * direction[None, :]
        )
        drift_effect = (
            target_gain @ rank1_coordinate.unsqueeze(-1)
        ).squeeze(-1)
        lift_identity = torch.eye(
            feature.shape[-1], dtype=feature.dtype, device=device,
        )
        lifted_drift = torch.linalg.solve(
            feature.T @ feature + args.drift_lift_ridge * lift_identity,
            feature.T @ drift_effect,
        )
        baseline_coefficient = baseline_coefficient + lifted_drift
        drift_delta_norm = float(lifted_drift.norm())
    context = AffineKANContext(torch.cat(
        (baseline_coefficient, *transformed_gain),
        dim=0,
    ))
    context.estimated_transform = transform.detach().cpu()
    context.transform_design_condition = float(torch.linalg.cond(
        design.T @ design + args.transform_ridge * identity,
    ))
    context.orthogonal_score_norm = float(
        (design.T @ residual / max(design.shape[0], 1)).norm(),
    )
    context.drift_delta_norm = drift_delta_norm
    return context, np.asarray(states)


@torch.no_grad()
def fit_paired_source_counterfactual_context(
    source_policy,
    basis,
    source_context,
    args,
    device,
):
    """Diagnostic upper bound using a retained source simulator as a twin."""
    target_environment = make_shifted_env(
        SHIFTS[args.target], args.seed + 200, args.env,
    )()
    source_environment = make_shifted_env(
        SHIFTS["source"], args.seed + 201, args.env,
    )()
    observation, _ = target_environment.reset(seed=args.seed + 200)
    source_environment.reset(seed=args.seed + 201)
    generator = torch.Generator(device=device).manual_seed(args.seed + 3)
    states = []
    innovations = []
    differences = []
    for _ in range(args.cognition_warmup):
        nominal = source_policy.action(observation)
        amplitude = torch.minimum(
            torch.full_like(nominal, args.warmup_noise),
            (1.0 - nominal.abs()).clamp_min(0.0),
        )
        sign = (
            2.0
            * torch.randint(
                0,
                2,
                nominal.shape,
                device=device,
                generator=generator,
            )
            - 1.0
        )
        action = nominal + amplitude * sign
        source_delta = one_step_from_observation(
            source_environment,
            observation,
            action.cpu().numpy(),
        )
        following, _, terminated, truncated, _ = target_environment.step(
            action.cpu().numpy(),
        )
        states.append(observation.copy())
        innovations.append((action - nominal).cpu().numpy())
        differences.append(
            (following - observation) - source_delta,
        )
        observation = (
            target_environment.reset()[0]
            if terminated or truncated
            else following
        )
    target_environment.close()
    source_environment.close()
    state = torch.as_tensor(
        np.asarray(states), dtype=torch.float32, device=device,
    )
    innovation = torch.as_tensor(
        np.asarray(innovations), dtype=torch.float32, device=device,
    )
    difference = torch.as_tensor(
        np.asarray(differences), dtype=torch.float32, device=device,
    )
    feature = basis(state)
    _, source_gain = source_context.drift_and_gain(basis, state)
    transform_design = torch.stack(
        [
            source_gain[:, :, source_index]
            * innovation[:, target_index:target_index + 1]
            for source_index in range(basis.action_dim)
            for target_index in range(basis.action_dim)
        ],
        dim=-1,
    ).reshape(-1, basis.action_dim ** 2)
    transform_identity = torch.eye(
        basis.action_dim ** 2,
        dtype=feature.dtype,
        device=device,
    )
    transform_delta = torch.linalg.solve(
        transform_design.T @ transform_design
        + args.transform_ridge * transform_identity,
        transform_design.T @ difference.reshape(-1),
    ).reshape(basis.action_dim, basis.action_dim)
    transform = (
        torch.eye(
            basis.action_dim,
            dtype=feature.dtype,
            device=device,
        )
        + transform_delta
    )
    control_difference = torch.einsum(
        "noi,ij,nj->no",
        source_gain,
        transform_delta,
        innovation,
    )
    drift_difference = difference - control_difference
    drift_identity = torch.eye(
        feature.shape[-1], dtype=feature.dtype, device=device,
    )
    drift_delta = torch.linalg.solve(
        feature.T @ feature + args.drift_ridge * drift_identity,
        feature.T @ drift_difference,
    )
    width = basis.feature_dim
    source_blocks = source_context.coefficients.reshape(
        1 + basis.action_dim, width, -1,
    )
    transformed_gain = [
        sum(
            source_blocks[source_index + 1]
            * transform[source_index, target_index]
            for source_index in range(basis.action_dim)
        )
        for target_index in range(basis.action_dim)
    ]
    context = AffineKANContext(torch.cat(
        (source_blocks[0] + drift_delta, *transformed_gain),
        dim=0,
    ))
    context.estimated_transform = transform.detach().cpu()
    context.transform_design_condition = float(torch.linalg.cond(
        transform_design.T @ transform_design
        + args.transform_ridge * transform_identity,
    ))
    context.paired_source_drift_delta_norm = float(drift_delta.norm())
    return context, np.asarray(states)


def load_source_twin(path, device):
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("model_type") == "sparse_kan":
        model = SparseComposableKANTwin(
            payload["state_scale"],
            payload["delta_scale"],
            action_dim=int(payload["action_dim"]),
            grid_size=int(payload["grid_size"]),
            spline_order=int(payload["spline_order"]),
            pair_modes=int(payload["pair_modes"]),
            support=float(payload["support"]),
        ).to(device)
    else:
        model = HopperSourceAffineTwin(
            payload["state_scale"],
            payload["delta_scale"],
            action_dim=int(payload["action_dim"]),
            hidden_dim=int(payload["hidden_dim"]),
            depth=int(payload["depth"]),
        ).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model


@torch.no_grad()
def fit_distilled_source_counterfactual_context(
    source_policy,
    basis,
    source_context,
    args,
    device,
    source_twin=None,
):
    """Replace the retained source simulator with a distilled source twin."""
    if source_twin is None:
        source_twin = load_source_twin(
            args.source_twin_checkpoint, device,
        )
    target_environment = make_shifted_env(
        SHIFTS[args.target], args.seed + 200, args.env,
    )()
    observation, _ = target_environment.reset(seed=args.seed + 200)
    generator = torch.Generator(device=device).manual_seed(args.seed + 3)
    states = []
    innovations = []
    differences = []
    for _ in range(args.cognition_warmup):
        nominal = source_policy.action(observation)
        amplitude = torch.minimum(
            torch.full_like(nominal, args.warmup_noise),
            (1.0 - nominal.abs()).clamp_min(0.0),
        )
        sign = (
            2.0
            * torch.randint(
                0,
                2,
                nominal.shape,
                device=device,
                generator=generator,
            )
            - 1.0
        )
        innovation = amplitude * sign
        action = nominal + innovation
        state = torch.as_tensor(
            observation, dtype=torch.float32, device=device,
        ).unsqueeze(0)
        source_delta = source_twin(
            state, innovation.unsqueeze(0),
        )[0].cpu().numpy()
        following, _, terminated, truncated, _ = target_environment.step(
            action.cpu().numpy(),
        )
        states.append(observation.copy())
        innovations.append(innovation.cpu().numpy())
        differences.append(
            (following - observation) - source_delta,
        )
        observation = (
            target_environment.reset()[0]
            if terminated or truncated
            else following
        )
    target_environment.close()
    state = torch.as_tensor(
        np.asarray(states), dtype=torch.float32, device=device,
    )
    innovation = torch.as_tensor(
        np.asarray(innovations), dtype=torch.float32, device=device,
    )
    difference = torch.as_tensor(
        np.asarray(differences), dtype=torch.float32, device=device,
    )
    feature = basis(state)
    _, source_gain = source_context.drift_and_gain(basis, state)
    transform_design = torch.stack(
        [
            source_gain[:, :, source_index]
            * innovation[:, target_index:target_index + 1]
            for source_index in range(basis.action_dim)
            for target_index in range(basis.action_dim)
        ],
        dim=-1,
    ).reshape(-1, basis.action_dim ** 2)
    transform_identity = torch.eye(
        basis.action_dim ** 2,
        dtype=feature.dtype,
        device=device,
    )
    transform_delta = torch.linalg.solve(
        transform_design.T @ transform_design
        + args.transform_ridge * transform_identity,
        transform_design.T @ difference.reshape(-1),
    ).reshape(basis.action_dim, basis.action_dim)
    transform = (
        torch.eye(
            basis.action_dim,
            dtype=feature.dtype,
            device=device,
        )
        + transform_delta
    )
    control_difference = torch.einsum(
        "noi,ij,nj->no",
        source_gain,
        transform_delta,
        innovation,
    )
    drift_difference = difference - control_difference
    drift_identity = torch.eye(
        feature.shape[-1], dtype=feature.dtype, device=device,
    )
    drift_delta = torch.linalg.solve(
        feature.T @ feature + args.drift_ridge * drift_identity,
        feature.T @ drift_difference,
    )
    width = basis.feature_dim
    source_blocks = source_context.coefficients.reshape(
        1 + basis.action_dim, width, -1,
    )
    transformed_gain = [
        sum(
            source_blocks[source_index + 1]
            * transform[source_index, target_index]
            for source_index in range(basis.action_dim)
        )
        for target_index in range(basis.action_dim)
    ]
    context = AffineKANContext(torch.cat(
        (source_blocks[0] + drift_delta, *transformed_gain),
        dim=0,
    ))
    context.estimated_transform = transform.detach().cpu()
    context.transform_design_condition = float(torch.linalg.cond(
        transform_design.T @ transform_design
        + args.transform_ridge * transform_identity,
    ))
    context.paired_source_drift_delta_norm = float(drift_delta.norm())
    context.source_twin_model = source_twin
    return context, np.asarray(states)


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


def main(args):
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    source_policy = FrozenSourcePolicy(
        args.source_model, args.source_norm, device, args.seed, env=args.env,
    )
    basis, source_context, estimator, delta_scale = load_cognition(
        args, device,
    )
    if args.cognition_mode == "distilled_source_twin":
        target_context, states = (
            fit_distilled_source_counterfactual_context(
                source_policy,
                basis,
                source_context,
                args,
                device,
            )
        )
    elif args.cognition_mode == "paired_source_oracle":
        target_context, states = fit_paired_source_counterfactual_context(
            source_policy,
            basis,
            source_context,
            args,
            device,
        )
    elif args.cognition_mode in (
        "orthogonal_transform",
        "orthogonal_full",
        "orthogonal_constant_drift",
        "orthogonal_rank1_drift",
    ):
        if not getattr(basis, "policy_centered", False):
            raise ValueError(
                "orthogonal transform requires a policy-centered checkpoint",
            )
        target_context, states = fit_orthogonal_control_transform(
            source_policy,
            basis,
            source_context,
            args,
            device,
        )
    elif args.cognition_mode == "global_transform":
        if not getattr(basis, "policy_centered", False):
            raise ValueError(
                "global transform requires a policy-centered checkpoint",
            )
        target_context, states = fit_global_control_transform(
            source_policy,
            basis,
            source_context,
            args,
            device,
        )
    elif args.cognition_mode == "stein":
        if not getattr(basis, "policy_centered", False):
            raise ValueError(
                "Stein mode requires a policy-centered source checkpoint",
            )
        target_context, states = fit_stein_centered_context(
            source_policy, basis, args, device,
        )
    else:
        states = cognition_warmup(
            source_policy, basis, estimator, args, device,
        )
        target_context = estimator.context()
    states = states[-args.states:]
    source_environment = make_shifted_env(
        SHIFTS["source"], args.seed + 7000, args.env,
    )()
    target_environment = make_shifted_env(
        SHIFTS[args.target], args.seed + 7001, args.env,
    )()
    source_environment.reset(seed=args.seed + 7000)
    target_environment.reset(seed=args.seed + 7001)

    baseline_error = []
    corrected_error = []
    action_change = []
    predicted_error = []
    mismatch_cosine = []
    source_gain_relative_error = []
    target_gain_relative_error = []
    target_gain_cosine = []
    estimated_source_gains = []
    true_source_gains = []
    true_target_gains = []
    true_action_corrections = []
    estimated_action_corrections = []
    source_twin_effect_errors = []
    scale = delta_scale.detach().cpu().numpy().clip(1e-6)
    for observation in states:
        source_action = source_policy.action(
            observation,
        ).cpu().numpy()
        corrected_action, _ = cognitive_action_and_features(
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
        if hasattr(target_context, "source_twin_model"):
            twin_state = torch.as_tensor(
                observation, dtype=torch.float32, device=device,
            ).unsqueeze(0)
            twin_desired = target_context.source_twin_model(
                twin_state,
                torch.zeros(
                    1, basis.action_dim, device=device,
                ),
            )[0].detach().cpu().numpy()
            source_twin_effect_errors.append(
                np.linalg.norm((twin_desired - desired) / scale),
            )
        baseline = one_step_from_observation(
            target_environment, observation, source_action,
        )
        corrected = one_step_from_observation(
            target_environment, observation, corrected_action,
        )
        baseline_error.append(
            np.linalg.norm((baseline - desired) / scale),
        )
        corrected_error.append(
            np.linalg.norm((corrected - desired) / scale),
        )
        action_change.append(
            np.linalg.norm(corrected_action - source_action),
        )
        state = torch.as_tensor(
            observation, dtype=torch.float32, device=device,
        ).unsqueeze(0)
        action = torch.as_tensor(
            corrected_action, dtype=torch.float32, device=device,
        ).unsqueeze(0)
        regressor_source_action = (
            torch.zeros(1, basis.action_dim, device=device)
            if getattr(basis, "policy_centered", False)
            else source_policy.action(observation).unsqueeze(0)
        )
        source_effect = source_context.acceleration(
            basis,
            state,
            regressor_source_action,
        )
        predicted_baseline = target_context.acceleration(
            basis, state, regressor_source_action,
        )
        predicted = target_context.acceleration(
            basis,
            state,
            (
                action - source_policy.action(observation).unsqueeze(0)
                if getattr(basis, "policy_centered", False)
                else action
            ),
        )
        predicted_error.append(float(
            ((predicted - source_effect) / delta_scale)
            .norm(dim=-1)
            .item()
        ))
        predicted_gap = (
            (predicted_baseline - source_effect) / delta_scale
        )[0]
        true_gap = torch.as_tensor(
            (baseline - desired) / scale,
            dtype=torch.float32,
            device=device,
        )
        mismatch_cosine.append(float(
            torch.nn.functional.cosine_similarity(
                predicted_gap.unsqueeze(0),
                true_gap.unsqueeze(0),
            ).item()
        ))
        source_columns = []
        target_columns = []
        for action_index in range(source_action.shape[0]):
            plus = source_action.copy()
            minus = source_action.copy()
            plus[action_index] = min(
                1.0, plus[action_index] + args.jacobian_epsilon,
            )
            minus[action_index] = max(
                -1.0, minus[action_index] - args.jacobian_epsilon,
            )
            denominator = max(
                plus[action_index] - minus[action_index], 1e-6,
            )
            source_columns.append(
                (
                    one_step_from_observation(
                        source_environment, observation, plus,
                    )
                    - one_step_from_observation(
                        source_environment, observation, minus,
                    )
                )
                / denominator
            )
            target_columns.append(
                (
                    one_step_from_observation(
                        target_environment, observation, plus,
                    )
                    - one_step_from_observation(
                        target_environment, observation, minus,
                    )
                )
                / denominator
            )
        true_source_gain = np.stack(source_columns, axis=-1) / scale[:, None]
        true_target_gain = np.stack(target_columns, axis=-1) / scale[:, None]
        _, estimated_source_gain = source_context.drift_and_gain(
            basis, state,
        )
        _, estimated_target_gain = target_context.drift_and_gain(
            basis, state,
        )
        estimated_source_gain = (
            estimated_source_gain[0].cpu().numpy() / scale[:, None]
        )
        estimated_target_gain = (
            estimated_target_gain[0].cpu().numpy() / scale[:, None]
        )
        estimated_source_gains.append(estimated_source_gain)
        true_source_gains.append(true_source_gain)
        true_target_gains.append(true_target_gain)
        target_norm = max(np.linalg.norm(true_target_gain), 1e-8)
        source_gain_relative_error.append(
            np.linalg.norm(estimated_source_gain - true_target_gain)
            / target_norm
        )
        target_gain_relative_error.append(
            np.linalg.norm(estimated_target_gain - true_target_gain)
            / target_norm
        )
        target_gain_cosine.append(
            np.vdot(estimated_target_gain, true_target_gain)
            / max(
                np.linalg.norm(estimated_target_gain)
                * np.linalg.norm(true_target_gain),
                1e-8,
            )
        )
        true_effect_correction = (desired - baseline) / scale
        true_action_correction = np.linalg.lstsq(
            true_target_gain,
            true_effect_correction,
            rcond=None,
        )[0]
        true_action_corrections.append(true_action_correction)
        estimated_action_corrections.append(
            corrected_action - source_action,
        )
    source_environment.close()
    target_environment.close()

    baseline_error = np.asarray(baseline_error)
    corrected_error = np.asarray(corrected_error)
    action_change = np.asarray(action_change)
    predicted_error = np.asarray(predicted_error)
    mismatch_cosine = np.asarray(mismatch_cosine)
    estimated_source_gains = np.asarray(estimated_source_gains)
    true_source_gains = np.asarray(true_source_gains)
    true_target_gains = np.asarray(true_target_gains)
    true_action_corrections = np.asarray(true_action_corrections)
    estimated_action_corrections = np.asarray(
        estimated_action_corrections,
    )
    centered_correction = (
        true_action_corrections - true_action_corrections.mean(axis=0)
    )
    correction_singular = np.linalg.svd(
        centered_correction, compute_uv=False,
    )
    correction_variance = correction_singular**2
    correction_variance_ratio = (
        correction_variance
        / max(correction_variance.sum(), 1e-12)
    )
    correction_cosine = np.sum(
        true_action_corrections * estimated_action_corrections,
        axis=-1,
    ) / np.maximum(
        np.linalg.norm(true_action_corrections, axis=-1)
        * np.linalg.norm(estimated_action_corrections, axis=-1),
        1e-8,
    )

    def oracle_transform_error(source_gains):
        source_matrix = source_gains.reshape(-1, source_gains.shape[-1])
        target_matrix = true_target_gains.reshape(
            -1, true_target_gains.shape[-1],
        )
        transform = np.linalg.lstsq(
            source_matrix, target_matrix, rcond=None,
        )[0]
        prediction = np.einsum("noi,ij->noj", source_gains, transform)
        relative = np.linalg.norm(
            prediction - true_target_gains, axis=(1, 2),
        ) / np.maximum(
            np.linalg.norm(true_target_gains, axis=(1, 2)), 1e-8,
        )
        return float(relative.mean()), transform.tolist()

    model_oracle_error, model_oracle_transform = oracle_transform_error(
        estimated_source_gains,
    )
    physics_oracle_error, physics_oracle_transform = oracle_transform_error(
        true_source_gains,
    )
    normalized_states = np.abs(
        np.asarray(states)
        / basis.state_scale.detach().cpu().numpy()[None, :]
    )
    output = {
        "experiment": "HopperPullbackTrueEffectCounterfactual",
        "env": args.env,
        "target": args.target,
        "states": int(len(states)),
        "physical_parameters_visible_to_learner": False,
        "target_simulator_used_for_evaluation_only": True,
        "baseline_effect_error_mean": float(baseline_error.mean()),
        "corrected_effect_error_mean": float(corrected_error.mean()),
        "error_ratio_mean": float(
            (corrected_error / np.maximum(baseline_error, 1e-8)).mean(),
        ),
        "correction_improves_fraction": float(
            (corrected_error < baseline_error).mean(),
        ),
        "predicted_post_correction_error_mean": float(
            predicted_error.mean(),
        ),
        "predicted_true_mismatch_cosine_mean": float(
            mismatch_cosine.mean(),
        ),
        "predicted_true_mismatch_positive_fraction": float(
            (mismatch_cosine > 0.0).mean(),
        ),
        "action_change_l2_mean": float(action_change.mean()),
        "source_gain_to_true_target_relative_error_mean": float(
            np.mean(source_gain_relative_error),
        ),
        "estimated_gain_to_true_target_relative_error_mean": float(
            np.mean(target_gain_relative_error),
        ),
        "estimated_gain_true_target_cosine_mean": float(
            np.mean(target_gain_cosine),
        ),
        "model_source_oracle_global_transform_relative_error_mean": (
            model_oracle_error
        ),
        "true_source_oracle_global_transform_relative_error_mean": (
            physics_oracle_error
        ),
        "model_source_oracle_global_transform": model_oracle_transform,
        "true_source_oracle_global_transform": physics_oracle_transform,
        "state_coordinate_exceedance_fraction": float(
            (normalized_states > 1.0).mean(),
        ),
        "state_sample_any_exceedance_fraction": float(
            (normalized_states > 1.0).any(axis=1).mean(),
        ),
        "state_coordinate_max_normalized": (
            normalized_states.max(axis=0).tolist()
        ),
        "true_action_correction_mean": (
            true_action_corrections.mean(axis=0).tolist()
        ),
        "true_action_correction_std": (
            true_action_corrections.std(axis=0).tolist()
        ),
        "true_action_correction_pca_variance_ratio": (
            correction_variance_ratio.tolist()
        ),
        "estimated_true_action_correction_cosine_mean": float(
            correction_cosine.mean(),
        ),
        "config": vars(args),
    }
    if hasattr(target_context, "estimated_transform"):
        output["estimated_control_transform"] = (
            target_context.estimated_transform.tolist()
        )
        output["transform_design_condition"] = (
            target_context.transform_design_condition
        )
    if hasattr(target_context, "orthogonal_score_norm"):
        output["orthogonal_score_norm"] = (
            target_context.orthogonal_score_norm
        )
        output["drift_delta_norm"] = target_context.drift_delta_norm
    if hasattr(target_context, "paired_source_drift_delta_norm"):
        output["paired_source_drift_delta_norm"] = (
            target_context.paired_source_drift_delta_norm
        )
    if source_twin_effect_errors:
        output["source_twin_target_state_effect_error_mean"] = float(
            np.mean(source_twin_effect_errors),
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
    parser.add_argument(
        "--env", choices=tuple(ENVS), default="hopper",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--target", default="combo_mild")
    parser.add_argument("--states", type=int, default=128)
    parser.add_argument("--cognition-warmup", type=int, default=256)
    parser.add_argument("--cognition-batch", type=int, default=64)
    parser.add_argument("--warmup-noise", type=float, default=0.05)
    parser.add_argument(
        "--warmup-exploration",
        choices=("gaussian_clipped", "symmetric"),
        default="gaussian_clipped",
    )
    parser.add_argument(
        "--cognition-mode",
        choices=(
            "existing",
            "stein",
            "global_transform",
            "orthogonal_transform",
            "orthogonal_full",
            "orthogonal_constant_drift",
            "orthogonal_rank1_drift",
            "paired_source_oracle",
            "distilled_source_twin",
        ),
        default="existing",
    )
    parser.add_argument("--stein-ridge", type=float, default=1.0)
    parser.add_argument("--transform-ridge", type=float, default=1.0)
    parser.add_argument("--drift-ridge", type=float, default=100.0)
    parser.add_argument(
        "--drift-coordinate-damping", type=float, default=0.05,
    )
    parser.add_argument("--drift-lift-ridge", type=float, default=100.0)
    parser.add_argument("--transform-iterations", type=int, default=5)
    parser.add_argument("--pullback-damping", type=float, default=0.05)
    parser.add_argument("--jacobian-epsilon", type=float, default=0.05)
    parser.add_argument(
        "--effect-metric",
        choices=("identity", "critic"),
        default="identity",
    )
    parser.add_argument("--metric-isotropic-floor", type=float, default=0.05)
    parser.add_argument(
        "--source-model",
        default="results/hopper_source_sb3_ppo_continued_seed1811.zip",
    )
    parser.add_argument(
        "--source-twin-checkpoint",
        default="results/hopper_source_affine_twin_seed1811.pt",
    )
    parser.add_argument(
        "--source-norm",
        default="results/hopper_source_sb3_vecnorm_continued_seed1811.pkl",
    )
    parser.add_argument(
        "--cognition-checkpoint",
        default="results/hopper_source_protokan_cognition_seed1811.pt",
    )
    parser.add_argument(
        "--json-out",
        default="results/hopper_pullback_effect_diagnosis_seed1811.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
