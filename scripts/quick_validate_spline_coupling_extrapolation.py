"""Known-context upper-bound gate for spline-coupled dynamics extrapolation.

Two actuator scales provide paired source transitions.  The same qpos, qvel
and action bank is queried at interpolation and extrapolation scales, removing
state-distribution shift from the comparison.  The actuator scale is exposed
to every model in this diagnostic only; a positive result is required before
replacing it with a transition-history latent.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import gymnasium as gym
import mujoco
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from kanrf import KAN, ProtoKAN
from kanrf._regularization import p_spline_penalty
from kanrf.spline_coupling import (
    diffuse_input_dimension_gradients_,
    protokan_hermite_penalty,
)


@dataclass(frozen=True)
class StateActionBank:
    states: np.ndarray
    qpos: np.ndarray
    qvel: np.ndarray
    actions: np.ndarray


class MLPDynamics(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


def full_state(base) -> np.ndarray:
    """Return the Markov, dynamically active state (arm + object)."""
    return np.concatenate(
        (
            np.asarray(base.data.qpos[:9], dtype=np.float32),
            np.asarray(base.data.qvel[:9], dtype=np.float32),
        )
    )


def collect_bank(count: int, seed: int, episode_steps: int) -> StateActionBank:
    """Collect Markov qpos/qvel states and smooth exploratory actions."""
    env = gym.make("Pusher-v5", max_episode_steps=episode_steps)
    rng = np.random.default_rng(seed)
    env.reset(seed=seed)
    previous = np.zeros(env.action_space.shape, dtype=np.float32)
    states, qpositions, qvelocities, actions = [], [], [], []
    for index in range(count):
        if index > 0 and index % episode_steps == 0:
            env.reset(seed=seed + index)
            previous.fill(0.0)
        proposal = rng.uniform(
            env.action_space.low, env.action_space.high
        ).astype(np.float32)
        if rng.random() < 0.18:
            action = proposal
        else:
            action = np.clip(
                0.72 * previous + 0.28 * proposal,
                env.action_space.low,
                env.action_space.high,
            ).astype(np.float32)
        base = env.unwrapped
        states.append(full_state(base))
        qpositions.append(
            np.asarray(base.data.qpos, dtype=np.float64).copy()
        )
        qvelocities.append(
            np.asarray(base.data.qvel, dtype=np.float64).copy()
        )
        actions.append(action.copy())
        env.step(action)
        previous = action
    env.close()
    return StateActionBank(
        states=np.stack(states),
        qpos=np.stack(qpositions),
        qvel=np.stack(qvelocities),
        actions=np.stack(actions),
    )


def transitions_at_scale(
    bank: StateActionBank,
    actuator_scale: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Query paired one-step transitions at one actuator scale."""
    env = gym.make("Pusher-v5")
    env.reset(seed=seed)
    base = env.unwrapped
    nominal_gear = base.model.actuator_gear.copy()
    base.model.actuator_gear[:, 0] = (
        nominal_gear[:, 0] * actuator_scale
    )
    mujoco.mj_forward(base.model, base.data)
    next_states = []
    for qpos, qvel, action in zip(bank.qpos, bank.qvel, bank.actions):
        base.set_state(qpos.copy(), qvel.copy())
        if base.data.ctrl.size:
            base.data.ctrl[:] = 0.0
        base.data.qacc_warmstart[:] = 0.0
        base.data.qfrc_applied[:] = 0.0
        base.data.xfrc_applied[:] = 0.0
        mujoco.mj_forward(base.model, base.data)
        base.step(action)
        next_states.append(full_state(base))
    env.close()
    next_states_array = np.stack(next_states)
    return bank.states, next_states_array - bank.states


def inputs_for_scale(
    bank: StateActionBank,
    scale: float,
) -> np.ndarray:
    context = np.full((len(bank.states), 1), scale, dtype=np.float32)
    return np.concatenate((bank.states, bank.actions, context), axis=1)


def make_dataset(
    bank: StateActionBank,
    scales: list[float],
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    inputs, targets = [], []
    for index, scale in enumerate(scales):
        _, delta = transitions_at_scale(
            bank, scale, seed=seed + 1009 * index
        )
        inputs.append(inputs_for_scale(bank, scale))
        targets.append(delta)
    return np.concatenate(inputs), np.concatenate(targets)


def make_model(
    kind: str,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    grid_size: int,
    prototypes: int,
    grid_range: float,
) -> nn.Module:
    if kind == "mlp":
        return MLPDynamics(input_dim, hidden_dim, output_dim)
    if kind.startswith("kan"):
        return KAN(
            [input_dim, hidden_dim, output_dim],
            grid_size=grid_size,
            spline_order=3,
            grid_range=grid_range,
        )
    if kind.startswith("protokan"):
        model = ProtoKAN(
            [input_dim, hidden_dim, output_dim],
            n_prototypes=prototypes,
            grid_range=grid_range,
        )
        # Ordered neighborhoods must stay geometrically meaningful.
        for layer in model.layers:
            layer.proto_pos.requires_grad_(False)
            layer.log_sigma.requires_grad_(False)
    else:
        model = None
    if model is not None:
        return model
    raise ValueError(kind)


@torch.no_grad()
def prediction_metrics(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    output_mean: torch.Tensor,
    output_std: torch.Tensor,
    dynamic_mask: torch.Tensor,
) -> dict[str, float]:
    prediction = model(inputs)
    normalized_mse = F.mse_loss(prediction, targets)
    raw_prediction = prediction * output_std + output_mean
    raw_target = targets * output_std + output_mean
    error = raw_prediction - raw_target
    dynamic_error = error[:, dynamic_mask]
    target_energy = raw_target[:, dynamic_mask].square().mean().sqrt()
    return {
        "normalized_mse": float(normalized_mse),
        "raw_rmse": float(error.square().mean().sqrt()),
        "dynamic_rmse": float(dynamic_error.square().mean().sqrt()),
        "relative_dynamic_rmse": float(
            dynamic_error.square().mean().sqrt()
            / target_energy.clamp_min(1e-12)
        ),
    }


@torch.no_grad()
def raw_predictions(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    output_mean: torch.Tensor,
    output_std: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    prediction = model(inputs) * output_std + output_mean
    target = targets * output_std + output_mean
    return prediction.cpu().numpy(), target.cpu().numpy()


def train_model(
    kind: str,
    train_inputs: torch.Tensor,
    train_targets: torch.Tensor,
    validation_inputs: torch.Tensor,
    validation_targets: torch.Tensor,
    output_mean: torch.Tensor,
    output_std: torch.Tensor,
    dynamic_mask: torch.Tensor,
    args: argparse.Namespace,
    run_seed: int,
) -> tuple[nn.Module, dict]:
    torch.manual_seed(run_seed)
    model = make_model(
        kind,
        train_inputs.shape[1],
        args.hidden_dim,
        train_targets.shape[1],
        args.grid_size,
        args.prototypes,
        args.grid_range,
    ).to(train_inputs.device)
    if args.freeze_base and kind != "mlp":
        for layer in model.layers:
            with torch.no_grad():
                layer.base_weight.zero_()
            layer.base_weight.requires_grad_(False)
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.learning_rate,
            momentum=args.momentum,
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=0.0,
        )
    generator = torch.Generator(device=train_inputs.device)
    generator.manual_seed(run_seed + 17)
    best_state = None
    best_validation = float("inf")
    started = perf_counter()
    history = []
    source_scale_count = len(args.source_scales)
    root_count = len(train_inputs) // source_scale_count
    for step in range(1, args.train_steps + 1):
        roots = torch.randint(
            0,
            root_count,
            (
                min(
                    max(args.batch_size // source_scale_count, 1),
                    root_count,
                ),
            ),
            generator=generator,
            device=train_inputs.device,
        )
        indices = torch.cat(
            [roots + scale_index * root_count for scale_index in range(
                source_scale_count
            )]
        )
        prediction = model(train_inputs[indices])
        selected_targets = train_targets[indices]
        pointwise_loss = F.mse_loss(prediction, selected_targets)
        grouped_prediction = prediction.reshape(
            source_scale_count, len(roots), -1
        )
        grouped_targets = selected_targets.reshape(
            source_scale_count, len(roots), -1
        )
        counterfactual_loss = F.mse_loss(
            grouped_prediction[-1] - grouped_prediction[0],
            grouped_targets[-1] - grouped_targets[0],
        )
        prediction_loss = (
            pointwise_loss
            + args.counterfactual_weight * counterfactual_loss
        )
        regularizer = prediction_loss.new_zeros(())
        if kind == "kan_pspline":
            regularizer = args.p_spline_weight * p_spline_penalty(model)
        elif kind == "protokan_diffusion_hermite":
            regularizer = (
                args.hermite_weight * protokan_hermite_penalty(model)
            )
        loss = prediction_loss + regularizer
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if kind in (
            "kan_diffusion",
            "protokan_diffusion",
            "protokan_diffusion_hermite",
        ):
            diffuse_input_dimension_gradients_(
                model,
                input_index=train_inputs.shape[1] - 1,
                strength=args.diffusion,
                layer_index=0,
            )
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        if (
            step == 1
            or step % args.log_every == 0
            or step == args.train_steps
        ):
            validation = prediction_metrics(
                model,
                validation_inputs,
                validation_targets,
                output_mean,
                output_std,
                dynamic_mask,
            )
            with torch.no_grad():
                validation_prediction = model(validation_inputs)
                validation_root_count = (
                    len(validation_inputs) // source_scale_count
                )
                validation_grouped_prediction = validation_prediction.reshape(
                    source_scale_count, validation_root_count, -1
                )
                validation_grouped_targets = validation_targets.reshape(
                    source_scale_count, validation_root_count, -1
                )
                validation_counterfactual = F.mse_loss(
                    validation_grouped_prediction[-1]
                    - validation_grouped_prediction[0],
                    validation_grouped_targets[-1]
                    - validation_grouped_targets[0],
                )
            validation_objective = (
                validation["normalized_mse"]
                + args.counterfactual_weight
                * float(validation_counterfactual)
            )
            record = {
                "step": step,
                "train_prediction_mse": float(pointwise_loss.detach()),
                "train_counterfactual_mse": float(
                    counterfactual_loss.detach()
                ),
                "regularizer": float(regularizer.detach()),
                "validation_counterfactual_mse": float(
                    validation_counterfactual
                ),
                "validation_objective": validation_objective,
                **{f"validation_{key}": value for key, value in validation.items()},
            }
            history.append(record)
            print(
                f"TRAIN kind={kind:18s} step={step:04d}/"
                f"{args.train_steps} train={record['train_prediction_mse']:.5f} "
                f"cf={record['train_counterfactual_mse']:.5f} "
                f"val={validation['normalized_mse']:.5f} "
                f"val_cf={float(validation_counterfactual):.5f} "
                f"dyn_rmse={validation['dynamic_rmse']:.6f} "
                f"reg={record['regularizer']:.3e}",
                flush=True,
            )
            if validation_objective < best_validation:
                best_validation = validation_objective
                best_state = copy.deepcopy(model.state_dict())
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "train_seconds": perf_counter() - started,
        "best_source_validation_mse": best_validation,
        "history": history,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=8128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--train-bank", type=int, default=2400)
    parser.add_argument("--validation-bank", type=int, default=600)
    parser.add_argument("--test-bank", type=int, default=800)
    parser.add_argument("--episode-steps", type=int, default=80)
    parser.add_argument(
        "--source-scales", nargs=2, type=float, default=[0.8, 1.0]
    )
    parser.add_argument(
        "--test-scales", nargs="+", type=float, default=[0.65, 0.9, 1.15]
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "mlp",
            "kan",
            "kan_pspline",
            "kan_diffusion",
            "protokan",
            "protokan_diffusion",
            "protokan_diffusion_hermite",
        ],
    )
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--grid-size", type=int, default=13)
    parser.add_argument("--prototypes", type=int, default=8)
    parser.add_argument("--grid-range", type=float, default=4.0)
    parser.add_argument("--train-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=384)
    parser.add_argument("--optimizer", choices=("sgd", "adamw"), default="sgd")
    parser.add_argument("--learning-rate", type=float, default=0.025)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--p-spline-weight", type=float, default=1e-3)
    parser.add_argument("--hermite-weight", type=float, default=1e-3)
    parser.add_argument("--counterfactual-weight", type=float, default=3.0)
    parser.add_argument("--diffusion", type=float, default=0.6)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument("--input-clip", type=float, default=4.0)
    parser.add_argument("--context-min", type=float, default=0.5)
    parser.add_argument("--context-max", type=float, default=1.3)
    parser.add_argument(
        "--freeze-base",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument(
        "--output",
        default="results/spline_coupling_extrapolation_quick.json",
    )
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA unavailable; falling back to CPU", flush=True)
        args.device = "cpu"
    device = torch.device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    print(
        f"CONFIG device={device} source={args.source_scales} "
        f"test={args.test_scales} optimizer={args.optimizer}",
        flush=True,
    )

    data_started = perf_counter()
    train_bank = collect_bank(
        args.train_bank, args.seed + 1, args.episode_steps
    )
    validation_bank = collect_bank(
        args.validation_bank, args.seed + 2, args.episode_steps
    )
    test_bank = collect_bank(
        args.test_bank, args.seed + 3, args.episode_steps
    )
    train_x, train_y = make_dataset(
        train_bank, list(args.source_scales), args.seed + 100
    )
    validation_x, validation_y = make_dataset(
        validation_bank, list(args.source_scales), args.seed + 200
    )
    test_sets = {}
    for index, scale in enumerate(args.test_scales):
        test_x, test_y = make_dataset(
            test_bank, [scale], args.seed + 300 + index
        )
        test_sets[str(scale)] = (test_x, test_y)

    input_mean = train_x.mean(axis=0)
    input_std = train_x.std(axis=0)
    input_std[input_std < 1e-5] = 1.0
    output_mean = train_y.mean(axis=0)
    raw_output_std = train_y.std(axis=0)
    dynamic_mask_np = raw_output_std > 1e-7
    output_std = raw_output_std.copy()
    output_std[~dynamic_mask_np] = 1.0

    def normalize_inputs(values: np.ndarray) -> np.ndarray:
        normalized = (values - input_mean) / input_std
        normalized[:, -1] = (
            2.0
            * (values[:, -1] - args.context_min)
            / (args.context_max - args.context_min)
            - 1.0
        )
        return np.clip(
            normalized, -args.input_clip, args.input_clip
        ).astype(np.float32)

    def normalize_targets(values: np.ndarray) -> np.ndarray:
        return ((values - output_mean) / output_std).astype(np.float32)

    train_inputs = torch.as_tensor(
        normalize_inputs(train_x), device=device
    )
    train_targets = torch.as_tensor(
        normalize_targets(train_y), device=device
    )
    validation_inputs = torch.as_tensor(
        normalize_inputs(validation_x), device=device
    )
    validation_targets = torch.as_tensor(
        normalize_targets(validation_y), device=device
    )
    output_mean_tensor = torch.as_tensor(
        output_mean, dtype=torch.float32, device=device
    )
    output_std_tensor = torch.as_tensor(
        output_std, dtype=torch.float32, device=device
    )
    dynamic_mask = torch.as_tensor(dynamic_mask_np, device=device)
    normalized_tests = {
        scale: (
            torch.as_tensor(normalize_inputs(values[0]), device=device),
            torch.as_tensor(normalize_targets(values[1]), device=device),
        )
        for scale, values in test_sets.items()
    }
    raw_normalized_test = np.concatenate(
        [
            normalize_inputs(values[0])
            for values in test_sets.values()
        ]
    )
    clipped_fraction = float(
        np.mean(np.isclose(np.abs(raw_normalized_test), args.input_clip))
    )
    print(
        f"DATA train={len(train_inputs)} val={len(validation_inputs)} "
        f"test_per_scale={args.test_bank} state_dim={train_bank.states.shape[1]} "
        f"action_dim={train_bank.actions.shape[1]} "
        f"dynamic_outputs={int(dynamic_mask_np.sum())} "
        f"test_clip_fraction={clipped_fraction:.4f} "
        f"seconds={perf_counter()-data_started:.2f}",
        flush=True,
    )

    results = {}
    for model_index, kind in enumerate(args.models):
        if kind.startswith("kan"):
            model_seed = args.seed + 101
        elif kind.startswith("protokan"):
            model_seed = args.seed + 202
        else:
            model_seed = args.seed + 303
        model, training = train_model(
            kind,
            train_inputs,
            train_targets,
            validation_inputs,
            validation_targets,
            output_mean_tensor,
            output_std_tensor,
            dynamic_mask,
            args,
            run_seed=model_seed,
        )
        evaluations = {}
        prediction_cache = {}
        for scale, (inputs, targets) in normalized_tests.items():
            evaluations[scale] = prediction_metrics(
                model,
                inputs,
                targets,
                output_mean_tensor,
                output_std_tensor,
                dynamic_mask,
            )
            prediction_cache[scale] = raw_predictions(
                model,
                inputs,
                targets,
                output_mean_tensor,
                output_std_tensor,
            )
        reference_scale = min(
            evaluations,
            key=lambda value: abs(
                float(value) - float(np.mean(args.source_scales))
            ),
        )
        reference_prediction, reference_target = prediction_cache[
            reference_scale
        ]
        for scale in evaluations:
            prediction, target = prediction_cache[scale]
            predicted_effect = (
                prediction[:, dynamic_mask_np]
                - reference_prediction[:, dynamic_mask_np]
            )
            true_effect = (
                target[:, dynamic_mask_np]
                - reference_target[:, dynamic_mask_np]
            )
            counterfactual_error = predicted_effect - true_effect
            effect_rmse = float(
                np.sqrt(np.mean(counterfactual_error ** 2))
            )
            effect_energy = float(np.sqrt(np.mean(true_effect ** 2)))
            evaluations[scale]["counterfactual_effect_rmse"] = effect_rmse
            evaluations[scale]["relative_counterfactual_effect_rmse"] = (
                effect_rmse / max(effect_energy, 1e-12)
            )
        extrapolation_scales = {
            scale
            for scale in evaluations
            if float(scale) < min(args.source_scales)
            or float(scale) > max(args.source_scales)
        }
        extrapolation_rmse = float(
            np.mean(
                [
                    evaluations[scale]["dynamic_rmse"]
                    for scale in extrapolation_scales
                ]
            )
        )
        extrapolation_effect_rmse = float(
            np.mean(
                [
                    evaluations[scale]["counterfactual_effect_rmse"]
                    for scale in extrapolation_scales
                ]
            )
        )
        extrapolation_effect_relative = float(
            np.mean(
                [
                    evaluations[scale][
                        "relative_counterfactual_effect_rmse"
                    ]
                    for scale in extrapolation_scales
                ]
            )
        )
        results[kind] = {
            "training": training,
            "evaluations": evaluations,
            "mean_extrapolation_dynamic_rmse": extrapolation_rmse,
            "mean_extrapolation_counterfactual_effect_rmse": (
                extrapolation_effect_rmse
            ),
            "mean_extrapolation_relative_counterfactual_effect_rmse": (
                extrapolation_effect_relative
            ),
        }
        print(
            f"RESULT kind={kind:18s} "
            f"external_dyn_rmse={extrapolation_rmse:.6f} "
            f"external_effect_rmse={extrapolation_effect_rmse:.6f} "
            f"relative_effect={extrapolation_effect_relative:.3f} "
            + " ".join(
                f"scale={scale}:"
                f"{metrics['dynamic_rmse']:.6f}"
                for scale, metrics in evaluations.items()
            ),
            flush=True,
        )

    ordinary = results.get("kan")
    coupled = results.get("kan_diffusion")
    proto = results.get("protokan")
    coupled_proto = results.get("protokan_diffusion")
    comparisons = {}
    if ordinary and coupled:
        comparisons["kan_diffusion_vs_kan"] = (
            1.0
            - coupled["mean_extrapolation_counterfactual_effect_rmse"]
            / ordinary["mean_extrapolation_counterfactual_effect_rmse"]
        )
    if proto and coupled_proto:
        comparisons["protokan_diffusion_vs_protokan"] = (
            1.0
            - coupled_proto[
                "mean_extrapolation_counterfactual_effect_rmse"
            ]
            / proto["mean_extrapolation_counterfactual_effect_rmse"]
        )
    print(f"COMPARISONS {comparisons}", flush=True)

    payload = {
        "experiment": "known_context_spline_coupling_extrapolation_gate",
        "scope": (
            "Privileged actuator scale is an input only to isolate the spline "
            "coupling mechanism. This is not the final blind adaptation setup."
        ),
        "configuration": vars(args),
        "data": {
            "markov_state": "qpos(11)+qvel(11)",
            "paired_state_action_bank": True,
            "input_dimension": int(train_inputs.shape[1]),
            "output_dimension": int(train_targets.shape[1]),
            "dynamic_output_dimensions": int(dynamic_mask_np.sum()),
            "test_input_clip_fraction": clipped_fraction,
            "generation_seconds": perf_counter() - data_started,
        },
        "results": results,
        "comparisons": comparisons,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"WROTE {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
