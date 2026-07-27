"""One-shot gate for aligned action-function modulation and latent adaptation."""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from kanrf import ActionModulatedProtoKAN, ProtoKAN
from scripts.quick_validate_spline_coupling_extrapolation import (
    collect_bank,
    make_dataset,
)


def make_plain_protokan(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    prototypes: int,
    grid_range: float,
) -> ProtoKAN:
    model = ProtoKAN(
        [input_dim, hidden_dim, output_dim],
        n_prototypes=prototypes,
        grid_range=grid_range,
    )
    for layer in model.layers:
        layer.proto_pos.requires_grad_(False)
        layer.log_sigma.requires_grad_(False)
    return model


def paired_indices(
    root_count: int,
    batch_roots: int,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    roots = torch.randint(
        root_count,
        (min(batch_roots, root_count),),
        generator=generator,
        device=device,
    )
    return roots, torch.cat((roots, roots + root_count))


def train_sources(
    pooled: ProtoKAN,
    modulated: ActionModulatedProtoKAN,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    validation_inputs: torch.Tensor,
    validation_targets: torch.Tensor,
    steps: int,
    batch_size: int,
    learning_rate: float,
    counterfactual_weight: float,
    seed: int,
    log_every: int,
) -> dict:
    pooled_optimizer = torch.optim.AdamW(
        pooled.parameters(), lr=learning_rate
    )
    modulated_optimizer = torch.optim.AdamW(
        modulated.parameters(), lr=learning_rate
    )
    generator = torch.Generator(device=inputs.device)
    generator.manual_seed(seed + 91)
    root_count = len(inputs) // 2
    source_latents = torch.tensor(
        [[-1.0], [1.0]], device=inputs.device
    )
    history = []
    best = {"pooled": (float("inf"), None), "modulated": (float("inf"), None)}

    for step in range(1, steps + 1):
        roots, indices = paired_indices(
            root_count, batch_size // 2, generator, inputs.device
        )
        batch_inputs = inputs[indices]
        batch_targets = targets[indices]

        pooled_prediction = pooled(batch_inputs)
        pooled_loss = F.mse_loss(pooled_prediction, batch_targets)
        pooled_optimizer.zero_grad(set_to_none=True)
        pooled_loss.backward()
        pooled_optimizer.step()

        latent_batch = source_latents[:, None, :].expand(
            2, len(roots), 1
        ).reshape(-1, 1)
        modulated_prediction = modulated(batch_inputs, latent_batch)
        grouped_prediction = modulated_prediction.reshape(2, len(roots), -1)
        grouped_target = batch_targets.reshape(2, len(roots), -1)
        pointwise = F.mse_loss(modulated_prediction, batch_targets)
        counterfactual = F.mse_loss(
            grouped_prediction[1] - grouped_prediction[0],
            grouped_target[1] - grouped_target[0],
        )
        modulated_loss = pointwise + counterfactual_weight * counterfactual
        modulated_optimizer.zero_grad(set_to_none=True)
        modulated_loss.backward()
        modulated_optimizer.step()

        if step == 1 or step % log_every == 0 or step == steps:
            with torch.no_grad():
                validation_roots = len(validation_inputs) // 2
                pooled_val = F.mse_loss(
                    pooled(validation_inputs), validation_targets
                )
                validation_latents = source_latents[:, None, :].expand(
                    2, validation_roots, 1
                ).reshape(-1, 1)
                modulated_val_prediction = modulated(
                    validation_inputs, validation_latents
                )
                modulated_val_pointwise = F.mse_loss(
                    modulated_val_prediction, validation_targets
                )
                grouped_val_prediction = modulated_val_prediction.reshape(
                    2, validation_roots, -1
                )
                grouped_val_target = validation_targets.reshape(
                    2, validation_roots, -1
                )
                modulated_val_cf = F.mse_loss(
                    grouped_val_prediction[1] - grouped_val_prediction[0],
                    grouped_val_target[1] - grouped_val_target[0],
                )
                modulated_val = (
                    modulated_val_pointwise
                    + counterfactual_weight * modulated_val_cf
                )
            values = {
                "step": step,
                "pooled_validation": float(pooled_val),
                "modulated_validation": float(modulated_val),
                "modulated_validation_pointwise": float(
                    modulated_val_pointwise
                ),
                "modulated_validation_counterfactual": float(
                    modulated_val_cf
                ),
            }
            history.append(values)
            print(
                "TRAIN "
                f"step={step:04d}/{steps} "
                f"pooled_val={values['pooled_validation']:.5f} "
                f"mod_val={values['modulated_validation_pointwise']:.5f} "
                f"mod_cf={values['modulated_validation_counterfactual']:.5f}",
                flush=True,
            )
            if float(pooled_val) < best["pooled"][0]:
                best["pooled"] = (
                    float(pooled_val),
                    copy.deepcopy(pooled.state_dict()),
                )
            if float(modulated_val) < best["modulated"][0]:
                best["modulated"] = (
                    float(modulated_val),
                    copy.deepcopy(modulated.state_dict()),
                )

    pooled.load_state_dict(best["pooled"][1])
    modulated.load_state_dict(best["modulated"][1])
    return {"history": history}


def fit_latent(
    model: ActionModulatedProtoKAN,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    steps: int,
    learning_rate: float,
) -> tuple[torch.Tensor, list[float]]:
    latent = nn.Parameter(torch.zeros(1, model.latent_dim, device=inputs.device))
    optimizer = torch.optim.Adam([latent], lr=learning_rate)
    history = []
    for step in range(steps):
        prediction = model(inputs, latent)
        loss = F.mse_loss(prediction, targets) + 1e-4 * latent.square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step in (0, steps // 2, steps - 1):
            history.append(float(loss.detach()))
    return latent.detach(), history


def full_finetune(
    source: ProtoKAN,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    steps: int,
    learning_rate: float,
) -> tuple[ProtoKAN, list[float]]:
    model = copy.deepcopy(source)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    history = []
    for step in range(steps):
        loss = F.mse_loss(model(inputs), targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step in (0, steps // 2, steps - 1):
            history.append(float(loss.detach()))
    return model, history


@torch.no_grad()
def raw_prediction(
    model,
    inputs: torch.Tensor,
    output_mean: torch.Tensor,
    output_std: torch.Tensor,
    latent: torch.Tensor | None = None,
) -> np.ndarray:
    normalized = model(inputs) if latent is None else model(inputs, latent)
    return (normalized * output_std + output_mean).cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=9137)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--train-bank", type=int, default=1600)
    parser.add_argument("--validation-bank", type=int, default=400)
    parser.add_argument("--support-bank", type=int, default=64)
    parser.add_argument("--test-bank", type=int, default=500)
    parser.add_argument("--source-scales", nargs=2, type=float, default=[0.8, 1.0])
    parser.add_argument("--test-scales", nargs="+", type=float,
                        default=[0.65, 0.9, 1.15])
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--prototypes", type=int, default=8)
    parser.add_argument("--grid-range", type=float, default=4.0)
    parser.add_argument("--train-steps", type=int, default=600)
    parser.add_argument("--adapt-steps", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=384)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--latent-learning-rate", type=float, default=5e-2)
    parser.add_argument("--finetune-learning-rate", type=float, default=1e-3)
    parser.add_argument("--counterfactual-weight", type=float, default=100.0)
    parser.add_argument("--log-every", type=int, default=150)
    parser.add_argument(
        "--output",
        default="results/_function_modulated_protokan_gate.json",
    )
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    print(f"CONFIG device={device} seed={args.seed}", flush=True)

    train_bank = collect_bank(args.train_bank, args.seed + 1, 80)
    validation_bank = collect_bank(args.validation_bank, args.seed + 2, 80)
    support_bank = collect_bank(args.support_bank, args.seed + 3, 80)
    test_bank = collect_bank(args.test_bank, args.seed + 4, 80)
    train_x, train_y = make_dataset(
        train_bank, list(args.source_scales), args.seed + 100
    )
    validation_x, validation_y = make_dataset(
        validation_bank, list(args.source_scales), args.seed + 200
    )
    support_sets = {}
    test_sets = {}
    for index, scale in enumerate(args.test_scales):
        support_sets[str(scale)] = make_dataset(
            support_bank, [scale], args.seed + 300 + index
        )
        test_sets[str(scale)] = make_dataset(
            test_bank, [scale], args.seed + 400 + index
        )

    # Remove privileged scale: source identity enters only through fixed
    # anonymous latents; target latents are inferred from support transitions.
    train_x = train_x[:, :-1]
    validation_x = validation_x[:, :-1]
    support_sets = {
        key: (values[0][:, :-1], values[1])
        for key, values in support_sets.items()
    }
    test_sets = {
        key: (values[0][:, :-1], values[1])
        for key, values in test_sets.items()
    }
    input_mean = train_x.mean(0)
    input_std = train_x.std(0)
    input_std[input_std < 1e-5] = 1.0
    output_mean_np = train_y.mean(0)
    output_std_np = train_y.std(0)
    dynamic_mask = output_std_np > 1e-7
    output_std_np[~dynamic_mask] = 1.0

    def tensors(values):
        x, y = values
        normalized_x = np.clip(
            (x - input_mean) / input_std, -4.0, 4.0
        ).astype(np.float32)
        normalized_y = (
            (y - output_mean_np) / output_std_np
        ).astype(np.float32)
        return (
            torch.as_tensor(normalized_x, device=device),
            torch.as_tensor(normalized_y, device=device),
        )

    train_inputs, train_targets = tensors((train_x, train_y))
    validation_inputs, validation_targets = tensors(
        (validation_x, validation_y)
    )
    support_tensors = {key: tensors(value) for key, value in support_sets.items()}
    test_tensors = {key: tensors(value) for key, value in test_sets.items()}
    output_mean = torch.as_tensor(
        output_mean_np, dtype=torch.float32, device=device
    )
    output_std = torch.as_tensor(
        output_std_np, dtype=torch.float32, device=device
    )

    input_dim = train_inputs.shape[1]
    output_dim = train_targets.shape[1]
    action_dim = train_bank.actions.shape[1]
    action_start = input_dim - action_dim
    torch.manual_seed(args.seed + 10)
    pooled = make_plain_protokan(
        input_dim, args.hidden_dim, output_dim,
        args.prototypes, args.grid_range,
    ).to(device)
    torch.manual_seed(args.seed + 10)
    modulated = ActionModulatedProtoKAN(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=output_dim,
        action_start=action_start,
        action_dim=action_dim,
        latent_dim=1,
        n_prototypes=args.prototypes,
        grid_range=args.grid_range,
    ).to(device)
    # Start from exactly the same shared backbone.
    modulated.backbone.load_state_dict(pooled.state_dict())
    training = train_sources(
        pooled, modulated, train_inputs, train_targets,
        validation_inputs, validation_targets,
        args.train_steps, args.batch_size, args.learning_rate,
        args.counterfactual_weight, args.seed, args.log_every,
    )

    methods = {"pooled": {}, "full_finetune": {}, "latent_adapt": {}}
    adaptation = {}
    for scale in map(str, args.test_scales):
        support_inputs, support_targets = support_tensors[scale]
        test_inputs, _ = test_tensors[scale]
        latent, latent_history = fit_latent(
            modulated, support_inputs, support_targets,
            args.adapt_steps, args.latent_learning_rate,
        )
        tuned, tune_history = full_finetune(
            pooled, support_inputs, support_targets,
            args.adapt_steps, args.finetune_learning_rate,
        )
        methods["pooled"][scale] = raw_prediction(
            pooled, test_inputs, output_mean, output_std
        )
        methods["full_finetune"][scale] = raw_prediction(
            tuned, test_inputs, output_mean, output_std
        )
        methods["latent_adapt"][scale] = raw_prediction(
            modulated, test_inputs, output_mean, output_std, latent
        )
        adaptation[scale] = {
            "latent": latent.cpu().numpy().tolist(),
            "latent_loss_trace": latent_history,
            "full_finetune_loss_trace": tune_history,
        }
        print(
            f"ADAPT scale={scale} latent={float(latent.item()):+.4f} "
            f"latent_loss={latent_history[-1]:.5f} "
            f"full_loss={tune_history[-1]:.5f}",
            flush=True,
        )

    raw_targets = {
        scale: values[1] for scale, values in test_sets.items()
    }
    reference = min(
        map(str, args.test_scales),
        key=lambda value: abs(
            float(value) - float(np.mean(args.source_scales))
        ),
    )
    results = {}
    external = [
        str(scale) for scale in args.test_scales
        if scale < min(args.source_scales) or scale > max(args.source_scales)
    ]
    for method, predictions in methods.items():
        per_scale = {}
        for scale in map(str, args.test_scales):
            error = (
                predictions[scale][:, dynamic_mask]
                - raw_targets[scale][:, dynamic_mask]
            )
            predicted_effect = (
                predictions[scale][:, dynamic_mask]
                - predictions[reference][:, dynamic_mask]
            )
            true_effect = (
                raw_targets[scale][:, dynamic_mask]
                - raw_targets[reference][:, dynamic_mask]
            )
            effect_error = predicted_effect - true_effect
            per_scale[scale] = {
                "dynamic_rmse": float(np.sqrt(np.mean(error ** 2))),
                "counterfactual_effect_rmse": float(
                    np.sqrt(np.mean(effect_error ** 2))
                ),
            }
        external_effect = float(np.mean([
            per_scale[scale]["counterfactual_effect_rmse"]
            for scale in external
        ]))
        external_dynamic = float(np.mean([
            per_scale[scale]["dynamic_rmse"] for scale in external
        ]))
        results[method] = {
            "per_scale": per_scale,
            "mean_external_dynamic_rmse": external_dynamic,
            "mean_external_counterfactual_effect_rmse": external_effect,
        }
        print(
            f"RESULT method={method:13s} "
            f"external_dyn={external_dynamic:.6f} "
            f"external_effect={external_effect:.6f}",
            flush=True,
        )

    latent_error = results["latent_adapt"][
        "mean_external_counterfactual_effect_rmse"
    ]
    comparisons = {
        "latent_vs_pooled_effect_improvement": (
            1.0 - latent_error
            / results["pooled"]["mean_external_counterfactual_effect_rmse"]
        ),
        "latent_vs_full_finetune_effect_improvement": (
            1.0 - latent_error
            / results["full_finetune"][
                "mean_external_counterfactual_effect_rmse"
            ]
        ),
    }
    print(f"COMPARISONS {comparisons}", flush=True)
    payload = {
        "experiment": "function_aligned_action_modulation_gate",
        "scope": (
            "Two anonymous source identities; target uses 64 reward-free "
            "transitions. One fixed protocol, no hyperparameter sweep."
        ),
        "configuration": vars(args),
        "training": training,
        "adaptation": adaptation,
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
