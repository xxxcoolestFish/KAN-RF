"""Gate a global mechanism latent with a state-dependent ProtoKAN decoder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from cpbn.global_mechanism_kan import GlobalMechanismKANDynamics
from scripts.diagnose_hopper_global_physics_context import (
    collect_transitions,
    fit_context,
    normalized_rmse,
    slice_data,
)
from scripts.validate_hopper_joint_online_adaptation import (
    FrozenSourcePolicy,
    load_cognition,
)


def cosine_matrix(mechanisms):
    flat = mechanisms.flatten(start_dim=1)
    normalized = flat / flat.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return (normalized @ normalized.T).cpu().tolist()


def evaluate_prefix(
    model,
    basis,
    source_context,
    full_context,
    adaptation,
    holdout,
    delta_scale,
    prefix,
    ridge,
):
    latent = model.infer_latent(
        basis,
        adaptation["state"][:prefix],
        adaptation["innovation"][:prefix],
        adaptation["delta"][:prefix],
        delta_scale,
        ridge=ridge,
    )
    context = model.context(latent)
    source_rmse = normalized_rmse(
        source_context, basis, holdout, delta_scale,
    )
    full_rmse = normalized_rmse(
        full_context, basis, holdout, delta_scale,
    )
    latent_rmse = normalized_rmse(
        context, basis, holdout, delta_scale,
    )
    full_gain = source_rmse - full_rmse
    retained = (
        (source_rmse - latent_rmse) / full_gain
        if full_gain > 1e-8
        else 0.0
    )
    return {
        "latent": latent.cpu().tolist(),
        "source_normalized_rmse": source_rmse,
        "full_normalized_rmse": full_rmse,
        "latent_normalized_rmse": latent_rmse,
        "full_adaptation_gain": full_gain,
        "retained_adaptation_gain": retained,
    }


def main(args):
    torch.manual_seed(args.seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu",
    )
    print(
        {
            "stage": "setup",
            "device": str(device),
            "mechanism_environments": args.mechanism_environments,
            "heldout_environments": args.heldout_environments,
            "adaptation_prefixes": args.adaptation_prefixes,
        },
        flush=True,
    )
    source_policy = FrozenSourcePolicy(
        args.source_model, args.source_norm, device, args.seed,
    )
    basis, source_context, estimator, delta_scale = load_cognition(
        args, device,
    )

    mechanism_contexts = []
    mechanism_data = []
    mechanism_records = {}
    for index, name in enumerate(args.mechanism_environments):
        data = collect_transitions(
            source_policy,
            name,
            args.mechanism_transitions,
            args,
            device,
            81000 + 1000 * index,
        )
        context = fit_context(estimator, data)
        reference_norm = 0.0
        if args.matched_source_centering:
            reference_data = collect_transitions(
                source_policy,
                "source",
                args.mechanism_transitions,
                args,
                device,
                85000 + 1000 * index,
            )
            reference_context = fit_context(estimator, reference_data)
            reference_shift = (
                reference_context.coefficients
                - source_context.coefficients
            )
            reference_norm = float(reference_shift.norm())
            context = type(context)(
                context.coefficients - reference_shift,
            )
        mechanism_contexts.append(context)
        mechanism_data.append(data)
        mechanism_records[name] = {
            "coefficient_difference_norm": float(
                (
                    context.coefficients
                    - source_context.coefficients
                ).norm()
            ),
            "matched_source_update_norm": reference_norm,
            "episodes": data["episodes"],
        }
        print(
            {
                "stage": "mechanism",
                "environment": name,
                **mechanism_records[name],
            },
            flush=True,
        )

    model = GlobalMechanismKANDynamics.from_contexts(
        source_context, mechanism_contexts,
    )
    if args.effect_whitening_transitions > 0:
        calibration = collect_transitions(
            source_policy,
            "source",
            args.effect_whitening_transitions,
            args,
            device,
            89000,
        )
        model = model.whiten_effects(
            basis,
            calibration["state"],
            calibration["innovation"],
            delta_scale,
            floor=args.effect_whitening_floor,
        )
    training_latents = torch.stack(
        [
            model.infer_latent(
                basis,
                data["state"],
                data["innovation"],
                data["delta"],
                delta_scale,
                ridge=args.latent_ridge,
            )
            for data in mechanism_data
        ],
        dim=0,
    )
    latent_scale = torch.sqrt(
        training_latents.square().mean(dim=0),
    ).clamp_min(1e-3)
    if args.checkpoint_out:
        Path(args.checkpoint_out).parent.mkdir(
            parents=True, exist_ok=True,
        )
        torch.save(
            {
                "source_coefficients": (
                    source_context.coefficients.detach().cpu()
                ),
                "mechanisms": model.mechanisms.detach().cpu(),
                "latent_scale": latent_scale.detach().cpu(),
                "training_latents": training_latents.detach().cpu(),
                "mechanism_environments": tuple(
                    args.mechanism_environments
                ),
                "physical_parameters_visible_to_learner": False,
            },
            args.checkpoint_out,
        )
    heldout_records = {}
    maximum_prefix = max(args.adaptation_prefixes)
    for index, name in enumerate(args.heldout_environments):
        total = maximum_prefix + args.holdout_transitions
        data = collect_transitions(
            source_policy,
            name,
            total,
            args,
            device,
            91000 + 1000 * index,
        )
        adaptation = slice_data(data, 0, maximum_prefix)
        holdout = slice_data(data, maximum_prefix, total)
        full_context = fit_context(estimator, adaptation)
        prefix_records = {
            str(prefix): evaluate_prefix(
                model,
                basis,
                source_context,
                full_context,
                adaptation,
                holdout,
                delta_scale,
                prefix,
                args.latent_ridge,
            )
            for prefix in args.adaptation_prefixes
        }
        half = maximum_prefix // 2
        first = model.infer_latent(
            basis,
            adaptation["state"][:half],
            adaptation["innovation"][:half],
            adaptation["delta"][:half],
            delta_scale,
            ridge=args.latent_ridge,
        )
        second = model.infer_latent(
            basis,
            adaptation["state"][half:],
            adaptation["innovation"][half:],
            adaptation["delta"][half:],
            delta_scale,
            ridge=args.latent_ridge,
        )
        stability = float(
            torch.nn.functional.cosine_similarity(
                first, second, dim=0,
            )
        )
        heldout_records[name] = {
            "episodes": data["episodes"],
            "split_half_latent_cosine": stability,
            "prefixes": prefix_records,
        }
        print(
            {
                "stage": "heldout",
                "environment": name,
                "split_half_latent_cosine": stability,
                "retained_gain": {
                    key: value["retained_adaptation_gain"]
                    for key, value in prefix_records.items()
                },
                "latent": prefix_records[str(maximum_prefix)]["latent"],
            },
            flush=True,
        )

    final_key = str(maximum_prefix)
    final_retained = [
        record["prefixes"][final_key]["retained_adaptation_gain"]
        for record in heldout_records.values()
        if record["prefixes"][final_key]["full_adaptation_gain"] > 0.0
    ]
    stable = [
        record["split_half_latent_cosine"]
        for record in heldout_records.values()
    ]
    gate_passed = bool(
        final_retained
        and min(final_retained) >= args.minimum_retained_gain
        and min(stable) >= args.minimum_split_half_cosine
    )
    output = {
        "experiment": "HopperGlobalMechanismLatentGate",
        "seed": args.seed,
        "device": str(device),
        "physical_parameters_visible_to_learner": False,
        "environment_names_used_as_model_inputs": False,
        "trajectory_grouping_used": True,
        "model": (
            "global latent z with state-action-dependent "
            "CompactInteractionKAN decoder"
        ),
        "mechanisms": mechanism_records,
        "mechanism_cosine_matrix": cosine_matrix(model.mechanisms),
        "training_latents": training_latents.cpu().tolist(),
        "latent_scale": latent_scale.cpu().tolist(),
        "heldout": heldout_records,
        "gate_passed": gate_passed,
        "config": vars(args),
    }
    Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json_out).write_text(
        json.dumps(output, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        {
            "stage": "summary",
            "final_retained_gain": final_retained,
            "split_half_latent_cosine": stable,
            "gate_passed": gate_passed,
        },
        flush=True,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1811)
    parser.add_argument(
        "--mechanism-environments",
        nargs="+",
        default=("payload_125", "friction_070", "actuator_080"),
    )
    parser.add_argument(
        "--heldout-environments",
        nargs="+",
        default=("payload_150", "combo_mild", "combo_medium"),
    )
    parser.add_argument("--mechanism-transitions", type=int, default=1024)
    parser.add_argument(
        "--adaptation-prefixes",
        nargs="+",
        type=int,
        default=(64, 128, 256, 512),
    )
    parser.add_argument("--holdout-transitions", type=int, default=512)
    parser.add_argument("--exploration-noise", type=float, default=0.2)
    parser.add_argument("--latent-ridge", type=float, default=1e-2)
    parser.add_argument(
        "--effect-whitening-transitions", type=int, default=512,
    )
    parser.add_argument(
        "--effect-whitening-floor", type=float, default=1e-4,
    )
    parser.add_argument(
        "--matched-source-centering",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--minimum-retained-gain", type=float, default=0.7)
    parser.add_argument("--minimum-split-half-cosine", type=float, default=0.8)
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
        default="results/hopper_source_centered_protokan_seed1811.pt",
    )
    parser.add_argument(
        "--json-out",
        default="results/hopper_global_mechanism_latent_seed1811.json",
    )
    parser.add_argument(
        "--checkpoint-out",
        default="results/hopper_global_mechanism_latent_seed1811.pt",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
