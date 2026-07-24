"""Compare KAN and learned-MLP cognition under identical policy transport."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from cpbn.generic_affine_kan import (
    AffineKANContext,
    CompactInteractionKANDictionary,
    LearnedMLPDictionary,
    fit_affine_kan_context,
)
from cpbn.global_mechanism_kan import GlobalMechanismKANDynamics
from cpbn.policy_mechanism_decoder import PolicyMechanismDecoder
from scripts.diagnose_hopper_global_physics_context import (
    collect_transitions,
    normalized_rmse,
    slice_data,
)
from scripts.prescreen_hopper_physics_shifts import SHIFTS
from scripts.validate_hopper_distilled_policy_mechanisms import evaluate
from scripts.validate_hopper_joint_online_adaptation import FrozenSourcePolicy


def train_mlp_dictionary(
    basis,
    source_data,
    delta_scale,
    args,
    device,
):
    head = nn.Linear(
        (1 + basis.action_dim) * basis.feature_dim,
        source_data["delta"].shape[-1],
        bias=False,
    ).to(device)
    optimizer = torch.optim.Adam(
        (*basis.parameters(), *head.parameters()),
        lr=args.mlp_learning_rate,
    )
    generator = torch.Generator(device=device).manual_seed(args.seed + 991)
    losses = []
    for step in range(1, args.mlp_gradient_steps + 1):
        sample = torch.randint(
            source_data["state"].shape[0],
            (args.batch_size,),
            generator=generator,
            device=device,
        )
        prediction = head(
            basis.context_features(
                source_data["state"][sample],
                source_data["innovation"][sample],
            ),
        )
        loss = (
            (prediction - source_data["delta"][sample])
            / delta_scale
        ).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % args.report_every == 0:
            losses.append({"step": step, "loss": float(loss.detach())})
            print(
                {"stage": "mlp_dictionary", **losses[-1]},
                flush=True,
            )
    return losses


@torch.no_grad()
def build_transport(
    basis,
    source_train,
    source_holdout,
    mechanism_data,
    matched_source_data,
    target_adaptation,
    target_holdout,
    delta_scale,
    args,
):
    source_context = fit_affine_kan_context(
        basis,
        source_train["state"],
        source_train["innovation"],
        source_train["delta"],
        ridge=args.context_ridge,
    )
    mechanism_contexts = []
    for data, reference in zip(mechanism_data, matched_source_data):
        context = fit_affine_kan_context(
            basis,
            data["state"],
            data["innovation"],
            data["delta"],
            ridge=args.context_ridge,
        )
        reference_context = fit_affine_kan_context(
            basis,
            reference["state"],
            reference["innovation"],
            reference["delta"],
            ridge=args.context_ridge,
        )
        reference_shift = (
            reference_context.coefficients - source_context.coefficients
        )
        mechanism_contexts.append(
            AffineKANContext(context.coefficients - reference_shift),
        )
    model = GlobalMechanismKANDynamics.from_contexts(
        source_context,
        mechanism_contexts,
    ).whiten_effects(
        basis,
        source_holdout["state"],
        source_holdout["innovation"],
        delta_scale,
        floor=args.whitening_floor,
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
    )
    latent_scale = training_latents.square().mean(dim=0).sqrt().clamp_min(
        1e-3,
    )
    target_latent = model.infer_latent(
        basis,
        target_adaptation["state"],
        target_adaptation["innovation"],
        target_adaptation["delta"],
        delta_scale,
        ridge=args.latent_ridge,
    )
    training_coordinates = training_latents / latent_scale
    target_coordinate = target_latent / latent_scale
    policy_coefficients = torch.linalg.lstsq(
        training_coordinates.T,
        target_coordinate,
    ).solution
    target_context = model.context(target_latent)
    return {
        "source_context": source_context,
        "model": model,
        "policy_coefficients": policy_coefficients,
        "source_rmse": normalized_rmse(
            source_context, basis, source_holdout, delta_scale,
        ),
        "target_source_rmse": normalized_rmse(
            source_context, basis, target_holdout, delta_scale,
        ),
        "target_adapted_rmse": normalized_rmse(
            target_context, basis, target_holdout, delta_scale,
        ),
        "target_latent": target_latent,
        "training_latents": training_latents,
    }


def main(args):
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print({"stage": "setup", "device": str(device)}, flush=True)
    source_policy = FrozenSourcePolicy(
        args.source_model, args.source_norm, device, args.seed,
    )
    total_source = args.source_transitions + args.holdout_transitions
    source_all = collect_transitions(
        source_policy, "source", total_source, args, device, 71000,
    )
    source_train = slice_data(source_all, 0, args.source_transitions)
    source_holdout = slice_data(
        source_all, args.source_transitions, total_source,
    )
    state_scale = source_train["state"].abs().quantile(
        0.99, dim=0,
    ).clamp_min(0.1)
    delta_scale = source_train["delta"].square().mean(
        dim=0,
    ).sqrt().clamp_min(1e-3)
    action_scale = torch.ones(3, device=device)
    kan_basis = CompactInteractionKANDictionary(
        state_scale,
        action_scale,
        pair_modes=args.pair_modes,
    ).to(device)
    mlp_basis = LearnedMLPDictionary(
        state_scale,
        action_scale,
        feature_dim=kan_basis.feature_dim,
        hidden_dim=args.mlp_hidden_dim,
    ).to(device)
    mlp_history = train_mlp_dictionary(
        mlp_basis,
        source_train,
        delta_scale,
        args,
        device,
    )
    mechanism_data = []
    matched_source_data = []
    for index, name in enumerate(args.mechanism_environments):
        mechanism_data.append(
            collect_transitions(
                source_policy,
                name,
                args.mechanism_transitions,
                args,
                device,
                72000 + 1000 * index,
            ),
        )
        matched_source_data.append(
            collect_transitions(
                source_policy,
                "source",
                args.mechanism_transitions,
                args,
                device,
                76000 + 1000 * index,
            ),
        )
    total_target = args.adaptation_transitions + args.holdout_transitions
    target_all = collect_transitions(
        source_policy,
        args.target,
        total_target,
        args,
        device,
        80000,
    )
    target_adaptation = slice_data(
        target_all, 0, args.adaptation_transitions,
    )
    target_holdout = slice_data(
        target_all, args.adaptation_transitions, total_target,
    )
    checkpoint = torch.load(
        args.decoder_checkpoint,
        map_location=device,
        weights_only=True,
    )
    decoder = PolicyMechanismDecoder(
        source_policy.mean,
        source_policy.variance,
        mechanism_dim=len(args.mechanism_environments),
    ).to(device)
    decoder.load_state_dict(checkpoint["decoder"])
    decoder.eval()

    records = {}
    for name, basis in (("kan", kan_basis), ("mlp", mlp_basis)):
        transport = build_transport(
            basis,
            source_train,
            source_holdout,
            mechanism_data,
            matched_source_data,
            target_adaptation,
            target_holdout,
            delta_scale,
            args,
        )
        policy = evaluate(
            source_policy,
            decoder,
            transport["policy_coefficients"],
            args.target,
            args,
            device,
        )
        records[name] = {
            "dictionary_trainable_parameters": sum(
                parameter.numel() for parameter in basis.parameters()
            ),
            "feature_dim": basis.feature_dim,
            "source_holdout_normalized_rmse": transport["source_rmse"],
            "target_source_normalized_rmse": transport[
                "target_source_rmse"
            ],
            "target_adapted_normalized_rmse": transport[
                "target_adapted_rmse"
            ],
            "target_latent": transport["target_latent"].cpu().tolist(),
            "policy_coefficients": transport[
                "policy_coefficients"
            ].cpu().tolist(),
            "policy": policy,
        }
        print({"stage": "result", "dictionary": name, **records[name]})
    output = {
        "experiment": "HopperCognitionDictionaryPolicyTransfer",
        "seed": args.seed,
        "device": str(device),
        "target": args.target,
        "physical_parameters_visible_to_learner": False,
        "reward_used_by_cognition": False,
        "same_online_linear_context_inference": True,
        "mlp_source_training": mlp_history,
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
    parser.add_argument("--target", default="combo_medium")
    parser.add_argument(
        "--mechanism-environments",
        nargs="+",
        default=("payload_125", "friction_070", "actuator_080"),
    )
    parser.add_argument("--source-transitions", type=int, default=8192)
    parser.add_argument("--mechanism-transitions", type=int, default=1024)
    parser.add_argument("--adaptation-transitions", type=int, default=512)
    parser.add_argument("--holdout-transitions", type=int, default=512)
    parser.add_argument("--exploration-noise", type=float, default=0.2)
    parser.add_argument("--pair-modes", type=int, default=1)
    parser.add_argument("--context-ridge", type=float, default=0.01)
    parser.add_argument("--latent-ridge", type=float, default=0.01)
    parser.add_argument("--whitening-floor", type=float, default=1e-4)
    parser.add_argument("--mlp-hidden-dim", type=int, default=64)
    parser.add_argument("--mlp-gradient-steps", type=int, default=1000)
    parser.add_argument("--mlp-learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--report-every", type=int, default=200)
    parser.add_argument("--evaluation-episodes", type=int, default=5)
    parser.add_argument(
        "--source-model",
        default="results/hopper_source_sb3_ppo_continued_seed1811.zip",
    )
    parser.add_argument(
        "--source-norm",
        default="results/hopper_source_sb3_vecnorm_continued_seed1811.pkl",
    )
    parser.add_argument(
        "--decoder-checkpoint",
        default="results/hopper_policy_mechanism_decoder_fair_combo_medium_seed1811.pt",
    )
    parser.add_argument(
        "--json-out",
        default="results/hopper_cognition_dictionary_transfer_seed1811.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
