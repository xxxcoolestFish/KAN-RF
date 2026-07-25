"""Analyse whether KAN drift coefficients W_t separate known physics shifts.

Fit target cognition at a fixed budget for every named shift, extract the
flattened drift-delta vector, and apply PCA to check whether the low-
dimensional representation distinguishes mass / friction / actuator / combo
changes.
"""

from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

# ── reuse existing pipeline ───────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.diagnose_hopper_pullback_effect import (
    fit_distilled_source_counterfactual_context,
    load_source_twin,
    solve_drift_trust_region,
    solve_spectral_ridge,
)
from scripts.prescreen_hopper_physics_shifts import ENVS, SHIFTS
from scripts.validate_hopper_joint_online_adaptation import (
    FrozenSourcePolicy,
    load_cognition,
)


def extract_raw_drift_delta(context, basis, source_context):
    """Recover drift_delta from the context coefficients."""
    width = basis.feature_dim
    source_blocks = source_context.coefficients.reshape(
        1 + basis.action_dim, width, -1,
    )
    target_blocks = context.coefficients.reshape(
        1 + basis.action_dim, width, -1,
    )
    return (target_blocks[0] - source_blocks[0]).detach().cpu()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=2048)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=1811)
    parser.add_argument("--env", choices=tuple(ENVS), default="hopper")
    parser.add_argument("--smooth-lambda", type=float, default=0.0)
    parser.add_argument("--json-out", default="results/kan_physics_pca.json")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    source_policy = FrozenSourcePolicy(
        "results/hopper_source_sb3_ppo_continued_seed1811.zip",
        "results/hopper_source_sb3_vecnorm_continued_seed1811.pkl",
        device, args.seed, env=args.env,
    )
    source_twin = load_source_twin(
        "results/hopper_source_affine_twin_cloud_seed1811.pt", device,
    )
    basis, source_context, _, delta_scale = load_cognition(
        argparse.Namespace(
            cognition_checkpoint="results/hopper_source_control_sobolev_calibrated_seed1811.pt",
            device=args.device,
        ),
        device,
    )

    # Override args for fitting
    fit_args = argparse.Namespace(
        target="source",
        seed=args.seed,
        env=args.env,
        device=args.device,
        cognition_warmup=args.budget,
        warmup_noise=0.3,
        transform_ridge=10.0,
        drift_ridge=100.0,
        drift_spectral_eta=0.0,
        drift_spectral_beta=1.0,
        drift_spectral_mode="max",
        drift_smooth_lambda=args.smooth_lambda,
        diagonal_transform=False,
    )

    shift_names = [
        "payload_125", "payload_150",
        "friction_070",
        "actuator_080", "actuator_065",
        "combo_mild", "combo_medium",
    ]
    shift_types = {
        "payload_125": "mass", "payload_150": "mass",
        "friction_070": "friction",
        "actuator_080": "actuator", "actuator_065": "actuator",
        "combo_mild": "combo", "combo_medium": "combo",
    }

    drift_vectors = {}
    for name in shift_names:
        fit_args.target = name
        print(f"Fitting {name} at budget {args.budget} ...", flush=True)
        context, _ = fit_distilled_source_counterfactual_context(
            source_policy, basis, source_context, fit_args, device, source_twin,
        )
        drift_delta = extract_raw_drift_delta(context, basis, source_context)
        drift_vectors[name] = {
            "vector": drift_delta.numpy().flatten(),
            "norm": float(drift_delta.norm()),
            "type": shift_types[name],
        }

    # ── PCA ───────────────────────────────────────────────────────────
    matrix = np.stack([d["vector"] for d in drift_vectors.values()])
    matrix_centered = matrix - matrix.mean(axis=0)
    _, singular, vh = np.linalg.svd(matrix_centered, full_matrices=False)
    explained = singular ** 2 / (singular ** 2).sum()

    projected = matrix_centered @ vh.T  # (n_shifts, n_shifts) in PC space

    pca_result = {
        "budget": args.budget,
        "smooth_lambda": args.smooth_lambda,
        "explained_variance_ratio": explained.tolist(),
        "shifts": [],
    }
    print(f"\nPCA explained variance: {explained[:5].tolist()}")
    print(f"First 2 PCs capture {explained[:2].sum():.1%} of variance\n")

    for idx, name in enumerate(shift_names):
        pc1, pc2 = float(projected[idx, 0]), float(projected[idx, 1])
        shift_type = shift_types[name]
        drift_norm = drift_vectors[name]["norm"]
        print(f"  {name:16s}  type={shift_type:10s}  norm={drift_norm:.4f}  "
              f"PC1={pc1:+.4f}  PC2={pc2:+.4f}")
        pca_result["shifts"].append({
            "name": name, "type": shift_type,
            "drift_norm": drift_norm,
            "pc1": pc1, "pc2": pc2,
        })

    Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json_out).write_text(
        json.dumps(pca_result, indent=2) + "\n", encoding="utf-8",
    )
    print(f"\nSaved → {args.json_out}")


if __name__ == "__main__":
    main()
