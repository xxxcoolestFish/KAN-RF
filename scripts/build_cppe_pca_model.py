"""Build and save complete CPPE PCA model with PC ranges.

One-shot script: fits all 7 canonical physics shifts, runs PCA, computes
PC ranges from observed projections, and saves everything to a single npz.

Usage: python scripts/build_cppe_pca_model.py --budget 512 --k 5
"""

from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.diagnose_hopper_pullback_effect import (
    fit_distilled_source_counterfactual_context,
    load_source_twin,
)
from scripts.prescreen_hopper_physics_shifts import ENVS, SHIFTS
from scripts.validate_hopper_joint_online_adaptation import (
    FrozenSourcePolicy,
    load_cognition,
)


def extract_drift_delta(context, basis, source_context):
    """Extract flattened drift delta from fitted context."""
    width = basis.feature_dim
    source_blocks = source_context.coefficients.reshape(
        1 + basis.action_dim, width, -1,
    )
    target_blocks = context.coefficients.reshape(
        1 + basis.action_dim, width, -1,
    )
    return (target_blocks[0] - source_blocks[0]).detach().cpu().numpy().flatten()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=512)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=1811)
    parser.add_argument("--pca-out", default="results/cppe_pca_model.npz")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    print(f"Building PCA model: budget={args.budget}, k={args.k}, device={device}")

    # ── Load components ─────────────────────────────────────────────────
    print("Loading source components ...", flush=True)
    source_policy = FrozenSourcePolicy(
        "results/hopper_source_sb3_ppo_continued_seed1811.zip",
        "results/hopper_source_sb3_vecnorm_continued_seed1811.pkl",
        device, args.seed, env="hopper",
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

    fit_args = argparse.Namespace(
        target="source", seed=args.seed, env="hopper",
        device=args.device,
        cognition_warmup=args.budget,
        warmup_noise=0.3,
        transform_ridge=10.0,
        drift_ridge=100.0,
        drift_spectral_eta=0.0,
        drift_spectral_beta=1.0,
        drift_spectral_mode="max",
        drift_smooth_lambda=0.0,
        diagonal_transform=False,
    )

    # ── Fit all 7 shifts ─────────────────────────────────────────────────
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
    drift_norms = {}
    for name in shift_names:
        print(f"  Fitting {name} ...", flush=True)
        fit_args.target = name
        context, _ = fit_distilled_source_counterfactual_context(
            source_policy, basis, source_context, fit_args, device, source_twin,
        )
        dv = extract_drift_delta(context, basis, source_context)
        drift_vectors[name] = dv
        drift_norms[name] = float(np.linalg.norm(dv))

    # ── PCA ─────────────────────────────────────────────────────────────
    matrix = np.stack(list(drift_vectors.values()))  # (7, drift_dim)
    mean = matrix.mean(axis=0)
    centered = matrix - mean
    U, s, Vh = np.linalg.svd(centered, full_matrices=False)
    explained = s ** 2 / (s ** 2).sum()

    k = min(args.k, len(s))
    z_values = centered @ Vh[:k, :].T  # (7, k)

    # PC ranges from observed projections
    pc_mins = z_values.min(axis=0)
    pc_maxs = z_values.max(axis=0)

    # Reconstruction quality
    reconstructed = mean + z_values @ Vh[:k, :]
    recon_errors = {}
    for i, name in enumerate(shift_names):
        err = float(np.linalg.norm(matrix[i] - reconstructed[i]) /
                    max(np.linalg.norm(matrix[i]), 1e-10))
        recon_errors[name] = err

    # ── Report ──────────────────────────────────────────────────────────
    print(f"\nPCA: {matrix.shape[1]} dims, {len(shift_names)} samples")
    print(f"k={k}:  cumulated variance = {explained[:k].sum():.1%}")
    print(f"        mean reconstruction error = {np.mean(list(recon_errors.values())):.3f}")
    print(f"\nPC ranges:")
    for i in range(k):
        print(f"  PC{i+1}: [{pc_mins[i]:+.4f}, {pc_maxs[i]:+.4f}]  "
              f"({explained[i]:.1%} variance)")
    print(f"\nPer-shift z values (first 2 PCs):")
    for i, name in enumerate(shift_names):
        print(f"  {name:16s} type={shift_types[name]:10s}  "
              f"norm={drift_norms[name]:.4f}  "
              f"z=({z_values[i,0]:+.4f}, {z_values[i,1]:+.4f})  "
              f"recon_err={recon_errors[name]:.3f}")

    # ── Save ────────────────────────────────────────────────────────────
    Path(args.pca_out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.pca_out,
        mean=mean,
        Vh=Vh,
        singular=s,
        explained=explained,
        drift_dim=matrix.shape[1],
        k=k,
        pc_mins=pc_mins,
        pc_maxs=pc_maxs,
        z_values=z_values,
        shift_names=np.array(shift_names),
        shift_types=np.array([shift_types[n] for n in shift_names]),
    )
    print(f"\nSaved -> {args.pca_out}")

    # Also save a readable JSON summary
    json_out = args.pca_out.replace(".npz", "_summary.json")
    summary = {
        "budget": args.budget, "k": k,
        "drift_dim": int(matrix.shape[1]),
        "explained_variance": explained[:k].tolist(),
        "cumulated_variance": float(explained[:k].sum()),
        "pc_ranges": [
            {"pc": i+1, "min": float(pc_mins[i]), "max": float(pc_maxs[i]),
             "explained": float(explained[i])}
            for i in range(k)
        ],
        "shifts": [
            {"name": name, "type": shift_types[name],
             "drift_norm": drift_norms[name],
             "z": z_values[i, :k].tolist(),
             "recon_error": recon_errors[name]}
            for i, name in enumerate(shift_names)
        ],
    }
    Path(json_out).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Saved -> {json_out}")


if __name__ == "__main__":
    main()
