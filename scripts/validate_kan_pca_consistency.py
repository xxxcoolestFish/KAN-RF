"""Validate that PCA coordinates of W_t are consistent and physically meaningful.

Test 1: same shift, different seeds → do PCA coords cluster?
Test 2: continuous parameter sweep (mass 1.0→1.5) → smooth PCA trajectory?
"""

from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.analyze_kan_physics_representation import extract_raw_drift_delta
from scripts.diagnose_hopper_pullback_effect import (
    fit_distilled_source_counterfactual_context,
    load_source_twin,
)
from scripts.prescreen_hopper_physics_shifts import ENVS, make_shifted_env
from scripts.validate_hopper_joint_online_adaptation import (
    FrozenSourcePolicy,
    load_cognition,
)


def fit_one(source_policy, basis, source_context, source_twin,
            fit_args, device):
    """Fit W_t for a single configuration, return flattened drift vector."""
    context, _ = fit_distilled_source_counterfactual_context(
        source_policy, basis, source_context, fit_args, device, source_twin,
    )
    drift_delta = extract_raw_drift_delta(context, basis, source_context)
    return drift_delta.numpy().flatten()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=2048)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--base-seed", type=int, default=1811)
    parser.add_argument("--env", choices=tuple(ENVS), default="hopper")
    parser.add_argument("--smooth-lambda", type=float, default=0.0)
    parser.add_argument("--json-out", default="results/kan_pca_consistency.json")
    args = parser.parse_args()

    torch.manual_seed(args.base_seed)
    device = torch.device(args.device)

    # ── shared components ─────────────────────────────────────────────
    source_policy = FrozenSourcePolicy(
        "results/hopper_source_sb3_ppo_continued_seed1811.zip",
        "results/hopper_source_sb3_vecnorm_continued_seed1811.pkl",
        device, args.base_seed, env=args.env,
    )
    source_twin = load_source_twin(
        "results/hopper_source_affine_twin_cloud_seed1811.pt", device,
    )
    basis, source_context, _, _ = load_cognition(
        argparse.Namespace(
            cognition_checkpoint="results/hopper_source_control_sobolev_calibrated_seed1811.pt",
            device=args.device,
        ),
        device,
    )

    fit_args = argparse.Namespace(
        target="source", seed=args.base_seed, env=args.env,
        device=args.device, cognition_warmup=args.budget,
        warmup_noise=0.3, transform_ridge=10.0, drift_ridge=100.0,
        drift_spectral_eta=0.0, drift_spectral_beta=1.0,
        drift_spectral_mode="max", drift_smooth_lambda=args.smooth_lambda,
        diagonal_transform=False,
    )

    # ── reference PCA basis (from the 7 canonical shifts) ─────────────
    ref_shifts = [
        "payload_125", "payload_150",
        "friction_070",
        "actuator_080", "actuator_065",
        "combo_mild", "combo_medium",
    ]
    ref_vectors = []
    for name in ref_shifts:
        fit_args.target = name
        print(f"Reference: {name} (seed {args.base_seed})", flush=True)
        vec = fit_one(source_policy, basis, source_context, source_twin,
                      fit_args, device)
        ref_vectors.append(vec)
    ref_matrix = np.stack(ref_vectors)
    ref_mean = ref_matrix.mean(axis=0)
    _, _, vh = np.linalg.svd(ref_matrix - ref_mean, full_matrices=False)
    P = vh[:4]  # top-4 PC projection (6061 → 4)

    results = {"budget": args.budget, "smooth_lambda": args.smooth_lambda}

    # ── Test 1: same shift, different seeds ───────────────────────────
    print("\n=== Test 1: payload_125, friction_070 across 5 seeds ===")
    test1 = {}
    for shift in ["payload_125", "friction_070"]:
        coords = []
        for s in [1811, 1813, 1817, 1819, 1823]:
            fit_args.target = shift
            fit_args.seed = s
            print(f"  {shift} seed {s}", flush=True)
            vec = fit_one(source_policy, basis, source_context, source_twin,
                          fit_args, device)
            z = (vec - ref_mean) @ P.T
            coords.append(z[:4].tolist())
        test1[shift] = coords
        print(f"  → z spread (±std): PC1={np.std([c[0] for c in coords]):.4f}, "
              f"PC2={np.std([c[1] for c in coords]):.4f}, "
              f"PC3={np.std([c[2] for c in coords]):.4f}")
    results["test1_same_shift_different_seeds"] = test1

    # ── Test 2: continuous mass sweep ─────────────────────────────────
    print("\n=== Test 2: mass sweep 1.00 → 1.50 ===")
    test2 = []
    for mass_scale in np.linspace(1.0, 1.5, 11):
        fit_args.target = None  # custom shift
        # We pass the shift dict through args.target normally, but for
        # custom shifts we directly create a named shift entry.
        # Re-use the existing pipeline by temporarily patching SHIFTS.
        from scripts.prescreen_hopper_physics_shifts import SHIFTS
        custom_name = f"_mass_{mass_scale:.2f}"
        SHIFTS[custom_name] = {"torso_mass": float(mass_scale)}
        fit_args.target = custom_name
        vec = fit_one(source_policy, basis, source_context, source_twin,
                      fit_args, device)
        z = (vec - ref_mean) @ P.T
        test2.append({
            "mass_scale": float(mass_scale),
            "z": z[:4].tolist(),
        })
        print(f"  mass={mass_scale:.2f} → z={z[:4].tolist()}", flush=True)
        del SHIFTS[custom_name]
    results["test2_mass_sweep"] = test2

    # ── Test 2b: friction sweep ───────────────────────────────────────
    print("\n=== Test 2b: friction sweep 0.70 → 1.20 ===")
    test2b = []
    for friction_scale in np.linspace(0.7, 1.2, 11):
        custom_name = f"_fric_{friction_scale:.2f}"
        SHIFTS[custom_name] = {"friction": float(friction_scale)}
        fit_args.target = custom_name
        vec = fit_one(source_policy, basis, source_context, source_twin,
                      fit_args, device)
        z = (vec - ref_mean) @ P.T
        test2b.append({
            "friction_scale": float(friction_scale),
            "z": z[:4].tolist(),
        })
        print(f"  friction={friction_scale:.2f} → z={z[:4].tolist()}", flush=True)
        del SHIFTS[custom_name]
    results["test2b_friction_sweep"] = test2b

    Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json_out).write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8",
    )
    print(f"\nSaved → {args.json_out}")


if __name__ == "__main__":
    main()
