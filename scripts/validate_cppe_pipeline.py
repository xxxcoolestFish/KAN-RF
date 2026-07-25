"""CPPE pipeline validation: PCA reconstruction, drift-vs-gain PCA, transport_action.

Three validation tests before full implementation:
  Test 1: PCA reconstruction error (z → ΔW round-trip)
  Test 2: Drift-only vs full-coefficient PCA
  Test 3: transport_action effectiveness via effect matching
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
from scripts.prescreen_hopper_physics_shifts import ENVS, SHIFTS, make_shifted_env
from scripts.validate_hopper_joint_online_adaptation import (
    FrozenSourcePolicy,
    load_cognition,
)


def extract_coefficient_blocks(context, basis, source_context):
    """Return (drift_delta, gain_delta) as flat numpy arrays."""
    width = basis.feature_dim
    source_blocks = source_context.coefficients.reshape(
        1 + basis.action_dim, width, -1,
    )
    target_blocks = context.coefficients.reshape(
        1 + basis.action_dim, width, -1,
    )
    drift_delta = (target_blocks[0] - source_blocks[0]).detach().cpu()
    gain_delta = (target_blocks[1:] - source_blocks[1:]).detach().cpu()
    return drift_delta, gain_delta


def fit_all_shifts(source_policy, basis, source_context, fit_args, device, source_twin):
    """Fit W_t for all 7 shifts, return coefficient deltas."""
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

    results = {}
    for name in shift_names:
        print(f"  Fitting {name} (budget={fit_args.cognition_warmup}) ...", flush=True)
        fit_args.target = name
        context, _ = fit_distilled_source_counterfactual_context(
            source_policy, basis, source_context, fit_args, device, source_twin,
        )
        drift_delta, gain_delta = extract_coefficient_blocks(
            context, basis, source_context,
        )
        results[name] = {
            "drift": drift_delta.numpy(),  # (feature_dim, state_dim)
            "gain": gain_delta.numpy(),     # (action_dim, feature_dim, state_dim)
            "type": shift_types[name],
        }
    return results


def run_pca(vectors, k=4):
    """Run PCA on a matrix (n_samples, n_features). Returns components and stats."""
    matrix = np.stack(vectors)  # (n, d)
    mean = matrix.mean(axis=0)
    centered = matrix - mean
    U, s, Vh = np.linalg.svd(centered, full_matrices=False)
    explained = s ** 2 / (s ** 2).sum()

    # Project to k PCs
    z = centered @ Vh.T[:, :k]  # (n, k)

    # Reconstruct from k PCs
    reconstructed = mean + z @ Vh[:k, :]  # (n, d)

    # Per-sample relative reconstruction error
    errors = []
    for i in range(len(vectors)):
        orig_norm = np.linalg.norm(matrix[i])
        recon_norm = np.linalg.norm(reconstructed[i])
        err = np.linalg.norm(matrix[i] - reconstructed[i]) / max(orig_norm, 1e-10)
        errors.append(float(err))

    return {
        "mean": mean,
        "Vh": Vh,           # (n_components, d)
        "singular": s,
        "explained": explained.tolist(),
        "z": z.tolist(),
        "reconstruction_errors": errors,
        "mean_error": float(np.mean(errors)),
        "max_error": float(np.max(errors)),
    }


def test_transport_action(source_policy, basis, source_context,
                          drift_delta, source_twin, device,
                          n_states=100):
    """Test whether transport_action cancels physics drift in effect space."""
    # Collect states from source environment
    env = make_shifted_env(SHIFTS["source"], 1811, "hopper")()
    obs, _ = env.reset(seed=1811)
    states = []
    for _ in range(n_states + 50):
        a = source_policy.action(obs)
        obs, _, terminated, truncated, _ = env.step(a.cpu().numpy())
        states.append(obs.copy())
        if terminated or truncated:
            obs, _ = env.reset()
    env.close()

    states = torch.tensor(np.stack(states[-n_states:]), device=device, dtype=torch.float32)
    nominal = source_policy.action(states)  # (n, action_dim)

    # Source effect
    source_effect = source_context.acceleration(basis, states, nominal)

    # Build a temporary context with drifted coefficients
    width = basis.feature_dim
    n_action = basis.action_dim
    source_blocks = source_context.coefficients.clone().reshape(
        1 + n_action, width, -1,
    )
    drift_delta_t = torch.tensor(drift_delta, device=device, dtype=torch.float32)
    # drift_delta has shape (feature_dim, state_dim)
    drifted_blocks = source_blocks.clone()
    drifted_blocks[0] = source_blocks[0] + drift_delta_t

    from cpbn.generic_affine_kan import AffineKANContext
    drifted_context = AffineKANContext(drifted_blocks.reshape_as(source_context.coefficients))

    # No correction: use π_source directly in drifted physics
    drifted_source_effect = drifted_context.acceleration(basis, states, nominal)
    raw_effect_error = torch.norm(drifted_source_effect - source_effect, dim=-1)

    # With transport_action
    corrected_actions = []
    for i in range(len(states)):
        a_cf = drifted_context.transport_action(
            basis, states[i:i+1],
            desired_effect=source_effect[i:i+1],
            nominal_action=nominal[i:i+1],
            regularization=1e-2,
        )
        corrected_actions.append(a_cf)
    corrected = torch.cat(corrected_actions, dim=0)

    corrected_effect = drifted_context.acceleration(basis, states, corrected)
    corrected_effect_error = torch.norm(corrected_effect - source_effect, dim=-1)

    # Stats
    results = {
        "mean_raw_error": float(raw_effect_error.mean()),
        "mean_corrected_error": float(corrected_effect_error.mean()),
        "improvement_ratio": float(
            (raw_effect_error.mean() - corrected_effect_error.mean())
            / max(raw_effect_error.mean(), 1e-10)
        ),
        "fraction_improved": float(
            (corrected_effect_error < raw_effect_error).float().mean()
        ),
    }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=512,
                        help="Warmup steps per shift (smaller = faster validation)")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=1811)
    parser.add_argument("--json-out", default="results/cppe_pipeline_validation.json")
    parser.add_argument("--pca-out", default="results/cppe_pca_model.npz")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    print(f"Device: {device}, Budget: {args.budget}", flush=True)

    # ── Load components ─────────────────────────────────────────────────
    print("\nLoading source components ...", flush=True)
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

    # ── Fit all shifts ───────────────────────────────────────────────────
    print(f"\nFitting {7} shifts ...", flush=True)
    shift_data = fit_all_shifts(
        source_policy, basis, source_context, fit_args, device, source_twin,
    )

    # ── Test 1: PCA reconstruction (drift only) ─────────────────────────
    print("\n" + "=" * 60)
    print("Test 1: PCA Reconstruction (drift only)")
    print("=" * 60)
    drift_vectors = [
        shift_data[name]["drift"].flatten()
        for name in shift_data
    ]
    dims = [shift_data[name]["drift"].shape[0] * shift_data[name]["drift"].shape[1]
            for name in shift_data]
    print(f"  Drift vector dim: {dims[0]} ({shift_data[list(shift_data.keys())[0]]['drift'].shape[0]} features x state_dim)")

    pca_drift = run_pca(drift_vectors, k=4)
    print(f"  Explained variance (k=4): {pca_drift['explained'][:5]}")
    print(f"  Cumulated (k=4): {sum(pca_drift['explained'][:4]):.1%}")
    print(f"  Mean reconstruction error: {pca_drift['mean_error']:.4f}")
    print(f"  Max  reconstruction error: {pca_drift['max_error']:.4f}")
    for name, err in zip(shift_data.keys(), pca_drift['reconstruction_errors']):
        status = "OK" if err < 0.10 else ("WARN" if err < 0.20 else "FAIL")
        print(f"    {name}: {err:.4f} [{status}]")

    # ── Test 2: Drift-only vs full-coefficient PCA ─────────────────────
    print("\n" + "=" * 60)
    print("Test 2: Drift-only vs Full-coefficient PCA")
    print("=" * 60)

    # Full coefficients: drift + gain flattened
    full_vectors = []
    for name in shift_data:
        drift_flat = shift_data[name]["drift"].flatten()
        gain_flat = shift_data[name]["gain"].flatten()
        full_vectors.append(np.concatenate([drift_flat, gain_flat]))
    full_dim = len(full_vectors[0])
    drift_dim = len(drift_vectors[0])
    print(f"  Drift dimension: {drift_dim}")
    print(f"  Full coefficient dimension: {full_dim}")
    print(f"  Gain delta norm ratio vs drift: ", end="")
    for name in shift_data:
        d_norm = np.linalg.norm(shift_data[name]["drift"])
        g_norm = np.linalg.norm(shift_data[name]["gain"])
        print(f"\n    {name}: ||ΔG||/||Δb|| = {g_norm / max(d_norm, 1e-10):.3f}", end="")

    pca_full = run_pca(full_vectors, k=4)
    print(f"\n\n  Full-coefficient PCA explained variance (k=4): {pca_full['explained'][:5]}")
    print(f"  Full cumulated (k=4): {sum(pca_full['explained'][:4]):.1%}")
    print(f"  Drift-only cumulated (k=4): {sum(pca_drift['explained'][:4]):.1%}")
    print(f"\n  Conclusion: ", end="")
    if sum(pca_full['explained'][:4]) >= sum(pca_drift['explained'][:4]) - 0.02:
        print("Drift-only PCA is sufficient — gain delta carries little additional info")
    else:
        print("[WARN] Gain delta contains additional physics info — consider full-coefficient PCA")

    # ── Test 3: transport_action effectiveness ─────────────────────────
    print("\n" + "=" * 60)
    print("Test 3: transport_action effectiveness")
    print("=" * 60)

    for shift_name in ["payload_125", "friction_070", "combo_medium"]:
        print(f"\n  Testing {shift_name} ...", flush=True)
        result = test_transport_action(
            source_policy, basis, source_context,
            shift_data[shift_name]["drift"],
            source_twin, device,
            n_states=100,
        )
        print(f"    Mean raw effect error:    {result['mean_raw_error']:.6f}")
        print(f"    Mean corrected effect error: {result['mean_corrected_error']:.6f}")
        print(f"    Improvement ratio:       {result['improvement_ratio']:.1%}")
        print(f"    Fraction improved:       {result['fraction_improved']:.1%}")
        status = "OK" if result['improvement_ratio'] > 0.5 else ("WARN" if result['improvement_ratio'] > 0.2 else "FAIL")
        print(f"    Verdict: [{status}]")

    # ── Save PCA model ──────────────────────────────────────────────────
    Path(args.pca_out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.pca_out,
        mean=pca_drift["mean"],
        Vh=pca_drift["Vh"],
        singular=pca_drift["singular"],
        explained=pca_drift["explained"],
        drift_dim=drift_dim,
    )
    print(f"\nPCA model saved → {args.pca_out}")

    # ── Save validation report ─────────────────────────────────────────
    report = {
        "budget": args.budget,
        "test1_reconstruction": {
            "explained_variance": pca_drift["explained"][:5],
            "cumulated_k4": float(sum(pca_drift["explained"][:4])),
            "mean_error": pca_drift["mean_error"],
            "max_error": pca_drift["max_error"],
            "per_shift": dict(zip(shift_data.keys(), pca_drift["reconstruction_errors"])),
        },
        "test2_drift_vs_full": {
            "drift_dim": drift_dim,
            "full_dim": full_dim,
            "drift_cumulated_k4": float(sum(pca_drift["explained"][:4])),
            "full_cumulated_k4": float(sum(pca_full["explained"][:4])),
        },
        "test3_transport_action": {},
    }
    for shift_name in ["payload_125", "friction_070", "combo_medium"]:
        r = test_transport_action(
            source_policy, basis, source_context,
            shift_data[shift_name]["drift"],
            source_twin, device,
            n_states=100,
        )
        report["test3_transport_action"][shift_name] = r

    Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json_out).write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8",
    )
    print(f"Report saved → {args.json_out}")
    print("\nDone.")


if __name__ == "__main__":
    main()
