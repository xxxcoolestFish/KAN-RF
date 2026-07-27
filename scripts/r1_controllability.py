"""R1: Reference Controllability — scan delta values, build c→c_achieved map.

Scans delta in two axes:
  Axis 1: forward velocity (vx) — change vx component of reference
  Axis 2: torso height (z) — change z component of reference

For each delta, runs KAN-B_t ILC and records achieved behavior + return.
"""
import sys, numpy as np, torch, argparse, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.prescreen_hopper_physics_shifts import SHIFTS, make_shifted_env
from scripts.validate_hopper_joint_online_adaptation import FrozenSourcePolicy, load_cognition
from scripts.diagnose_hopper_pullback_effect import fit_distilled_source_counterfactual_context, load_source_twin
from scripts.ilc1_ablation import *

DEVICE = torch.device("cuda")
SHIFT = SHIFTS["friction_070"]
N_TRIALS = 30  # fewer trials for scan
ETA = 0.3

# Scan points
DELTA_VX = [-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
DELTA_Z  = [-0.15, -0.05, 0.0, 0.05, 0.15, 0.25]


def make_axis_ref(ref_tr_obs, ref_tr_act, axis, delta):
    """Modify reference along one behavioral axis.

    axis='vx': change forward velocity (dim 6)
    axis='z':  change torso height (dim 0)
    """
    ref = ref_tr_obs.copy()
    if axis == 'vx':
        ref[:, 6] += delta  # vx dimension
    elif axis == 'z':
        ref[:, 0] += delta  # z (height) dimension
    elif axis == 'combo':
        ref[:, 6] += delta   # vx
        ref[:, 0] += delta * 0.5  # height (coupled)
    return ref


def extract_behavior_features(results, ref_tr_obs):
    """Extract achieved behavior from ILC trial returns and errors."""
    returns = np.array([r["return"] for r in results])
    # Use last 10 trials as "converged" behavior
    last10_r = np.mean(returns[-10:]) if len(returns) >= 10 else np.mean(returns)
    first5_r = np.mean(returns[:5])
    last5_r = np.mean(returns[-5:])
    errors = [r.get("error") for r in results if r.get("error") is not None]
    last10_err = np.mean(errors[-10:]) if len(errors) >= 10 else (np.mean(errors) if errors else float('nan'))

    return {
        "mean_return": float(np.mean(returns)),
        "last10_return": float(last10_r),
        "first5_return": float(first5_r),
        "last5_return": float(last5_r),
        "learning_delta": float(last5_r - first5_r),
        "mean_error": float(np.mean(errors)) if errors else None,
        "last10_error": float(last10_err) if not np.isnan(last10_err) else None,
        "n_cycles": sum(1 for r in results if r.get("error") is not None),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1811)
    parser.add_argument("--n-trials", type=int, default=N_TRIALS)
    args = parser.parse_args()

    print("=" * 60)
    print("R1: Reference Controllability Scan")
    print(f"  Delta vx: {DELTA_VX}")
    print(f"  Delta z:  {DELTA_Z}")
    print("=" * 60)

    # Load
    print("\n[1/3] Loading components...", flush=True)
    sp = FrozenSourcePolicy("results/hopper_source_sb3_ppo_continued_seed1811.zip",
        "results/hopper_source_sb3_vecnorm_continued_seed1811.pkl", DEVICE, args.seed, env="hopper")
    st = load_source_twin("results/hopper_source_affine_twin_cloud_seed1811.pt", DEVICE)
    basis, sc, _, _ = load_cognition(argparse.Namespace(
        cognition_checkpoint="results/hopper_source_control_sobolev_calibrated_seed1811.pt", device="cuda"), DEVICE)
    fa = argparse.Namespace(target="friction_070", seed=args.seed, env="hopper", device="cuda",
        cognition_warmup=1024, warmup_noise=0.3, transform_ridge=10.0, drift_ridge=100.0,
        drift_spectral_eta=0.0, drift_spectral_beta=1.0, drift_spectral_mode="max",
        drift_smooth_lambda=0.0, diagonal_transform=False)
    tc, _ = fit_distilled_source_counterfactual_context(sp, basis, sc, fa, DEVICE, st)

    # Build base reference
    print("\n[2/3] Building Transport reference...", flush=True)
    ref_tr_obs, ref_tr_act, ref_tr_len = build_reference(sp, basis, sc, tc, SHIFT, DEVICE, args.seed, n_ep=10)
    print(f"  Transport ref: {ref_tr_len} steps", flush=True)

    # Scan
    print(f"\n[3/3] Scanning ({len(DELTA_VX)} vx × {len(DELTA_Z)} z)...", flush=True)

    all_results = {}

    # First: vx axis scan
    for delta in DELTA_VX:
        ref = make_axis_ref(ref_tr_obs, ref_tr_act, 'vx', delta)
        label = f"vx={delta:+.2f}"
        sys.stdout.write(f"  {label}: "); sys.stdout.flush()

        results = run_ilc_experiment(sp, basis, sc, tc, SHIFT, DEVICE, args.seed,
                                     args.n_trials, ref, ref_tr_act, ref_tr_len, "kan", None, eta=ETA)
        features = extract_behavior_features(results, ref_tr_obs)
        features["desired_delta"] = delta
        features["axis"] = "vx"
        all_results[label] = features
        print(f"R={features['mean_return']:.1f} last10={features['last10_return']:.1f} "
              f"Δlearn={features['learning_delta']:+.1f} err={features['mean_error']:.4f}", flush=True)

    # Second: z axis scan
    for delta in DELTA_Z:
        ref = make_axis_ref(ref_tr_obs, ref_tr_act, 'z', delta)
        label = f"z={delta:+.2f}"
        sys.stdout.write(f"  {label}: "); sys.stdout.flush()

        results = run_ilc_experiment(sp, basis, sc, tc, SHIFT, DEVICE, args.seed,
                                     args.n_trials, ref, ref_tr_act, ref_tr_len, "kan", None, eta=ETA)
        features = extract_behavior_features(results, ref_tr_obs)
        features["desired_delta"] = delta
        features["axis"] = "z"
        all_results[label] = features
        print(f"R={features['mean_return']:.1f} last10={features['last10_return']:.1f} "
              f"Δlearn={features['learning_delta']:+.1f} err={features['mean_error']:.4f}", flush=True)

    # Third: combo axis (previous delta_0.15, 0.30)
    for delta in [0.15, 0.30, 0.40]:
        ref = make_axis_ref(ref_tr_obs, ref_tr_act, 'combo', delta)
        label = f"combo={delta:+.2f}"
        sys.stdout.write(f"  {label}: "); sys.stdout.flush()

        results = run_ilc_experiment(sp, basis, sc, tc, SHIFT, DEVICE, args.seed,
                                     args.n_trials, ref, ref_tr_act, ref_tr_len, "kan", None, eta=ETA)
        features = extract_behavior_features(results, ref_tr_obs)
        features["desired_delta"] = delta
        features["axis"] = "combo"
        all_results[label] = features
        print(f"R={features['mean_return']:.1f} last10={features['last10_return']:.1f} "
              f"Δlearn={features['learning_delta']:+.1f} err={features['mean_error']:.4f}", flush=True)

    # Summary
    print(f"\n{'='*60}")
    print(f"  {'Command':>12s} {'Axis':>6s} | {'Return':>8s} {'Last10':>8s} {'ΔLearn':>8s} {'Error':>8s}")
    print(f"  {'-'*52}")
    for label, r in sorted(all_results.items(), key=lambda x: x[1].get("mean_return", 0), reverse=True):
        print(f"  {label:>12s} {r['axis']:>6s} | {r['mean_return']:>8.1f} {r['last10_return']:>8.1f} "
              f"{r['learning_delta']:>+8.1f} {r.get('mean_error', 0):>8.4f}")

    # Best per axis
    for axis in ["vx", "z", "combo"]:
        ax_results = [(k, v) for k, v in all_results.items() if v["axis"] == axis]
        if ax_results:
            best = max(ax_results, key=lambda x: x[1]["mean_return"])
            print(f"  Best {axis}: {best[0]} → R={best[1]['mean_return']:.1f}")

    json.dump({"results": all_results, "config": {"deltas_vx": DELTA_VX, "deltas_z": DELTA_Z}},
              open("results/r1_controllability.json", "w"), indent=2)
    print("\n  Saved to results/r1_controllability.json")


if __name__ == "__main__":
    main()
