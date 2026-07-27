"""ILC-1 fast multi-seed: essential conditions only, pre-built reference."""
import sys, numpy as np, torch, argparse, json
from pathlib import Path
sys.path.insert(0, ".")
from scripts.prescreen_hopper_physics_shifts import SHIFTS, make_shifted_env
from scripts.validate_hopper_joint_online_adaptation import FrozenSourcePolicy, load_cognition
from scripts.diagnose_hopper_pullback_effect import fit_distilled_source_counterfactual_context, load_source_twin
from scripts.ilc1_ablation import *

DEVICE = torch.device("cuda")
SHIFT = SHIFTS["friction_070"]
N_TRIALS = 30
ETA = 0.3
REF_NPZ = "results/ilc1_reference.npz"

# Essential conditions per seed
CONDITIONS = [
    ("kan", "tr"),
    ("kan", "delta_0.15"),
    ("kan", "delta_0.30"),
    ("proportional", "tr"),
    ("shuffled", "tr"),
    ("shuffled", "delta_0.15"),
    ("negative", "tr"),
]

def build_and_save_ref(sp, basis, sc, tc, seed):
    print("Building reference (once)...", flush=True)
    ref_obs, ref_act, ref_cyc_len = build_reference(sp, basis, sc, tc, SHIFT, DEVICE, seed, n_ep=10)
    np.savez(REF_NPZ, ref_obs=ref_obs, ref_act=ref_act, ref_cyc_len=ref_cyc_len)
    print(f"  Saved: ref_cyc_len={ref_cyc_len}", flush=True)
    return ref_obs, ref_act, ref_cyc_len

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=str, default="1911,2011")
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    print(f"ILC-1 Fast: {len(seeds)} seeds × {len(CONDITIONS)} conditions × {N_TRIALS} trials")

    # Load once
    print("[1/3] Loading...", flush=True)
    sp = FrozenSourcePolicy("results/hopper_source_sb3_ppo_continued_seed1811.zip",
        "results/hopper_source_sb3_vecnorm_continued_seed1811.pkl", DEVICE, seeds[0], env="hopper")
    st = load_source_twin("results/hopper_source_affine_twin_cloud_seed1811.pt", DEVICE)
    basis, sc, _, _ = load_cognition(argparse.Namespace(
        cognition_checkpoint="results/hopper_source_control_sobolev_calibrated_seed1811.pt", device="cuda"), DEVICE)
    fa0 = argparse.Namespace(target="friction_070", seed=seeds[0], env="hopper", device="cuda",
        cognition_warmup=1024, warmup_noise=0.3, transform_ridge=10.0, drift_ridge=100.0,
        drift_spectral_eta=0.0, drift_spectral_beta=1.0, drift_spectral_mode="max",
        drift_smooth_lambda=0.0, diagonal_transform=False)
    tc0, _ = fit_distilled_source_counterfactual_context(sp, basis, sc, fa0, DEVICE, st)

    # Build reference once
    if Path(REF_NPZ).exists():
        d = np.load(REF_NPZ)
        ref_obs, ref_act, ref_cyc_len = d["ref_obs"], d["ref_act"], int(d["ref_cyc_len"])
        print(f"  Loaded reference: {ref_cyc_len} steps", flush=True)
    else:
        ref_obs, ref_act, ref_cyc_len = build_and_save_ref(sp, basis, sc, tc0, seeds[0])

    refs = {
        "tr": (ref_obs, ref_act, ref_cyc_len),
        "delta_0.15": (make_progressive_ref(ref_obs, None, 0.15), ref_act, ref_cyc_len),
        "delta_0.30": (make_progressive_ref(ref_obs, None, 0.30), ref_act, ref_cyc_len),
    }

    all_results = {}

    for seed in seeds:
        # Fit KAN for this seed
        if seed == seeds[0]:
            tc = tc0
        else:
            fa = argparse.Namespace(target="friction_070", seed=seed, env="hopper", device="cuda",
                cognition_warmup=1024, warmup_noise=0.3, transform_ridge=10.0, drift_ridge=100.0,
                drift_spectral_eta=0.0, drift_spectral_beta=1.0, drift_spectral_mode="max",
                drift_smooth_lambda=0.0, diagonal_transform=False)
            tc, _ = fit_distilled_source_counterfactual_context(sp, basis, sc, fa, DEVICE, st)

        print(f"\n[2/3] Seed {seed}...", flush=True)

        for bt_type, ref_name in CONDITIONS:
            r_obs, r_act, r_len = refs[ref_name]
            label = f"s{seed}/{bt_type}/{ref_name}"
            sys.stdout.write(f"  {label}: "); sys.stdout.flush()

            results = run_ilc_experiment(sp, basis, sc, tc, SHIFT, DEVICE, seed,
                                         N_TRIALS, r_obs, r_act, r_len, bt_type, None, eta=ETA)
            returns = [r["return"] for r in results]
            errors = [r["error"] for r in results if r["error"] is not None]
            n_cyc = sum(1 for r in results if r["error"] is not None)
            mr, f5, l5 = np.mean(returns), np.mean(returns[:5]), np.mean(returns[-5:])
            me = np.mean(errors) if errors else float('nan')

            print(f"R={mr:.1f} (f5={f5:.1f} l5={l5:.1f}) E={me:.4f} cyc={n_cyc}/{N_TRIALS}", flush=True)

            all_results[label] = {
                "seed": seed, "bt_type": bt_type, "ref": ref_name,
                "mean_return": float(mr), "first5_return": float(f5), "last5_return": float(l5),
                "mean_error": float(me) if not np.isnan(me) else None,
                "n_cycles": n_cyc, "returns": returns,
                "errors": [float(e) if e is not None else None for e in errors],
            }

    # ── Summary ─────────────────────────────────────────────────────────
    print(f"\n[3/3] Summary across {len(seeds)} seeds")
    print(f"  {'B_t':>15s} {'Ref':>12s} | {'Mean R':>10s} {'First5':>10s} {'Last5':>10s} {'ΔLearn':>10s} {'ΔvsTr':>10s}")
    print(f"  {'-'*80}")

    tr_base = np.mean([v["mean_return"] for k, v in all_results.items()
                       if v["bt_type"] == "kan" and v["ref"] == "tr"])

    for bt in ["kan", "proportional", "shuffled", "negative"]:
        for ref in ["tr", "delta_0.15", "delta_0.30"]:
            key_prefix = f"s"
            bt_ref_results = [v for k, v in all_results.items()
                             if v["bt_type"] == bt and v["ref"] == ref]
            if not bt_ref_results:
                continue
            avg_r = np.mean([r["mean_return"] for r in bt_ref_results])
            avg_f5 = np.mean([r["first5_return"] for r in bt_ref_results])
            avg_l5 = np.mean([r["last5_return"] for r in bt_ref_results])
            delta_learn = avg_l5 - avg_f5
            vs_tr = avg_r - tr_base
            print(f"  {bt:>15s} {ref:>12s} | {avg_r:>10.1f} {avg_f5:>10.1f} {avg_l5:>10.1f} "
                  f"{delta_learn:>+10.1f} {vs_tr:>+10.1f}")

    print(f"\n  Transport baseline (est): {tr_base:.1f}")
    print(f"  Source policy: 672")

    json.dump({"config": {"seeds": seeds, "n_trials": N_TRIALS, "conditions": CONDITIONS},
               "transport_baseline": float(tr_base), "results": all_results},
              open("results/ilc1_fast.json", "w"), indent=2)
    print("  Saved to results/ilc1_fast.json")

if __name__ == "__main__":
    main()
