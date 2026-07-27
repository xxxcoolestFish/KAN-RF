"""R1.5: Achievement audit + instruction switch step-response.

Part A: Fixed commands with per-trial achievement metrics.
Part B: Switch step-response (c1 → c2, keep ILC residual).
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
N_TRIALS_PER_SEGMENT = 20
ETA = 0.3

FIXED_CMDS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]
SWITCHES = [
    (0.0, 0.10),
    (0.10, 0.0),
    (0.10, 0.20),
    (0.20, 0.10),
    (0.30, 0.10),
]


def make_vx_ref(ref_tr_obs, ref_tr_act, vx_delta):
    ref = ref_tr_obs.copy()
    ref[:, 6] += vx_delta
    return ref


def run_ilc_with_metrics(sp, basis, sc, tc, shift, device, seed, n_trials,
                         ref_obs, ref_act, ref_cyc_len, u_init=None):
    """Run ILC and collect per-trial achievement metrics.

    Returns: list of per-trial dicts with return, achieved_vx, error, |u|, fall.
    """
    a_dim = basis.action_dim
    u_table = np.zeros((N_PHASE, a_dim), dtype=np.float32) if u_init is None else u_init.copy()
    results = []

    for trial in range(n_trials):
        env = make_shifted_env(shift, seed + trial * 100, "hopper")()
        obs, _ = env.reset()
        total_r, total_vx, vx_count = 0.0, 0.0, 0
        detector = CycleDetector(); detector.reset()
        obs_trace, act_trace, td_steps = [], [], []
        step = 0; fell = False

        while True:
            s_t = torch.as_tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
            nominal = sp.action(s_t)
            s_eff = sc.acceleration(basis, s_t, nominal)
            a_tr = tc.transport_action(basis, s_t, desired_effect=s_eff,
                                       nominal_action=nominal, regularization=1e-2
                                       ).clamp(-1, 1).squeeze(0).cpu().numpy()

            if len(td_steps) > 0:
                phase = min((step - td_steps[-1]) / ref_cyc_len, 0.99)
                u_ff = u_table[min(int(phase * N_PHASE), N_PHASE - 1)]
                a_final = np.clip(a_tr + u_ff, -1, 1)
            else:
                a_final = a_tr

            next_obs, reward, terminated, truncated, _ = env.step(a_final)
            total_r += float(reward)
            total_vx += next_obs[5]  # x_velocity
            vx_count += 1
            obs_trace.append(obs.copy()); act_trace.append(a_final.copy())
            if detector.update(env): td_steps.append(step)
            obs = next_obs; step += 1
            if terminated or truncated:
                if terminated: fell = True
                break

        env.close()

        trial_res = {"trial": trial, "return": float(total_r),
                     "achieved_vx": float(total_vx / max(vx_count, 1)),
                     "fell": fell, "length": step}

        cycle_data = None
        for i in range(len(td_steps) - 1):
            s, e = td_steps[i], td_steps[i + 1]
            if 25 <= (e - s) <= 70:
                cycle_data = {"obs": np.array(obs_trace[s:e]), "actions": np.array(act_trace[s:e])}
                break

        if cycle_data is not None:
            L = cycle_data["obs"].shape[0]
            t_old = np.linspace(0, 1, L); t_new = np.linspace(0, 1, N_PHASE)
            aligned = np.zeros((N_PHASE, cycle_data["obs"].shape[1]))
            for d in range(cycle_data["obs"].shape[1]):
                aligned[:, d] = np.interp(t_new, t_old, cycle_data["obs"][:, d])
            task_dims = [0, 2, 6, 3, 4]; tw = np.array([2.0, 1.0, 1.0, 0.5, 0.5])
            err = sum(np.sum((tw * (ref_obs[i, task_dims] - aligned[i, task_dims])) ** 2)
                     for i in range(N_PHASE))
            trial_res["error"] = float(np.sqrt(err / N_PHASE))
            trial_res["residual_norm"] = float(np.linalg.norm(u_table))
            trial_res["cycle_found"] = True
            u_table, _ = ilc_update_with_bt(u_table, cycle_data["obs"], ref_obs,
                                            make_vx_bt_func(basis, tc), None, eta=ETA)
        else:
            trial_res["error"] = None; trial_res["residual_norm"] = float(np.linalg.norm(u_table))
            trial_res["cycle_found"] = False

        results.append(trial_res)

    return results, u_table


def make_vx_bt_func(basis, tc):
    def f(args, s_np):
        s_t = torch.as_tensor(s_np, device=DEVICE, dtype=torch.float32).unsqueeze(0)
        return get_kan_bt(basis, tc, s_t)
    return f


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1811)
    args = parser.parse_args()

    print("=" * 60)
    print("R1.5: Achievement Audit + Step Response")
    print("=" * 60)

    # Load
    print("\n[1/3] Loading...", flush=True)
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
    ref_tr_obs, ref_tr_act, ref_tr_len = build_reference(sp, basis, sc, tc, SHIFT, DEVICE, args.seed, n_ep=10)
    print(f"  Reference: {ref_tr_len} steps", flush=True)

    all_results = {}

    # ── Part A: Fixed commands with achievement metrics ─────────────────
    print(f"\n[2/3] Part A: Fixed commands ({len(FIXED_CMDS)} commands × {N_TRIALS_PER_SEGMENT} trials)...", flush=True)

    for vx in FIXED_CMDS:
        ref = make_vx_ref(ref_tr_obs, ref_tr_act, vx)
        label = f"fixed_vx={vx:.2f}"
        sys.stdout.write(f"  {label}: "); sys.stdout.flush()

        results, final_u = run_ilc_with_metrics(
            sp, basis, sc, tc, SHIFT, DEVICE, args.seed, N_TRIALS_PER_SEGMENT,
            ref, ref_tr_act, ref_tr_len)

        returns = [r["return"] for r in results]
        achieved_vx = [r["achieved_vx"] for r in results if not r["fell"]]
        errors = [r["error"] for r in results if r.get("cycle_found")]
        fell_rate = np.mean([r["fell"] for r in results])
        last10_r = np.mean(returns[-10:])
        last10_vx = np.mean(achieved_vx[-10:]) if achieved_vx else 0
        last10_err = np.mean(errors[-10:]) if errors else float('nan')
        final_residual = float(np.linalg.norm(final_u))

        print(f"R={np.mean(returns):.1f} last10R={last10_r:.1f} "
              f"vx={np.mean(achieved_vx):.2f} last10vx={last10_vx:.2f} "
              f"err={last10_err:.4f} |u|={final_residual:.3f} fall={fell_rate:.0%}", flush=True)

        all_results[label] = {
            "desired_vx": vx, "mean_return": float(np.mean(returns)),
            "last10_return": float(last10_r), "mean_achieved_vx": float(np.mean(achieved_vx)),
            "last10_achieved_vx": float(last10_vx), "last10_error": float(last10_err) if not np.isnan(last10_err) else None,
            "final_residual": final_residual, "fell_rate": float(fell_rate),
            "per_trial": [{"trial": r["trial"], "return": r["return"],
                           "achieved_vx": r["achieved_vx"], "fell": r["fell"],
                           "error": r.get("error"), "residual_norm": r.get("residual_norm")}
                          for r in results],
        }

    # ── Part B: Switches ────────────────────────────────────────────────
    print(f"\n[3/3] Part B: Switches ({len(SWITCHES)} switches × 2×{N_TRIALS_PER_SEGMENT} trials)...", flush=True)

    for c1, c2 in SWITCHES:
        ref1 = make_vx_ref(ref_tr_obs, ref_tr_act, c1)
        ref2 = make_vx_ref(ref_tr_obs, ref_tr_act, c2)
        label = f"switch_{c1:.2f}→{c2:.2f}"
        sys.stdout.write(f"  {label}: "); sys.stdout.flush()

        # Segment 1: c1
        results1, u_after_s1 = run_ilc_with_metrics(
            sp, basis, sc, tc, SHIFT, DEVICE, args.seed, N_TRIALS_PER_SEGMENT,
            ref1, ref_tr_act, ref_tr_len)
        s1_last5_r = np.mean([r["return"] for r in results1[-5:]])
        s1_last5_vx = np.mean([r["achieved_vx"] for r in results1[-5:] if not r["fell"]])
        s1_final_u = float(np.linalg.norm(u_after_s1))

        # Segment 2: c2 (keep ILC residual from s1)
        results2, u_after_s2 = run_ilc_with_metrics(
            sp, basis, sc, tc, SHIFT, DEVICE, args.seed, N_TRIALS_PER_SEGMENT,
            ref2, ref_tr_act, ref_tr_len, u_init=u_after_s1)
        s2_first5_r = np.mean([r["return"] for r in results2[:5]])
        s2_last5_r = np.mean([r["return"] for r in results2[-5:]])
        s2_first5_vx = np.mean([r["achieved_vx"] for r in results2[:5] if not r["fell"]])
        s2_last5_vx = np.mean([r["achieved_vx"] for r in results2[-5:] if not r["fell"]])
        s2_final_u = float(np.linalg.norm(u_after_s2))

        # Compare with fixed c2 (from Part A)
        fixed_c2 = all_results.get(f"fixed_vx={c2:.2f}", {})
        fixed_c2_last5 = fixed_c2.get("last10_return", float('nan'))

        print(f"s1_last5=[R={s1_last5_r:.1f} vx={s1_last5_vx:.2f} |u|={s1_final_u:.3f}] "
              f"→ s2_first5=[R={s2_first5_r:.1f} vx={s2_first5_vx:.2f}] "
              f"s2_last5=[R={s2_last5_r:.1f} vx={s2_last5_vx:.2f} |u|={s2_final_u:.3f}] "
              f"vs_fixed_c2_last5={fixed_c2_last5:.1f}", flush=True)

        all_results[label] = {
            "c1": c1, "c2": c2,
            "s1_last5_return": float(s1_last5_r), "s1_last5_vx": float(s1_last5_vx),
            "s1_final_residual": s1_final_u,
            "s2_first5_return": float(s2_first5_r), "s2_first5_vx": float(s2_first5_vx),
            "s2_last5_return": float(s2_last5_r), "s2_last5_vx": float(s2_last5_vx),
            "s2_final_residual": s2_final_u,
            "fixed_c2_last10": float(fixed_c2_last5) if not np.isnan(fixed_c2_last5) else None,
            "per_trial_s1": [{"trial": r["trial"], "return": r["return"],
                              "achieved_vx": r["achieved_vx"], "fell": r["fell"]}
                             for r in results1],
            "per_trial_s2": [{"trial": r["trial"], "return": r["return"],
                              "achieved_vx": r["achieved_vx"], "fell": r["fell"]}
                             for r in results2],
        }

    # Summary
    print(f"\n{'='*60}")
    print("  Part A: Fixed Command Achievement")
    print(f"  {'vx_desired':>10s} {'Return':>8s} {'Achieved_vx':>10s} {'Error':>8s} {'|u|':>8s} {'Fall':>6s}")
    for vx in FIXED_CMDS:
        r = all_results[f"fixed_vx={vx:.2f}"]
        print(f"  {vx:>10.2f} {r['last10_return']:>8.1f} {r['last10_achieved_vx']:>10.2f} "
              f"{r.get('last10_error', 0) or 0:>8.4f} {r['final_residual']:>8.3f} {r['fell_rate']:>6.0%}")

    print(f"\n  Part B: Switch Step Response")
    print(f"  {'Switch':>16s} {'S1_last5':>10s} {'S2_first5':>10s} {'S2_last5':>10s} {'Fixed_c2':>10s}")
    for c1, c2 in SWITCHES:
        r = all_results[f"switch_{c1:.2f}→{c2:.2f}"]
        print(f"  {c1:.2f}→{c2:.2f}         {r['s1_last5_return']:>10.1f} {r['s2_first5_return']:>10.1f} "
              f"{r['s2_last5_return']:>10.1f} {r.get('fixed_c2_last10', 0) or 0:>10.1f}")

    json.dump({"fixed": {k: v for k, v in all_results.items() if k.startswith("fixed")},
               "switches": {k: v for k, v in all_results.items() if k.startswith("switch")}},
              open("results/r1p5_step_response.json", "w"), indent=2)
    print("\n  Saved to results/r1p5_step_response.json")


if __name__ == "__main__":
    main()
