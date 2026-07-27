"""R1.5 fix: corrected achieved_vx + residual reset ablation.

Minimal confirmatory experiment:
  Fixed: c ∈ {0, 0.10, 0.20}
  Switches: 0→0.10, 0.20→0.10
  Each switch: keep residual vs reset residual vs conditional residual
"""
import sys, numpy as np, torch, argparse, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.prescreen_hopper_physics_shifts import SHIFTS, make_shifted_env
from scripts.validate_hopper_joint_online_adaptation import FrozenSourcePolicy, load_cognition
from scripts.diagnose_hopper_pullback_effect import fit_distilled_source_counterfactual_context, load_source_twin
from scripts.ilc1_ablation import *

DEVICE = torch.device("cuda"); SHIFT = SHIFTS["friction_070"]
N_TRIALS = 25; ETA = 0.3
VX_IDX = 6  # Hopper-v5 x-velocity index (was 5, bug!)

def make_vx_ref(ref_obs, ref_act, delta):
    r = ref_obs.copy(); r[:, VX_IDX] += delta; return r

def run_ilc_trials(sp, basis, sc, tc, seed, n_trials, ref_obs, ref_act, ref_len,
                   u_init=None, record_achieved=True):
    """Run ILC trials. u_init=None starts from zero. Returns (results_list, final_u)."""
    a_dim = basis.action_dim
    u = np.zeros((N_PHASE, a_dim), dtype=np.float32) if u_init is None else u_init.copy()
    results = []
    for trial in range(n_trials):
        env = make_shifted_env(SHIFT, seed + trial * 100, "hopper")()
        obs, _ = env.reset(); total_r, total_vx, vx_n = 0.0, 0.0, 0
        detector = CycleDetector(); detector.reset()
        obs_trace, act_trace, td_steps = [], [], []; step = 0; fell = False
        while True:
            s_t = torch.as_tensor(obs, device=DEVICE, dtype=torch.float32).unsqueeze(0)
            nominal = sp.action(s_t)
            s_eff = sc.acceleration(basis, s_t, nominal)
            a_tr = tc.transport_action(basis, s_t, desired_effect=s_eff,
                                       nominal_action=nominal, regularization=1e-2
                                       ).clamp(-1,1).squeeze(0).cpu().numpy()
            if len(td_steps) > 0:
                phase = min((step - td_steps[-1]) / ref_len, 0.99)
                a_final = np.clip(a_tr + u[min(int(phase * N_PHASE), N_PHASE - 1)], -1, 1)
            else:
                a_final = a_tr
            next_obs, reward, terminated, truncated, _ = env.step(a_final)
            total_r += float(reward)
            if record_achieved and len(next_obs) > VX_IDX:
                total_vx += float(next_obs[VX_IDX]); vx_n += 1
            obs_trace.append(obs.copy()); act_trace.append(a_final.copy())
            if detector.update(env): td_steps.append(step)
            obs = next_obs; step += 1
            if terminated or truncated:
                fell = (step < 80)  # short episode = genuine fall
                break
        env.close()

        tr = {"trial": trial, "return": float(total_r), "fell": fell, "length": step}
        if record_achieved and vx_n > 0:
            tr["achieved_vx"] = float(total_vx / vx_n)

        cd = None
        for i in range(len(td_steps) - 1):
            s, e = td_steps[i], td_steps[i + 1]
            if 25 <= (e - s) <= 70:
                cd = {"obs": np.array(obs_trace[s:e]), "actions": np.array(act_trace[s:e])}; break

        if cd is not None:
            L = cd["obs"].shape[0]; t_old = np.linspace(0, 1, L); t_new = np.linspace(0, 1, N_PHASE)
            al = np.zeros((N_PHASE, cd["obs"].shape[1]))
            for d in range(cd["obs"].shape[1]): al[:, d] = np.interp(t_new, t_old, cd["obs"][:, d])
            td_ = [0, 2, 6, 3, 4]; tw = np.array([2., 1., 1., .5, .5])
            err = sum(np.sum((tw * (ref_obs[i, td_] - al[i, td_])) ** 2) for i in range(N_PHASE))
            tr["error"] = float(np.sqrt(err / N_PHASE))
            tr["residual_norm"] = float(np.linalg.norm(u))
            u, _ = ilc_update_with_bt(u, cd["obs"], ref_obs,
                lambda a, s: get_kan_bt(basis, tc, torch.as_tensor(s, device=DEVICE, dtype=torch.float32).unsqueeze(0)),
                None, eta=ETA)
        else:
            tr["error"] = None; tr["residual_norm"] = float(np.linalg.norm(u))
        results.append(tr)
    return results, u


def summarize(rs, u_final, label=""):
    r = [x["return"] for x in rs]; alive = [x for x in rs if not x["fell"]]
    vx = [x["achieved_vx"] for x in alive if "achieved_vx" in x]
    err = [x["error"] for x in rs if x.get("error") is not None]
    fell = np.mean([x["fell"] for x in rs])
    return (f"{label}: R={np.mean(r):.1f} last5={np.mean(r[-5:]):.1f} "
            f"vx={np.mean(vx):.2f} err={np.mean(err):.4f} "
            f"|u|={np.linalg.norm(u_final):.3f} fell={fell:.0%}")


def main():
    p = argparse.ArgumentParser(); p.add_argument("--seed", type=int, default=1811); a = p.parse_args()
    print("R1.5 fix: achievement + residual reset ablation")
    print("[1/2] Loading..."); sys.stdout.flush()
    sp = FrozenSourcePolicy("results/hopper_source_sb3_ppo_continued_seed1811.zip",
        "results/hopper_source_sb3_vecnorm_continued_seed1811.pkl", DEVICE, a.seed, env="hopper")
    st = load_source_twin("results/hopper_source_affine_twin_cloud_seed1811.pt", DEVICE)
    basis, sc, _, _ = load_cognition(argparse.Namespace(
        cognition_checkpoint="results/hopper_source_control_sobolev_calibrated_seed1811.pt", device="cuda"), DEVICE)
    fa = argparse.Namespace(target="friction_070", seed=a.seed, env="hopper", device="cuda",
        cognition_warmup=1024, warmup_noise=0.3, transform_ridge=10.0, drift_ridge=100.0,
        drift_spectral_eta=0.0, drift_spectral_beta=1.0, drift_spectral_mode="max",
        drift_smooth_lambda=0.0, diagonal_transform=False)
    tc, _ = fit_distilled_source_counterfactual_context(sp, basis, sc, fa, DEVICE, st)
    ro, ra, rl = build_reference(sp, basis, sc, tc, SHIFT, DEVICE, a.seed, n_ep=10)
    print(f"  Ref: {rl} steps"); sys.stdout.flush()

    # ── Fixed commands ──
    print("\n[2/2] Fixed + switches + residual ablation")
    fixed_results = {}
    for vx in [0.0, 0.10, 0.20]:
        ref = make_vx_ref(ro, ra, vx)
        rs, u = run_ilc_trials(sp, basis, sc, tc, a.seed, N_TRIALS, ref, ra, rl)
        fixed_results[vx] = {"results": rs, "final_u": u}
        print("  " + summarize(rs, u, f"FIXED vx={vx:.2f}")); sys.stdout.flush()

    # ── Switches ──
    print("\n  Switches:"); sys.stdout.flush()
    switch_results = {}
    for c1, c2 in [(0.0, 0.10), (0.20, 0.10)]:
        ref1 = make_vx_ref(ro, ra, c1); ref2 = make_vx_ref(ro, ra, c2)

        # S1: command c1
        r1, u1 = run_ilc_trials(sp, basis, sc, tc, a.seed, N_TRIALS, ref1, ra, rl)
        s1_last5 = np.mean([x["return"] for x in r1[-5:]])
        print(f"  S1({c1}): last5R={s1_last5:.1f} |u|={np.linalg.norm(u1):.3f}"); sys.stdout.flush()

        # S2 variants:
        # (a) Reset residual (fresh start = same as fixed c2, from fixed_results)
        fixed_c2_r = np.mean([x["return"] for x in fixed_results[c2]["results"]])
        fixed_c2_last5 = np.mean([x["return"] for x in fixed_results[c2]["results"][-5:]])

        # (b) Keep residual from c1
        r2_keep, u2_keep = run_ilc_trials(sp, basis, sc, tc, a.seed + 1000, N_TRIALS,
                                           ref2, ra, rl, u_init=u1)
        print("  " + summarize(r2_keep, u2_keep, f"  S2({c2}) KEEP residual"))
        sys.stdout.flush()

        # (c) Reset residual
        r2_reset, u2_reset = run_ilc_trials(sp, basis, sc, tc, a.seed + 2000, N_TRIALS,
                                             ref2, ra, rl, u_init=None)
        print("  " + summarize(r2_reset, u2_reset, f"  S2({c2}) RESET residual"))
        sys.stdout.flush()

        switch_results[f"{c1}→{c2}"] = {
            "s1_last5": float(s1_last5),
            "keep": {"last5": float(np.mean([x["return"] for x in r2_keep[-5:]])),
                     "mean": float(np.mean([x["return"] for x in r2_keep]))},
            "reset": {"last5": float(np.mean([x["return"] for x in r2_reset[-5:]])),
                      "mean": float(np.mean([x["return"] for x in r2_reset]))},
            "fixed_c2_last5": float(fixed_c2_last5),
            "fixed_c2_mean": float(fixed_c2_r),
        }

    # ── Summary ──
    print(f"\n{'='*60}")
    print("  Fixed Commands (achieved vx corrected):")
    print(f"  {'vx':>6s} {'Return':>8s} {'Last5':>8s} {'Ach_vx':>8s} {'Error':>8s} {'|u|':>8s} {'Fall':>6s}")
    for vx in [0.0, 0.10, 0.20]:
        rs = fixed_results[vx]["results"]; u = fixed_results[vx]["final_u"]
        r_all = [x["return"] for x in rs]; alive = [x for x in rs if not x["fell"]]
        vx_a = [x["achieved_vx"] for x in alive if "achieved_vx" in x]
        err = [x["error"] for x in rs if x.get("error") is not None]
        print(f"  {vx:>6.2f} {np.mean(r_all):>8.1f} {np.mean(r_all[-5:]):>8.1f} "
              f"{np.mean(vx_a):>8.2f} {np.mean(err):>8.4f} "
              f"{np.linalg.norm(u):>8.3f} {np.mean([x['fell'] for x in rs]):>6.0%}")

    print(f"\n  Switches (keep vs reset residual):")
    print(f"  {'Switch':>12s} {'Keep_last5':>12s} {'Reset_last5':>12s} {'Fixed_c2':>12s}")
    for sw, sr in switch_results.items():
        print(f"  {sw:>12s} {sr['keep']['last5']:>12.1f} {sr['reset']['last5']:>12.1f} "
              f"{sr['fixed_c2_last5']:>12.1f}")

    json.dump({"fixed": {str(k): {"returns": [x["return"] for x in v["results"]],
                                   "achieved_vx": [x.get("achieved_vx") for x in v["results"] if not x["fell"]]}
                          for k, v in fixed_results.items()},
               "switches": switch_results},
              open("results/r1p5_fixed.json", "w"), indent=2)
    print("  Saved to results/r1p5_fixed.json")


if __name__ == "__main__":
    main()
