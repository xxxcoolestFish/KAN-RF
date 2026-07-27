"""E2: Pre-termination trajectory alignment.

Runs Transport and KAN-ILC episodes, records last 50 steps before termination.
Aligns all failure episodes at tau=0 (termination), computes health margins,
state trajectories, action signatures, and KAN prediction quality.
"""
import sys, numpy as np, torch, argparse, json, time
from pathlib import Path
if sys.platform == 'win32':
    try: sys.stdout.reconfigure(encoding='utf-8')
    except: pass
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.prescreen_hopper_physics_shifts import SHIFTS, make_shifted_env
from scripts.validate_hopper_joint_online_adaptation import FrozenSourcePolicy, load_cognition
from scripts.diagnose_hopper_pullback_effect import fit_distilled_source_counterfactual_context, load_source_twin
from scripts.ilc1_ablation import *

DEVICE = torch.device("cuda")
N_EPISODES = 50
N_ILC_WARMUP = 25
ETA = 0.3
PRE_STEPS = 50  # steps before termination to record

# Hopper-v5 obs indices (from Gymnasium source)
IDX_Z, IDX_ANGLE = 0, 1
IDX_THIGH, IDX_LEG, IDX_FOOT = 2, 3, 4
IDX_VX, IDX_VZ, IDX_VANG = 5, 6, 7
IDX_VTHIGH, IDX_VLEG, IDX_VFOOT = 8, 9, 10
Z_MIN, ANGLE_MAX = 0.7, 0.2


def record_failing_episode(env_factory, sp, basis, sc, tc, seed, method,
                            u_table=None, ref_len=64):
    """Run episode with full recording. If it fails, return last PRE_STEPS."""
    env = env_factory()
    obs, _ = env.reset(seed=seed)
    step = 0; total_r = 0.0
    detector = CycleDetector(); detector.reset()
    td_steps = []

    # Full traces
    obs_trace, act_trace, a_tr_trace = [], [], []
    ilc_trace, reward_trace = [], []
    kan_err_trace = []  # KAN single-step prediction error

    while True:
        s_t = torch.as_tensor(obs, device=DEVICE, dtype=torch.float32).unsqueeze(0)
        nominal = sp.action(s_t)
        s_eff = sc.acceleration(basis, s_t, nominal)
        a_tr = tc.transport_action(basis, s_t, desired_effect=s_eff,
                                   nominal_action=nominal, regularization=1e-2
                                   ).clamp(-1, 1).squeeze(0).cpu().numpy()

        # KAN prediction error (one-step)
        kan_effect = tc.acceleration(basis, s_t, torch.as_tensor(a_tr, device=DEVICE).unsqueeze(0))
        kan_pred = obs + kan_effect.squeeze(0).cpu().numpy()

        u_ff = np.zeros_like(a_tr)
        if u_table is not None and len(td_steps) > 0:
            phase = min((step - td_steps[-1]) / ref_len, 0.99)
            u_ff = u_table[min(int(phase * N_PHASE), N_PHASE - 1)]
        a_final = np.clip(a_tr + u_ff, -1, 1)

        next_obs, reward, terminated, truncated, info = env.step(a_final)

        obs_trace.append(obs.copy())
        act_trace.append(a_final.copy())
        a_tr_trace.append(a_tr.copy())
        ilc_trace.append(u_ff.copy())
        reward_trace.append(float(reward))
        # KAN prediction error (norm of difference from actual next obs)
        kan_err_trace.append(float(np.linalg.norm(kan_pred - next_obs)))

        total_r += float(reward)
        if detector.update(env): td_steps.append(step)
        obs = next_obs; step += 1
        if terminated or truncated: break

    env.close()

    fell = bool(terminated)
    if not fell: return None  # only save failing episodes

    L = len(obs_trace)
    n_save = min(PRE_STEPS, L)
    return {
        "method": method, "seed": seed, "length": L,
        "total_return": float(total_r), "fell": True,
        # Last n_save steps (tau=0 is last step = termination)
        "obs": [obs_trace[i].tolist() for i in range(L - n_save, L)],
        "actions": [act_trace[i].tolist() for i in range(L - n_save, L)],
        "transport": [a_tr_trace[i].tolist() for i in range(L - n_save, L)],
        "ilc_residual": [ilc_trace[i].tolist() for i in range(L - n_save, L)],
        "rewards": reward_trace[L - n_save:],
        "kan_err": kan_err_trace[L - n_save:],
    }


def compute_aligned_stats(failures):
    """Align all failures at tau=0, compute per-step means and stds.

    Returns dict of (tau, mean, std) for each metric.
    """
    if not failures:
        return None
    n = len(failures)
    # Find max tau (longest pre-termination window)
    max_tau = max(len(f["obs"]) for f in failures)

    metrics = {
        "z": [], "angle": [], "vx": [], "vz": [],
        "thigh": [], "leg": [], "foot": [],
        "vthigh": [], "vleg": [], "vfoot": [],
        "a0": [], "a1": [], "a2": [],
        "a_tr0": [], "a_tr1": [], "a_tr2": [],
        "ilc0": [], "ilc1": [], "ilc2": [],
        "a_sat": [],  # fraction of actions at ±1
        "kan_err": [], "reward": [],
        # Health margins
        "m_z": [], "m_angle": [],
    }

    for tau in range(max_tau):
        for key in metrics:
            vals = []
            for f in failures:
                idx = len(f["obs"]) - max_tau + tau
                if idx >= 0:
                    obs = f["obs"][idx]
                    if key == "z": vals.append(obs[IDX_Z])
                    elif key == "angle": vals.append(obs[IDX_ANGLE])
                    elif key == "vx": vals.append(obs[IDX_VX])
                    elif key == "vz": vals.append(obs[IDX_VZ])
                    elif key == "thigh": vals.append(obs[IDX_THIGH])
                    elif key == "leg": vals.append(obs[IDX_LEG])
                    elif key == "foot": vals.append(obs[IDX_FOOT])
                    elif key == "vthigh": vals.append(obs[IDX_VTHIGH])
                    elif key == "vleg": vals.append(obs[IDX_VLEG])
                    elif key == "vfoot": vals.append(obs[IDX_VFOOT])
                    elif key == "a0": vals.append(f["actions"][idx][0])
                    elif key == "a1": vals.append(f["actions"][idx][1])
                    elif key == "a2": vals.append(f["actions"][idx][2])
                    elif key == "a_tr0": vals.append(f["transport"][idx][0])
                    elif key == "a_tr1": vals.append(f["transport"][idx][1])
                    elif key == "a_tr2": vals.append(f["transport"][idx][2])
                    elif key == "ilc0": vals.append(f["ilc_residual"][idx][0])
                    elif key == "ilc1": vals.append(f["ilc_residual"][idx][1])
                    elif key == "ilc2": vals.append(f["ilc_residual"][idx][2])
                    elif key == "a_sat":
                        a = f["actions"][idx]
                        vals.append(float(np.mean(np.abs(a) > 0.99)))
                    elif key == "kan_err": vals.append(f["kan_err"][idx])
                    elif key == "reward": vals.append(f["rewards"][idx])
                    elif key == "m_z":
                        vals.append(max(0, obs[IDX_Z] - Z_MIN))
                    elif key == "m_angle":
                        vals.append(max(0, ANGLE_MAX - abs(obs[IDX_ANGLE])))

            if vals:
                metrics[key].append((tau - max_tau + 1, float(np.mean(vals)), float(np.std(vals))))
            else:
                metrics[key].append((tau - max_tau + 1, float('nan'), float('nan')))

    return metrics


def diagnose_failure_mode(aligned):
    """Analyze failure mode from aligned trajectories."""
    if aligned is None:
        return "no_data"

    # Check z margin trend (last 20 steps)
    mz_vals = [m[1] for m in aligned["m_z"][-20:] if not np.isnan(m[1])]
    ma_vals = [m[1] for m in aligned["m_angle"][-20:] if not np.isnan(m[1])]
    z_vals = [m[1] for m in aligned["z"][-20:] if not np.isnan(m[1])]
    a_vals = [m[1] for m in aligned["angle"][-20:] if not np.isnan(m[1])]

    # Check which margin degrades first
    if len(mz_vals) >= 5 and len(ma_vals) >= 5:
        z_trend = mz_vals[-1] - mz_vals[0]  # negative = decreasing
        a_trend = np.mean(np.abs(a_vals[-5:])) - np.mean(np.abs(a_vals[:5]))  # positive = angle growing

        if a_trend > 0.02 and z_trend > -0.05:
            return "angle_first"  # angle goes bad before z drops
        elif z_trend < -0.1:
            return "z_first"  # height drops first
        else:
            return "simultaneous"  # both degrade together
    return "undetermined"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=1811)
    a = p.parse_args()

    print("=" * 72)
    print("E2: Pre-termination Trajectory Alignment")
    print(f"  Recording last {PRE_STEPS} steps before termination")
    print(f"  {N_EPISODES} episodes for Transport and KAN-ILC")
    print("=" * 72)

    # ── Load ────────────────────────────────────────────────────────────
    print("\n[1/3] Loading...", flush=True)
    t0 = time.time()
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
    print(f"  Load time: {time.time()-t0:.1f}s", flush=True)

    shift = SHIFTS["friction_070"]
    env_factory = lambda s=0: make_shifted_env(shift, s, "hopper")()
    ref_len = 64

    # ── ILC warmup ──────────────────────────────────────────────────────
    print(f"\n[2/3] ILC warmup ({N_ILC_WARMUP} trials, self-reference)...", flush=True)
    t0 = time.time()
    ro, ra, rl = build_reference(sp, basis, sc, tc, shift, DEVICE, a.seed, n_ep=10)
    # c=0.00: self-reference (no vx modification)
    ref_000 = ro.copy()

    a_dim = basis.action_dim
    u_table = np.zeros((N_PHASE, a_dim), dtype=np.float32)

    for trial in range(N_ILC_WARMUP):
        env = env_factory(a.seed + trial * 100)
        obs, _ = env.reset(seed=a.seed + trial * 100)
        step = 0
        detector = CycleDetector(); detector.reset()
        obs_trace, act_trace, td_steps = [], [], []

        while True:
            s_t = torch.as_tensor(obs, device=DEVICE, dtype=torch.float32).unsqueeze(0)
            nominal = sp.action(s_t)
            s_eff = sc.acceleration(basis, s_t, nominal)
            a_tr = tc.transport_action(basis, s_t, desired_effect=s_eff,
                                       nominal_action=nominal, regularization=1e-2
                                       ).clamp(-1, 1).squeeze(0).cpu().numpy()
            if len(td_steps) > 0:
                phase = min((step - td_steps[-1]) / ref_len, 0.99)
                u_ff = u_table[min(int(phase * N_PHASE), N_PHASE - 1)]
                a_final = np.clip(a_tr + u_ff, -1, 1)
            else:
                a_final = a_tr
            next_obs, _, terminated, truncated, _ = env.step(a_final)
            obs_trace.append(obs.copy()); act_trace.append(a_final.copy())
            if detector.update(env): td_steps.append(step)
            obs = next_obs; step += 1
            if terminated or truncated: break
        env.close()

        cycle_data = None
        for i in range(len(td_steps) - 1):
            s, e = td_steps[i], td_steps[i + 1]
            if 25 <= (e - s) <= 70:
                cycle_data = {"obs": np.array(obs_trace[s:e]),
                              "actions": np.array(act_trace[s:e])}; break
        if cycle_data is not None:
            u_table, _ = ilc_update_with_bt(u_table, cycle_data["obs"], ref_000,
                lambda args, s_np: get_kan_bt(basis, tc,
                    torch.as_tensor(s_np, device=DEVICE, dtype=torch.float32).unsqueeze(0)),
                None, eta=ETA)

    print(f"  ILC converged: |u|={np.linalg.norm(u_table):.3f} ({time.time()-t0:.1f}s)", flush=True)

    # ── Record failing episodes ─────────────────────────────────────────
    print(f"\n[3/3] Recording {N_EPISODES} episodes for Transport and KAN-ILC...", flush=True)

    all_failures = {"transport": [], "kan_ilc": []}

    for i in range(N_EPISODES):
        ep_seed = 9000 + i * 100

        # Transport
        tr_fail = record_failing_episode(env_factory, sp, basis, sc, tc, ep_seed,
                                          "transport", u_table=None, ref_len=ref_len)
        if tr_fail: all_failures["transport"].append(tr_fail)

        # KAN-ILC
        ilc_fail = record_failing_episode(env_factory, sp, basis, sc, tc, ep_seed,
                                           "kan_ilc", u_table=u_table, ref_len=ref_len)
        if ilc_fail: all_failures["kan_ilc"].append(ilc_fail)

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{N_EPISODES}: Transport={len(all_failures['transport'])} "
                  f"KAN-ILC={len(all_failures['kan_ilc'])} failures", flush=True)

    # ── Analysis ────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("  E2 Analysis Results")
    print(f"  Transport failures: {len(all_failures['transport'])}/{N_EPISODES}")
    print(f"  KAN-ILC failures:   {len(all_failures['kan_ilc'])}/{N_EPISODES}")

    tr_aligned = compute_aligned_stats(all_failures["transport"])
    ilc_aligned = compute_aligned_stats(all_failures["kan_ilc"])

    # Failure mode diagnosis
    tr_mode = diagnose_failure_mode(tr_aligned) if tr_aligned else "no_data"
    ilc_mode = diagnose_failure_mode(ilc_aligned) if ilc_aligned else "no_data"
    print(f"\n  Failure mode: Transport={tr_mode}, KAN-ILC={ilc_mode}")

    # Key metrics at tau=-1 (last step before termination) and tau=-20 (early warning)
    for label, aligned, fails in [("Transport", tr_aligned, all_failures["transport"]),
                                    ("KAN-ILC", ilc_aligned, all_failures["kan_ilc"])]:
        if aligned is None:
            print(f"\n  {label}: No failure data")
            continue

        print(f"\n  ── {label} ({len(fails)} failures) ──")

        # Health margins at key taus
        for tau_name, tau_idx in [("tau=-1 (termination)", -1), ("tau=-10", -10),
                                    ("tau=-25", -25), ("tau=-50", -50)]:
            idx = tau_idx
            if idx < -len(aligned["z"]): continue
            mz = aligned["m_z"][idx]
            ma = aligned["m_angle"][idx]
            z = aligned["z"][idx]
            a = aligned["angle"][idx]
            vx = aligned["vx"][idx]
            kan = aligned["kan_err"][idx]
            a_sat = aligned["a_sat"][idx]
            ilc0 = aligned.get("ilc0", [(0,0,0)])[idx] if aligned.get("ilc0") else (0,0)

            if not np.isnan(mz[1]):
                print(f"    {tau_name:>20s}: z={z[1]:.4f}+/-{z[2]:.4f}  "
                      f"angle={a[1]:.4f}+/-{a[2]:.4f}  "
                      f"vx={vx[1]:.3f}+/-{vx[2]:.4f}  "
                      f"m_z={mz[1]:.3f}  m_a={ma[1]:.3f}  "
                      f"kan_err={kan[1]:.3f}  a_sat={a_sat[1]:.1%}  "
                      f"|ilc|={np.mean([x[1] for x in aligned['ilc0'][-5:]]) if aligned.get('ilc0') else 0:.4f}")

        # ILC residual trend in last 20 steps
        if aligned.get("ilc0"):
            ilc_vals = [m[1] for m in aligned["ilc0"] if not np.isnan(m[1])]
            ilc_trend = ilc_vals[-1] - ilc_vals[0] if len(ilc_vals) >= 5 else 0
            print(f"    ILC residual trend (last 20): {ilc_trend:+.4f} "
                  f"(start={ilc_vals[0]:.4f} end={ilc_vals[-1]:.4f})")

        # KAN error trend
        kan_vals = [m[1] for m in aligned["kan_err"] if not np.isnan(m[1])]
        if len(kan_vals) >= 5:
            print(f"    KAN error: start={kan_vals[0]:.3f} end={kan_vals[-1]:.3f} "
                  f"(trend={kan_vals[-1]-kan_vals[0]:+.3f})")

        # Action saturation trend
        sat_vals = [m[1] for m in aligned["a_sat"] if not np.isnan(m[1])]
        if len(sat_vals) >= 5:
            print(f"    Action sat: start={sat_vals[0]:.1%} end={sat_vals[-1]:.1%}")

    # ── Comparison: Transport vs KAN-ILC failure signatures ─────────────
    print(f"\n  {'='*60}")
    print("  Comparison: Transport vs KAN-ILC at tau=-5 (just before termination)")

    for metric in ["z", "angle", "vx", "vz", "m_z", "m_angle", "kan_err", "a_sat"]:
        tr_val = tr_aligned[metric][-5][1] if tr_aligned and not np.isnan(tr_aligned[metric][-5][1]) else float('nan')
        ilc_val = ilc_aligned[metric][-5][1] if ilc_aligned and not np.isnan(ilc_aligned[metric][-5][1]) else float('nan')
        delta = ilc_val - tr_val if not (np.isnan(tr_val) or np.isnan(ilc_val)) else float('nan')
        print(f"    {metric:>12s}: Transport={tr_val:.4f}  KAN-ILC={ilc_val:.4f}  delta={delta:+.4f}")

    # ── Questions ───────────────────────────────────────────────────────
    print(f"\n  Key Questions:")
    # Q1: Gradual or sudden?
    if tr_aligned:
        z_last10 = [m[1] for m in tr_aligned["z"][-10:] if not np.isnan(m[1])]
        if len(z_last10) >= 5:
            z_drop = z_last10[-1] - z_last10[0]
            print(f"  Q1: z drops by {z_drop:.3f} in last 10 steps → "
                  f"{'GRADUAL instability' if abs(z_drop) > 0.02 else 'SUDDEN contact failure'}")

    # Q2: Does ILC residual grow before failure?
    print(f"  Q2: ILC residual growth before failure — see trend above")

    # Q3: Does KAN distort before failure?
    print(f"  Q3: KAN error trend before failure — see trend above")

    # Q4/Q5: Same failure mode?
    print(f"  Q4: Transport mode={tr_mode}, KAN-ILC mode={ilc_mode}")

    # ── Save ────────────────────────────────────────────────────────────
    json.dump({
        "config": {"n_episodes": N_EPISODES, "pre_steps": PRE_STEPS},
        "transport": {"n_failures": len(all_failures["transport"]),
                       "aligned": tr_aligned, "mode": tr_mode},
        "kan_ilc": {"n_failures": len(all_failures["kan_ilc"]),
                     "aligned": ilc_aligned, "mode": ilc_mode},
    }, open("results/e2_pre_termination.json", "w"), indent=2)
    print(f"\n  Saved to results/e2_pre_termination.json")


if __name__ == "__main__":
    main()
