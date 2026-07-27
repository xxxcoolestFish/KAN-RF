"""E3: Oracle rescue — intervene when health margins drop.

Tests whether KAN-ILC's main gap is lack of recovery capability.
Triggers rescue when z-margin or angle-margin drops below threshold,
switching to: Source-policy, Transport (no ILC), or Target-Oracle.

Also tests fixed-step rescue at tau=-10,-20,-40 before expected termination.
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
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

DEVICE = torch.device("cuda")
N_EPISODES = 20
N_ILC_WARMUP = 25; ETA = 0.3; REF_LEN = 64
M_Z_THRESHOLD = 0.10  # trigger rescue when z margin < 0.10


def run_rescue_episode(env_factory, sp, basis, sc, tc, seed, u_table,
                        rescue_type, rescue_fn, mz_threshold):
    """Run KAN-ILC with rescue intervention.

    rescue_type: "none", "source", "transport", "oracle", "fixed_tau_N"
    rescue_fn: function(obs) -> action, only needed for source/oracle
    """
    env = env_factory(seed)
    obs, _ = env.reset(seed=seed)
    step = 0; total_r = 0.0
    detector = CycleDetector(); detector.reset()
    td_steps = []

    rescue_active = False
    rescue_steps = 0
    n_interventions = 0
    intervention_steps = []
    post_rescue_survived = 0

    while True:
        s_t = torch.as_tensor(obs, device=DEVICE, dtype=torch.float32).unsqueeze(0)
        nominal = sp.action(s_t)
        s_eff = sc.acceleration(basis, s_t, nominal)
        a_tr = tc.transport_action(basis, s_t, desired_effect=s_eff,
                                   nominal_action=nominal, regularization=1e-2
                                   ).clamp(-1, 1).squeeze(0).cpu().numpy()

        # Health margins
        z, angle = obs[0], obs[1]
        m_z = max(0, z - 0.7)
        m_angle = max(0, 0.2 - abs(angle))

        # Decide rescue — parse trigger from condition name
        if rescue_type.startswith("fixed_tau"):
            tau = int(rescue_type.split("_")[-1])
            expected_T = 185
            should_rescue = (not rescue_active and step > expected_T - tau)
        elif rescue_type == "none":
            should_rescue = False
        else:
            # Parse threshold from name: "oracle_mz010" -> 0.10, "source_mz010" -> 0.10
            thresh_str = rescue_type.split("_mz")[-1] if "_mz" in rescue_type else "010"
            thresh = float(thresh_str) / 100  # "010" -> 0.10, "015" -> 0.15
            should_rescue = (not rescue_active and (m_z < thresh or m_angle < 0.02))

        if should_rescue and rescue_type != "none":
            rescue_active = True
            n_interventions += 1
            intervention_steps.append(step)

        # Action selection
        if rescue_active and (rescue_type.startswith("source")):
            a_final = rescue_fn(s_t).squeeze(0).cpu().numpy()
            rescue_steps += 1
        elif rescue_active and (rescue_type.startswith("oracle") or rescue_type.startswith("fixed_tau")):
            obs_norm = ((obs - rescue_fn[1].obs_rms.mean) /
                        (rescue_fn[1].obs_rms.var + 1e-8) ** 0.5)
            a_final, _ = rescue_fn[0].predict(obs_norm, deterministic=True)
            rescue_steps += 1
        elif rescue_active and rescue_type == "transport":
            a_final = a_tr  # pure Transport, no ILC
            rescue_steps += 1
        else:
            # Normal KAN-ILC
            u_ff = np.zeros_like(a_tr)
            if u_table is not None and len(td_steps) > 0:
                phase = min((step - td_steps[-1]) / REF_LEN, 0.99)
                u_ff = u_table[min(int(phase * N_PHASE), N_PHASE - 1)]
            a_final = np.clip(a_tr + u_ff, -1, 1)

        next_obs, reward, terminated, truncated, info = env.step(a_final)
        total_r += float(reward)

        if detector.update(env): td_steps.append(step)
        obs = next_obs; step += 1
        if terminated or truncated:
            if rescue_active:
                post_rescue_survived = step - intervention_steps[-1] if intervention_steps else 0
            break

    env.close()
    return {"total_return": float(total_r), "length": step,
            "fell": bool(terminated), "truncated": bool(truncated),
            "n_interventions": n_interventions,
            "rescue_steps": rescue_steps,
            "intervention_step": intervention_steps[0] if intervention_steps else None,
            "post_rescue_survived": post_rescue_survived,
            "rescue_type": rescue_type}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=1811)
    p.add_argument("--n-episodes", type=int, default=N_EPISODES)
    a = p.parse_args()

    print("=" * 72)
    print("E3: Oracle Rescue Experiment")
    print(f"  Trigger: m_z < {M_Z_THRESHOLD} or m_theta < 0.02")
    print(f"  {a.n_episodes} episodes per condition")
    print("=" * 72)

    # ── Load ────────────────────────────────────────────────────────────
    print("\n[1/4] Loading...", flush=True)
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

    # Oracle for rescue
    oracle_model = PPO.load("results/hopper_friction_oracle_seed1811.zip", device=DEVICE)
    oracle_env = DummyVecEnv([make_shifted_env(SHIFTS["friction_070"], a.seed, "hopper")])
    oracle_norm = VecNormalize.load("results/hopper_friction_oracle_norm_seed1811.pkl", oracle_env)
    oracle_norm.training = False; oracle_norm.norm_reward = False; oracle_env.close()

    shift = SHIFTS["friction_070"]
    env_factory = lambda s=0: make_shifted_env(shift, s, "hopper")()
    print(f"  Load time: {time.time()-t0:.1f}s", flush=True)

    # ── ILC warmup ──────────────────────────────────────────────────────
    print(f"\n[2/4] ILC warmup ({N_ILC_WARMUP} trials, self-reference)...", flush=True)
    t0 = time.time()
    ro, ra, rl = build_reference(sp, basis, sc, tc, shift, DEVICE, a.seed, n_ep=10)
    ref_000 = ro.copy()
    a_dim = basis.action_dim
    u_table = np.zeros((N_PHASE, a_dim), dtype=np.float32)

    for trial in range(N_ILC_WARMUP):
        env = env_factory(a.seed + trial * 100)
        obs, _ = env.reset(seed=a.seed + trial * 100)
        step = 0; detector = CycleDetector(); detector.reset()
        obs_trace, act_trace, td_steps = [], [], []
        while True:
            s_t = torch.as_tensor(obs, device=DEVICE, dtype=torch.float32).unsqueeze(0)
            nominal = sp.action(s_t); s_eff = sc.acceleration(basis, s_t, nominal)
            a_tr = tc.transport_action(basis, s_t, desired_effect=s_eff,
                                       nominal_action=nominal, regularization=1e-2
                                       ).clamp(-1, 1).squeeze(0).cpu().numpy()
            if len(td_steps) > 0:
                phase = min((step - td_steps[-1]) / REF_LEN, 0.99)
                a_final = np.clip(a_tr + u_table[min(int(phase * N_PHASE), N_PHASE - 1)], -1, 1)
            else: a_final = a_tr
            next_obs, _, terminated, truncated, _ = env.step(a_final)
            obs_trace.append(obs.copy()); act_trace.append(a_final.copy())
            if detector.update(env): td_steps.append(step)
            obs = next_obs; step += 1
            if terminated or truncated: break
        env.close()
        cd = None
        for i in range(len(td_steps)-1):
            s, e = td_steps[i], td_steps[i+1]
            if 25 <= (e-s) <= 70: cd = {"obs": np.array(obs_trace[s:e]), "actions": np.array(act_trace[s:e])}; break
        if cd is not None:
            u_table, _ = ilc_update_with_bt(u_table, cd["obs"], ref_000,
                lambda args, s_np: get_kan_bt(basis, tc,
                    torch.as_tensor(s_np, device=DEVICE, dtype=torch.float32).unsqueeze(0)),
                None, eta=ETA)
    print(f"  Converged: |u|={np.linalg.norm(u_table):.3f} ({time.time()-t0:.1f}s)", flush=True)

    # ── Rescue conditions ───────────────────────────────────────────────
    conditions = [
        ("none", "No rescue (KAN-ILC baseline)"),
        ("oracle_mz010", "Oracle at m_z < 0.10"),
        ("oracle_mz015", "Oracle at m_z < 0.15"),
        ("source_mz010", "Source policy at m_z < 0.10"),
        ("transport_mz010", "Disable ILC at m_z < 0.10"),
        ("fixed_tau_20", "Oracle at T-20 (early warning)"),
        ("fixed_tau_40", "Oracle at T-40 (early warning)"),
    ]

    # ── Run ─────────────────────────────────────────────────────────────
    print(f"\n[3/4] Running {a.n_episodes} episodes x {len(conditions)} conditions...", flush=True)

    all_results = {}
    for rescue_type, label in conditions:
        sys.stdout.write(f"\n  {label}: "); sys.stdout.flush()
        t0 = time.time()
        episodes = []
        for i in range(a.n_episodes):
            ep_seed = 9000 + i * 100
            # Map condition to rescue function
            if rescue_type.startswith("fixed_tau") or rescue_type.startswith("oracle"):
                rf = (oracle_model, oracle_norm)
            elif rescue_type.startswith("source"):
                rf = sp.action
            else:
                rf = None
            result = run_rescue_episode(env_factory, sp, basis, sc, tc, ep_seed,
                u_table, rescue_type, rf, M_Z_THRESHOLD)
            episodes.append(result)

        returns = [r["total_return"] for r in episodes]
        lengths = [r["length"] for r in episodes]
        fell = sum(1 for r in episodes if r["fell"])
        rescued = sum(1 for r in episodes if r["n_interventions"] > 0)
        rescue_dur = [r["rescue_steps"] for r in episodes if r["n_interventions"] > 0]
        post_rescue = [r["post_rescue_survived"] for r in episodes if r["post_rescue_survived"] > 0]
        recovered = sum(1 for r in episodes if r["truncated"])  # survived to time limit

        print(f"R={np.mean(returns):.1f}+/-{np.std(returns):.0f}  "
              f"T={np.mean(lengths):.1f}  "
              f"Fell={fell}/{a.n_episodes}  "
              f"Rescued={rescued}  "
              f"Rescue_dur={np.mean(rescue_dur):.1f}s" if rescue_dur else f"Rescued={rescued}",
              f"  Recovered={recovered}",
              f"({time.time()-t0:.1f}s)", flush=True)

        all_results[rescue_type] = {
            "label": label,
            "mean_return": float(np.mean(returns)), "std_return": float(np.std(returns)),
            "mean_length": float(np.mean(lengths)), "std_length": float(np.std(lengths)),
            "fell_count": fell, "fell_rate": float(fell / a.n_episodes),
            "n_rescued": rescued, "n_recovered": recovered,
            "mean_rescue_duration": float(np.mean(rescue_dur)) if rescue_dur else 0,
            "mean_post_rescue": float(np.mean(post_rescue)) if post_rescue else 0,
            "returns": [float(x) for x in returns],
            "lengths": [int(x) for x in lengths],
        }

    # ── Summary ─────────────────────────────────────────────────────────
    print(f"\n[4/4] E3 Results", flush=True)
    print("=" * 72)
    print(f"  {'Condition':35s} {'Return':>8s} {'Length':>8s} {'Fell%':>7s} "
          f"{'Rescued':>8s} {'Recovered':>10s} {'RescueDur':>10s}")
    print(f"  {'-'*80}")

    baseline_R = all_results.get("none", {}).get("mean_return", 571)
    baseline_T = all_results.get("none", {}).get("mean_length", 183)

    for key, r in all_results.items():
        print(f"  {r['label']:35s} {r['mean_return']:>8.1f} {r['mean_length']:>8.1f} "
              f"{r['fell_rate']:>6.0%} {r['n_rescued']:>8d} {r['n_recovered']:>10d} "
              f"{r['mean_rescue_duration']:>10.1f}")

    # ── Verdict ─────────────────────────────────────────────────────────
    oracle_rescue = all_results.get("oracle", {})
    transport_rescue = all_results.get("transport", {})
    source_rescue = all_results.get("source", {})

    print(f"\n  Analysis:")
    oracle_delta = oracle_rescue.get("mean_return", 0) - baseline_R
    print(f"    Oracle rescue delta: {oracle_delta:+.1f} return, "
          f"+{oracle_rescue.get('mean_length', 0) - baseline_T:.1f} steps")

    if oracle_rescue.get("n_recovered", 0) > 0:
        print(f"    Oracle can recover {oracle_rescue['n_recovered']}/{a.n_episodes} episodes!")
    else:
        print(f"    Oracle cannot fully recover any episode (0/{a.n_episodes} survive to 1000)")

    trans_delta = transport_rescue.get("mean_return", 0) - baseline_R
    if abs(trans_delta) < 5:
        print(f"    Disabling ILC residual has negligible effect ({trans_delta:+.1f}) — ILC residual is near-zero")

    json.dump({"config": {"n_episodes": a.n_episodes, "m_z_threshold": M_Z_THRESHOLD},
               "results": all_results},
              open("results/e3_oracle_rescue.json", "w"), indent=2)
    print(f"\n  Saved to results/e3_oracle_rescue.json")


if __name__ == "__main__":
    main()
