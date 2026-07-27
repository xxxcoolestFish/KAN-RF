"""E0: Unified evaluation — all methods under identical conditions.

Evaluates 5 methods on Hopper-v5:
  1. Source policy on source physics
  2. Source policy on friction physics
  3. Transport on friction physics
  4. KAN-ILC (c=0.10) on friction physics
  5. Target Oracle policy on friction physics

All use identical: env config, eval seeds, episode count.
Records: R, T, R_per_step per episode + per-step data for E1 decomposition.
"""
import sys, os, numpy as np, torch, argparse, json, time
# Fix Windows GBK encoding issue with Unicode characters
if sys.platform == 'win32':
    try: sys.stdout.reconfigure(encoding='utf-8')
    except: pass
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.prescreen_hopper_physics_shifts import SHIFTS, make_shifted_env
from scripts.validate_hopper_joint_online_adaptation import FrozenSourcePolicy, load_cognition
from scripts.diagnose_hopper_pullback_effect import fit_distilled_source_counterfactual_context, load_source_twin
from scripts.ilc1_ablation import *
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

DEVICE = torch.device("cuda")
N_EVAL_EPISODES = 50
EVAL_SEED_BASE = 9000
VX_IDX = 5  # Hopper-v5: obs[5] = velocity of x-coordinate of torso
N_ILC_WARMUP = 25  # ILC convergence trials before eval
ETA = 0.3

# Hopper-v5 observation (from Gymnasium source):
#   obs[0]=z  obs[1]=angle(rad)  obs[2]=thigh  obs[3]=leg  obs[4]=foot
#   obs[5]=vx  obs[6]=vz  obs[7]=vang  obs[8]=vthigh  obs[9]=vleg  obs[10]=vfoot


def hopper_reward_components_from_info(action, info):
    """Extract exact Hopper-v5 reward decomposition from env.step() info dict.

    Hopper-v5 info contains: 'reward_forward', 'reward_survive', 'reward_ctrl' (negative).
    Returns (forward_r, healthy_r, ctrl_cost, total).
    """
    fwd = float(info.get('reward_forward', 0))
    healthy = float(info.get('reward_survive', 0))
    ctrl = -float(info.get('reward_ctrl', 0))  # info has negative ctrl
    return fwd, healthy, ctrl, fwd + healthy - ctrl


def run_episode_source(env_factory, source_policy, seed):
    """Run source policy on given env. Returns per-step data."""
    env = env_factory()
    obs, _ = env.reset(seed=seed)
    step = 0
    total_r = 0.0
    per_step = {"obs": [], "actions": [], "next_obs": [],
                "rewards": [], "forward_r": [], "healthy_r": [], "ctrl_cost": []}

    while True:
        s_t = torch.as_tensor(obs, device=DEVICE, dtype=torch.float32).unsqueeze(0)
        a = source_policy.action(s_t).squeeze(0).cpu().numpy()
        next_obs, reward, terminated, truncated, info = env.step(a)
        fwd_r, healthy_r, ctrl, _ = hopper_reward_components_from_info(a, info)

        per_step["obs"].append(obs.copy())
        per_step["actions"].append(a.copy())
        per_step["next_obs"].append(next_obs.copy())
        per_step["rewards"].append(float(reward))
        per_step["forward_r"].append(fwd_r)
        per_step["healthy_r"].append(healthy_r)
        per_step["ctrl_cost"].append(ctrl)

        total_r += float(reward)
        obs = next_obs
        step += 1
        if terminated or truncated:
            break

    env.close()
    return {"total_return": float(total_r), "length": step,
            "terminated": bool(terminated), "truncated": bool(truncated),
            "fell": bool(terminated),
            "per_step": per_step, "seed": seed}


def run_episode_transport(env_factory, sp, basis, sc, tc, seed, u_table=None):
    """Run Transport (±ILC) on friction env. Returns per-step data."""
    env = env_factory()
    obs, _ = env.reset(seed=seed)
    step = 0; total_r = 0.0
    per_step = {"obs": [], "actions": [], "next_obs": [],
                "rewards": [], "forward_r": [], "healthy_r": [], "ctrl_cost": [],
                "transport_actions": [], "ilc_residuals": []}
    detector = CycleDetector(); detector.reset()
    td_steps = []

    while True:
        s_t = torch.as_tensor(obs, device=DEVICE, dtype=torch.float32).unsqueeze(0)
        nominal = sp.action(s_t)
        s_eff = sc.acceleration(basis, s_t, nominal)
        a_tr = tc.transport_action(basis, s_t, desired_effect=s_eff,
                                   nominal_action=nominal, regularization=1e-2
                                   ).clamp(-1, 1).squeeze(0).cpu().numpy()

        u_ff = np.zeros_like(a_tr)
        if u_table is not None and len(td_steps) > 0:
            # Use nominal cycle length = 64 (Transport reference)
            ref_len = 64
            phase = min((step - td_steps[-1]) / ref_len, 0.99)
            u_ff = u_table[min(int(phase * N_PHASE), N_PHASE - 1)]
        a_final = np.clip(a_tr + u_ff, -1, 1)

        next_obs, reward, terminated, truncated, info = env.step(a_final)
        fwd_r, healthy_r, ctrl, _ = hopper_reward_components_from_info(a_final, info)

        per_step["obs"].append(obs.copy())
        per_step["actions"].append(a_final.copy())
        per_step["next_obs"].append(next_obs.copy())
        per_step["rewards"].append(float(reward))
        per_step["forward_r"].append(fwd_r)
        per_step["healthy_r"].append(healthy_r)
        per_step["ctrl_cost"].append(ctrl)
        per_step["transport_actions"].append(a_tr.copy())
        per_step["ilc_residuals"].append(u_ff.copy())

        total_r += float(reward)
        if detector.update(env):
            td_steps.append(step)
        obs = next_obs; step += 1
        if terminated or truncated:
            break

    env.close()
    return {"total_return": float(total_r), "length": step,
            "terminated": bool(terminated), "truncated": bool(truncated),
            "fell": bool(terminated),
            "per_step": per_step, "seed": seed,
            "n_touchdowns": len(td_steps)}


def run_episode_oracle(env_factory, oracle_model, oracle_norm, seed):
    """Run target oracle PPO on friction env."""
    env = env_factory()
    obs, _ = env.reset(seed=seed)
    step = 0; total_r = 0.0
    per_step = {"obs": [], "actions": [], "next_obs": [],
                "rewards": [], "forward_r": [], "healthy_r": [], "ctrl_cost": []}

    while True:
        obs_norm = ((obs - oracle_norm.obs_rms.mean) /
                    (oracle_norm.obs_rms.var + 1e-8) ** 0.5)
        a, _ = oracle_model.predict(obs_norm, deterministic=True)
        next_obs, reward, terminated, truncated, info = env.step(a)
        fwd_r, healthy_r, ctrl, _ = hopper_reward_components_from_info(a, info)

        per_step["obs"].append(obs.copy())
        per_step["actions"].append(a.copy())
        per_step["next_obs"].append(next_obs.copy())
        per_step["rewards"].append(float(reward))
        per_step["forward_r"].append(fwd_r)
        per_step["healthy_r"].append(healthy_r)
        per_step["ctrl_cost"].append(ctrl)

        total_r += float(reward)
        obs = next_obs; step += 1
        if terminated or truncated:
            break

    env.close()
    return {"total_return": float(total_r), "length": step,
            "terminated": bool(terminated), "truncated": bool(truncated),
            "fell": bool(terminated),
            "per_step": per_step, "seed": seed}


def make_vx_ref(ref_obs, ref_act, delta):
    r = ref_obs.copy(); r[:, VX_IDX] += delta; return r


def compute_ilc_tracking_error(cycle_obs, ref_obs):
    """Compute per-phase tracking error for a cycle."""
    L = cycle_obs.shape[0]; s_dim = cycle_obs.shape[1]
    t_old = np.linspace(0, 1, L); t_new = np.linspace(0, 1, N_PHASE)
    aligned = np.zeros((N_PHASE, s_dim))
    for d in range(s_dim):
        aligned[:, d] = np.interp(t_new, t_old, cycle_obs[:, d])
    td_ = [0, 2, 6, 3, 4]; tw = np.array([2., 1., 1., .5, .5])
    err = sum(np.sum((tw * (ref_obs[i, td_] - aligned[i, td_])) ** 2)
              for i in range(N_PHASE))
    return float(np.sqrt(err / N_PHASE)), aligned


def run_ilc_with_eval_recording(sp, basis, sc, tc, shift, seed, n_warmup,
                                 ref_obs, ref_act, ref_len, eval_seed_base,
                                 n_eval=N_EVAL_EPISODES, eta=ETA):
    """Run ILC warmup + eval, returning per-step eval data.

    Warmup: n_warmup trials with ILC updates (converges u_table).
    Eval: n_eval trials with FROZEN u_table, identical seeds to other methods.
    """
    a_dim = basis.action_dim
    u_table = np.zeros((N_PHASE, a_dim), dtype=np.float32)
    warmup_results = []

    # --- Warmup phase ---
    for trial in range(n_warmup):
        env = make_shifted_env(shift, seed + trial * 100, "hopper")()
        obs, _ = env.reset(seed=seed + trial * 100)
        total_r = 0.0; step = 0
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

            next_obs, reward, terminated, truncated, info = env.step(a_final)
            total_r += float(reward)
            obs_trace.append(obs.copy()); act_trace.append(a_final.copy())
            if detector.update(env): td_steps.append(step)
            obs = next_obs; step += 1
            if terminated or truncated: break

        env.close()

        # Extract cycle + ILC update
        cycle_data = None
        for i in range(len(td_steps) - 1):
            s, e = td_steps[i], td_steps[i + 1]
            if 25 <= (e - s) <= 70:
                cycle_data = {"obs": np.array(obs_trace[s:e]),
                              "actions": np.array(act_trace[s:e])}; break

        if cycle_data is not None:
            u_table, err = ilc_update_with_bt(
                u_table, cycle_data["obs"], ref_obs,
                lambda a, s_np: get_kan_bt(basis, tc,
                    torch.as_tensor(s_np, device=DEVICE, dtype=torch.float32).unsqueeze(0)),
                None, eta=eta)
            warmup_results.append({"trial": trial, "return": float(total_r),
                                    "error": float(err), "|u|": float(np.linalg.norm(u_table))})
        else:
            warmup_results.append({"trial": trial, "return": float(total_r),
                                    "error": None, "|u|": float(np.linalg.norm(u_table))})

    # --- Eval phase (frozen u_table) ---
    eval_results = []
    for i in range(n_eval):
        ep_seed = eval_seed_base + i * 100
        result = run_episode_transport(
            lambda s=ep_seed: make_shifted_env(shift, s, "hopper")(),
            sp, basis, sc, tc, ep_seed, u_table=u_table)
        eval_results.append(result)

    return {"warmup": warmup_results, "eval": eval_results}


def summarize(results, label):
    """Print summary stats for a list of episode results."""
    returns = [r["total_return"] for r in results]
    lengths = [r["length"] for r in results]
    fell_count = sum(1 for r in results if r["fell"])
    # Per-step reward
    per_step_rs = [r["total_return"] / max(r["length"], 1) for r in results]
    # First-100 return
    first100 = []
    for r in results:
        if r.get("per_step") and len(r["per_step"]["rewards"]) >= 100:
            first100.append(sum(r["per_step"]["rewards"][:100]))
        elif r.get("per_step"):
            first100.append(sum(r["per_step"]["rewards"]))
    # Last-50 return
    last50 = []
    for r in results:
        if r.get("per_step") and len(r["per_step"]["rewards"]) >= 50:
            last50.append(sum(r["per_step"]["rewards"][-50:]))

    print(f"  {label:35s}: R={np.mean(returns):8.1f}+/-{np.std(returns):5.0f}  "
          f"T={np.mean(lengths):7.1f}+/-{np.std(lengths):5.0f}  "
          f"R/step={np.mean(per_step_rs):.4f}  "
          f"Fell={fell_count}/{len(results)} ({fell_count/len(results):.0%})  "
          f"F100={np.mean(first100):.1f}  L50={np.mean(last50):.1f}" if last50 else "")
    return {"mean_return": float(np.mean(returns)), "std_return": float(np.std(returns)),
            "mean_length": float(np.mean(lengths)), "std_length": float(np.std(lengths)),
            "mean_per_step_r": float(np.mean(per_step_rs)),
            "fell_rate": float(fell_count / len(results)),
            "mean_first100": float(np.mean(first100)) if first100 else None,
            "mean_last50": float(np.mean(last50)) if last50 else None,
            "returns": [float(x) for x in returns],
            "lengths": [int(x) for x in lengths]}


def compute_survival_curve(episodes, max_T=1000):
    """S(t) = P(T >= t) for t = 0..max_T."""
    lengths = [r["length"] for r in episodes]
    S = np.ones(max_T + 1)
    for t in range(1, max_T + 1):
        S[t] = np.mean(np.array(lengths) >= t)
    return S.tolist()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=1811)
    p.add_argument("--n-eval", type=int, default=N_EVAL_EPISODES)
    p.add_argument("--n-ilc-warmup", type=int, default=N_ILC_WARMUP)
    a = p.parse_args()

    print("=" * 72)
    print("E0: Unified Evaluation — 5 methods × 50 episodes")
    print(f"  Env: Hopper-v5, max_episode_steps=1000, terminate_when_unhealthy=True")
    print(f"  ILC warmup: {a.n_ilc_warmup} trials, eval: {a.n_eval} episodes each")
    print("=" * 72)

    # ── Load components ─────────────────────────────────────────────────
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

    # KAN-ILC reference
    ro, ra, rl = build_reference(sp, basis, sc, tc, SHIFTS["friction_070"], DEVICE, a.seed, n_ep=10)
    ref_010 = make_vx_ref(ro, ra, 0.10)
    print(f"  Reference: {rl} steps, c=0.10 ref built", flush=True)

    # Oracle
    oracle_model = PPO.load("results/hopper_friction_oracle_seed1811.zip", device=DEVICE)
    oracle_env = DummyVecEnv([make_shifted_env(SHIFTS["friction_070"], a.seed, "hopper")])
    oracle_norm = VecNormalize.load("results/hopper_friction_oracle_norm_seed1811.pkl", oracle_env)
    oracle_norm.training = False; oracle_norm.norm_reward = False
    oracle_env.close()
    print(f"  Oracle loaded", flush=True)
    print(f"  Load time: {time.time()-t0:.1f}s", flush=True)

    # ── Run evaluations ─────────────────────────────────────────────────
    print(f"\n[2/3] Running {a.n_eval} episodes per method...", flush=True)

    all_results = {}
    survival_curves = {}

    # Factory functions for each physics
    def src_factory(): return make_shifted_env(SHIFTS["source"], 0, "hopper")()
    def frc_factory(): return make_shifted_env(SHIFTS["friction_070"], 0, "hopper")()

    # 1. Source on source
    print("\n  --- Source on Source ---", flush=True)
    t0 = time.time()
    src_on_src = [run_episode_source(src_factory, sp, EVAL_SEED_BASE + i * 100)
                  for i in range(a.n_eval)]
    all_results["source_on_source"] = summarize(src_on_src, "Source→Source")
    survival_curves["source_on_source"] = compute_survival_curve(src_on_src)
    print(f"    ({time.time()-t0:.1f}s)", flush=True)

    # 2. Source on friction
    print("\n  --- Source on Friction ---", flush=True)
    t0 = time.time()
    src_on_frc = [run_episode_source(frc_factory, sp, EVAL_SEED_BASE + i * 100)
                  for i in range(a.n_eval)]
    all_results["source_on_friction"] = summarize(src_on_frc, "Source→Friction")
    survival_curves["source_on_friction"] = compute_survival_curve(src_on_frc)
    print(f"    ({time.time()-t0:.1f}s)", flush=True)

    # 3. Transport on friction
    print("\n  --- Transport on Friction ---", flush=True)
    t0 = time.time()
    tr_on_frc = [run_episode_transport(frc_factory, sp, basis, sc, tc,
                                        EVAL_SEED_BASE + i * 100, u_table=None)
                 for i in range(a.n_eval)]
    all_results["transport"] = summarize(tr_on_frc, "Transport→Friction")
    survival_curves["transport"] = compute_survival_curve(tr_on_frc)
    print(f"    ({time.time()-t0:.1f}s)", flush=True)

    # 4. KAN-ILC (c=0.10)
    print(f"\n  --- KAN-ILC (c=0.10, {a.n_ilc_warmup} warmup) ---", flush=True)
    t0 = time.time()
    ilc_data = run_ilc_with_eval_recording(
        sp, basis, sc, tc, SHIFTS["friction_070"], a.seed, a.n_ilc_warmup,
        ref_010, ra, rl, EVAL_SEED_BASE, n_eval=a.n_eval)

    # Warmup summary
    wu_returns = [x["return"] for x in ilc_data["warmup"]]
    print(f"    Warmup: R first5={np.mean(wu_returns[:5]):.1f}  "
          f"last5={np.mean(wu_returns[-5:]):.1f}  "
          f"|u|={ilc_data['warmup'][-1]['|u|']:.3f}", flush=True)

    all_results["kan_ilc_c010"] = summarize(ilc_data["eval"], "KAN-ILC (c=0.10)")
    survival_curves["kan_ilc_c010"] = compute_survival_curve(ilc_data["eval"])
    print(f"    ({time.time()-t0:.1f}s)", flush=True)

    # 5. Oracle on friction
    print("\n  --- Oracle on Friction ---", flush=True)
    t0 = time.time()
    ora_on_frc = [run_episode_oracle(frc_factory, oracle_model, oracle_norm,
                                      EVAL_SEED_BASE + i * 100)
                  for i in range(a.n_eval)]
    all_results["oracle"] = summarize(ora_on_frc, "Oracle→Friction")
    survival_curves["oracle"] = compute_survival_curve(ora_on_frc)
    print(f"    ({time.time()-t0:.1f}s)", flush=True)

    # ── Final comparison table ──────────────────────────────────────────
    print(f"\n[3/3] E0 Results", flush=True)
    print("=" * 72)
    print(f"  {'Method':35s} {'Return':>10s} {'Length':>10s} {'R/step':>8s} "
          f"{'F100':>8s} {'L50':>8s} {'Fell%':>8s}")
    print(f"  {'-'*80}")

    method_order = ["source_on_source", "source_on_friction", "transport",
                    "kan_ilc_c010", "oracle"]
    for key in method_order:
        r = all_results[key]
        print(f"  {key:35s} {r['mean_return']:>10.1f} {r['mean_length']:>10.1f} "
              f"{r['mean_per_step_r']:>8.4f} "
              f"{r.get('mean_first100', 0) or 0:>8.1f} "
              f"{r.get('mean_last50', 0) or 0:>8.1f} "
              f"{r['fell_rate']:>7.0%}")

    # Gap analysis
    src_src_R = all_results["source_on_source"]["mean_return"]
    kan_ilc_R = all_results["kan_ilc_c010"]["mean_return"]
    ora_R = all_results["oracle"]["mean_return"]
    print(f"\n  Gap analysis:")
    print(f"    Source→Source:          {src_src_R:.1f}  (upper bound)")
    print(f"    Oracle→Friction:        {ora_R:.1f}  (reward-trained on target)")
    print(f"    KAN-ILC:                {kan_ilc_R:.1f}  (reward-free)")
    print(f"    KAN-ILC / Source:       {kan_ilc_R/src_src_R*100:.1f}%")
    print(f"    KAN-ILC / Oracle:       {kan_ilc_R/ora_R*100:.1f}%")

    # Per-step reward comparison
    src_src_rbar = all_results["source_on_source"]["mean_per_step_r"]
    kan_rbar = all_results["kan_ilc_c010"]["mean_per_step_r"]
    src_src_T = all_results["source_on_source"]["mean_length"]
    kan_T = all_results["kan_ilc_c010"]["mean_length"]
    print(f"\n  Per-step reward: Source={src_src_rbar:.4f}  KAN-ILC={kan_rbar:.4f}")
    print(f"  Episode length:   Source={src_src_T:.0f}  KAN-ILC={kan_T:.0f}")

    # ── Save ────────────────────────────────────────────────────────────
    # Strip per_step data from summary (too large for quick viewing)
    # Save full data separately
    summary = {
        "config": {"n_eval_episodes": a.n_eval, "n_ilc_warmup": a.n_ilc_warmup,
                   "eval_seed_base": EVAL_SEED_BASE, "seed": a.seed,
                   "max_episode_steps": 1000, "terminate_when_unhealthy": True,
                   "friction": 0.70, "env": "Hopper-v5"},
        "summary": all_results,
        "survival_curves": survival_curves,
    }
    json.dump(summary, open("results/e0_unified_eval.json", "w"), indent=2)

    # Save full per-step data for E1
    full_data = {
        "source_on_source": [{k: v for k, v in ep.items() if k != "per_step"}
                            for ep in src_on_src],
        "source_on_friction": [{k: v for k, v in ep.items() if k != "per_step"}
                              for ep in src_on_frc],
        "transport": [{k: v for k, v in ep.items() if k != "per_step"}
                     for ep in tr_on_frc],
        "kan_ilc_c010": [{k: v for k, v in ep.items() if k != "per_step"}
                         for ep in ilc_data["eval"]],
        "kan_ilc_warmup": ilc_data["warmup"],
        "oracle": [{k: v for k, v in ep.items() if k != "per_step"}
                  for ep in ora_on_frc],
    }

    # Save per-step data for E1 (rewards, forward_r, healthy_r, ctrl_cost per step)
    per_step_summary = {}
    for method_key, episodes in [
        ("source_on_source", src_on_src),
        ("source_on_friction", src_on_frc),
        ("transport", tr_on_frc),
        ("kan_ilc_c010", ilc_data["eval"]),
        ("oracle", ora_on_frc),
    ]:
        ps = []
        for ep in episodes:
            p = ep.get("per_step", {})
            ps.append({
                "forward_r": p.get("forward_r", []),
                "healthy_r": p.get("healthy_r", []),
                "ctrl_cost": p.get("ctrl_cost", []),
                "rewards": p.get("rewards", []),
                "length": ep["length"],
                "total_return": ep["total_return"],
                "fell": ep["fell"],
                "seed": ep["seed"],
            })
        per_step_summary[method_key] = ps

    json.dump({"config": summary["config"], "per_step": per_step_summary},
              open("results/e0_per_step_data.json", "w"), indent=2)

    print(f"  Saved: results/e0_unified_eval.json (summary)")
    print(f"  Saved: results/e0_per_step_data.json (per-step for E1)")


if __name__ == "__main__":
    main()
