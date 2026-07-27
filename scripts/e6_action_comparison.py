"""E6: Action comparison — is ILC undoing Transport?

Compares KAN-ILC action vs Transport, Source, and Oracle at identical states.
Computes:
  d_src = |a_KAN-ILC - a_source|
  d_tr  = |a_KAN-ILC - a_transport|
  d_or  = |a_KAN-ILC - a_oracle|
  cos(u_ILC, a_tr - a_src)
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
N_EPS = 30; N_ILC_WARMUP = 25; ETA = 0.3


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=1811)
    a = p.parse_args()

    print("=" * 72)
    print("E6: ILC vs Transport Action Comparison")
    print(f"  Comparing actions at identical states across methods")
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

    oracle_model = PPO.load("results/hopper_friction_oracle_seed1811.zip", device=DEVICE)
    oe = DummyVecEnv([make_shifted_env(SHIFTS["friction_070"], a.seed, "hopper")])
    oracle_norm = VecNormalize.load("results/hopper_friction_oracle_norm_seed1811.pkl", oe)
    oracle_norm.training = False; oracle_norm.norm_reward = False; oe.close()

    shift = SHIFTS["friction_070"]
    env_factory = lambda s=0: make_shifted_env(shift, s, "hopper")()
    print(f"  Load time: {time.time()-t0:.1f}s", flush=True)

    # ── ILC warmup ──────────────────────────────────────────────────────
    print(f"\n[2/3] ILC warmup ({N_ILC_WARMUP} trials, self-reference)...", flush=True)
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
                phase = min((step - td_steps[-1]) / rl, 0.99)
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

    # ── Compare actions at identical states ─────────────────────────────
    print(f"\n[3/3] Comparing actions over {N_EPS} episodes...", flush=True)

    metrics = {"d_src": [], "d_tr": [], "d_oracle": [], "cos_ilc_trcorr": [],
               "ilc_norm": [], "tr_correction_norm": []}

    for ep in range(N_EPS):
        env = env_factory(9000 + ep * 100)
        obs, _ = env.reset(seed=9000 + ep * 100)
        step = 0; detector = CycleDetector(); detector.reset()
        td_steps = []

        while True:
            s_t = torch.as_tensor(obs, device=DEVICE, dtype=torch.float32).unsqueeze(0)
            nominal = sp.action(s_t)
            a_src = nominal.squeeze(0).cpu().numpy()

            s_eff = sc.acceleration(basis, s_t, nominal)
            a_tr = tc.transport_action(basis, s_t, desired_effect=s_eff,
                                       nominal_action=nominal, regularization=1e-2
                                       ).clamp(-1, 1).squeeze(0).cpu().numpy()

            # Oracle action at this state
            obs_norm = ((obs - oracle_norm.obs_rms.mean) /
                        (oracle_norm.obs_rms.var + 1e-8) ** 0.5)
            a_oracle, _ = oracle_model.predict(obs_norm, deterministic=True)

            # KAN-ILC action
            u_ff = np.zeros_like(a_tr)
            if u_table is not None and len(td_steps) > 0:
                phase = min((step - td_steps[-1]) / rl, 0.99)
                u_ff = u_table[min(int(phase * N_PHASE), N_PHASE - 1)]
            a_ilc = np.clip(a_tr + u_ff, -1, 1)

            # Transport correction = a_tr - a_src
            tr_correction = a_tr - a_src

            # Record metrics
            metrics["d_src"].append(float(np.linalg.norm(a_ilc - a_src)))
            metrics["d_tr"].append(float(np.linalg.norm(a_ilc - a_tr)))
            metrics["d_oracle"].append(float(np.linalg.norm(a_ilc - a_oracle)))
            metrics["ilc_norm"].append(float(np.linalg.norm(u_ff)))

            tr_corr_norm = np.linalg.norm(tr_correction)
            metrics["tr_correction_norm"].append(float(tr_corr_norm))
            if tr_corr_norm > 1e-10 and np.linalg.norm(u_ff) > 1e-10:
                cos_val = float(np.dot(u_ff, tr_correction) /
                               (np.linalg.norm(u_ff) * tr_corr_norm + 1e-10))
            else:
                cos_val = float('nan')
            metrics["cos_ilc_trcorr"].append(cos_val)

            next_obs, reward, terminated, truncated, _ = env.step(a_ilc)
            if detector.update(env): td_steps.append(step)
            obs = next_obs; step += 1
            if terminated or truncated: break
        env.close()

    # ── Report ──────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("  E6 Results: Action Comparison")
    print(f"  Total state-action pairs: {len(metrics['d_src'])}")
    print(f"  {'Metric':30s} {'Mean':>10s} {'Std':>10s}")
    print(f"  {'-'*55}")

    valid_cos = [x for x in metrics["cos_ilc_trcorr"] if not np.isnan(x)]

    for metric, label in [
        ("d_src", "|a_ILC - a_source|"),
        ("d_tr", "|a_ILC - a_transport|"),
        ("d_oracle", "|a_ILC - a_oracle|"),
        ("ilc_norm", "|u_ILC| (residual norm)"),
        ("tr_correction_norm", "|a_tr - a_src| (Transport correction)"),
    ]:
        vals = metrics[metric]
        print(f"  {label:30s} {np.mean(vals):>10.4f} {np.std(vals):>10.4f}")

    if valid_cos:
        mean_cos = np.mean(valid_cos)
        print(f"  {'cos(u_ILC, a_tr - a_src)':30s} {mean_cos:>10.4f} {np.std(valid_cos):>10.4f}")
        if mean_cos < -0.3:
            print(f"\n  >>> ILC residual OPPOSES Transport correction")
            print(f"      ILC is partially undoing Transport's conservative pullback")
        elif abs(mean_cos) < 0.3:
            print(f"\n  >>> ILC residual is nearly orthogonal to Transport correction")
            print(f"      ILC adds new behavior, not undoing Transport")
        else:
            print(f"\n  >>> ILC residual REINFORCES Transport correction")

    # Distance ratios
    d_tr_mean = np.mean(metrics["d_tr"])
    d_src_mean = np.mean(metrics["d_src"])
    d_oracle_mean = np.mean(metrics["d_oracle"])
    print(f"\n  Distance ratios:")
    print(f"    |ILC - Tr| / |ILC - Src| = {d_tr_mean/d_src_mean:.2f}")
    print(f"    |ILC - Tr| / |ILC - Oracle| = {d_tr_mean/d_oracle_mean:.2f}")
    if d_tr_mean < d_src_mean * 0.5:
        print(f"    => ILC action is CLOSER to Transport than to Source")
    if d_tr_mean < d_oracle_mean * 0.5:
        print(f"    => ILC action is CLOSER to Transport than to Oracle")
    if d_oracle_mean < d_src_mean * 0.5:
        print(f"    => Oracle action is CLOSER to ILC than Source is")

    summary = {k: {"mean": float(np.mean(v)), "std": float(np.std(v))}
               for k, v in metrics.items()}
    if valid_cos:
        summary["cos_ilc_trcorr"]["mean"] = float(np.mean(valid_cos))
        summary["cos_ilc_trcorr"]["n_valid"] = len(valid_cos)
    json.dump(summary, open("results/e6_action_comparison.json", "w"), indent=2)
    print(f"\n  Saved to results/e6_action_comparison.json")


if __name__ == "__main__":
    main()
