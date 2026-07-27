"""E5: Reference space upper bound — test if higher-dim reference helps ILC.

Compares ILC with references at different dimensionalities:
  1. 1-dim: current vx delta
  2. 2,4,8-dim: PCA bases from Transport/oracle trajectories
  3. Source reference (source policy trajectory)
  4. Oracle reference (target oracle trajectory)

All use frozen KAN, identical ILC hyperparameters.
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
N_TRIALS = 25; N_EVAL = 20; ETA = 0.3
VX_IDX = 5  # correct Hopper-v5 vx index


def make_pca_ref(ref_obs, n_dims):
    """Build reference with n_dims PCA perturbations.

    Modifies the reference trajectory along the first n_dims PCA components
    of the Transport reference variation.
    """
    # Center the reference
    mean_ref = ref_obs.mean(axis=0)
    centered = ref_obs - mean_ref
    # SVD
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    # Vt rows = principal directions in observation space

    refs = []
    for d in range(1, n_dims + 1):
        # Scale delta: small perturbation along each PC direction
        delta = Vt[d - 1] * 0.1 * S[0]
        ref_new = ref_obs.copy()
        for i in range(N_PHASE):
            ref_new[i] += delta
        refs.append(ref_new)
    return refs


def run_ilc_and_eval(sp, basis, sc, tc, shift, seed, ref_obs, ref_act, ref_len,
                      n_trials, n_eval, eta, label):
    """Run ILC then eval. Returns (warmup_returns, eval_returns, eval_lengths)."""
    a_dim = basis.action_dim
    u_table = np.zeros((N_PHASE, a_dim), dtype=np.float32)
    warmup_rs = []

    for trial in range(n_trials):
        env = make_shifted_env(shift, seed + trial * 100, "hopper")()
        obs, _ = env.reset(seed=seed + trial * 100)
        step = 0; total_r = 0.0
        detector = CycleDetector(); detector.reset()
        obs_trace, act_trace, td_steps = [], [], []

        while True:
            s_t = torch.as_tensor(obs, device=DEVICE, dtype=torch.float32).unsqueeze(0)
            nominal = sp.action(s_t); s_eff = sc.acceleration(basis, s_t, nominal)
            a_tr = tc.transport_action(basis, s_t, desired_effect=s_eff,
                                       nominal_action=nominal, regularization=1e-2
                                       ).clamp(-1, 1).squeeze(0).cpu().numpy()
            if len(td_steps) > 0:
                phase = min((step - td_steps[-1]) / ref_len, 0.99)
                a_final = np.clip(a_tr + u_table[min(int(phase * N_PHASE), N_PHASE - 1)], -1, 1)
            else: a_final = a_tr
            next_obs, reward, terminated, truncated, _ = env.step(a_final)
            total_r += float(reward)
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
            u_table, _ = ilc_update_with_bt(u_table, cycle_data["obs"], ref_obs,
                lambda a, s_np: get_kan_bt(basis, tc,
                    torch.as_tensor(s_np, device=DEVICE, dtype=torch.float32).unsqueeze(0)),
                None, eta=eta)
        warmup_rs.append(float(total_r))

    # Eval
    eval_rs, eval_lens = [], []
    for i in range(n_eval):
        env = make_shifted_env(shift, 9000 + i * 100, "hopper")()
        obs, _ = env.reset(seed=9000 + i * 100)
        step = 0; total_r = 0.0
        detector = CycleDetector(); detector.reset()
        td_steps = []
        while True:
            s_t = torch.as_tensor(obs, device=DEVICE, dtype=torch.float32).unsqueeze(0)
            nominal = sp.action(s_t); s_eff = sc.acceleration(basis, s_t, nominal)
            a_tr = tc.transport_action(basis, s_t, desired_effect=s_eff,
                                       nominal_action=nominal, regularization=1e-2
                                       ).clamp(-1, 1).squeeze(0).cpu().numpy()
            if len(td_steps) > 0:
                phase = min((step - td_steps[-1]) / ref_len, 0.99)
                a_final = np.clip(a_tr + u_table[min(int(phase * N_PHASE), N_PHASE - 1)], -1, 1)
            else: a_final = a_tr
            next_obs, reward, terminated, truncated, _ = env.step(a_final)
            total_r += float(reward)
            if detector.update(env): td_steps.append(step)
            obs = next_obs; step += 1
            if terminated or truncated: break
        env.close()
        eval_rs.append(float(total_r)); eval_lens.append(step)

    return warmup_rs, eval_rs, eval_lens


def collect_oracle_cycles(oracle_model, oracle_norm, shift, seed, n_ep=10):
    """Collect oracle trajectory cycles for reference building."""
    cycles = []
    for ep in range(n_ep):
        env = make_shifted_env(shift, seed + 30000 + ep * 100, "hopper")()
        obs, _ = env.reset(seed=seed + 30000 + ep * 100)
        obs_trace, act_trace, td_steps = [], [], []
        detector = CycleDetector(); detector.reset()
        step = 0
        while True:
            obs_norm = ((obs - oracle_norm.obs_rms.mean) /
                        (oracle_norm.obs_rms.var + 1e-8) ** 0.5)
            a, _ = oracle_model.predict(obs_norm, deterministic=True)
            next_obs, _, terminated, truncated, _ = env.step(a)
            obs_trace.append(obs.copy()); act_trace.append(a.copy())
            if detector.update(env): td_steps.append(step)
            obs = next_obs; step += 1
            if terminated or truncated: break
        env.close()
        for i in range(len(td_steps) - 1):
            s, e = td_steps[i], td_steps[i + 1]
            if 25 <= (e - s) <= 70:
                cycles.append({"obs": np.array(obs_trace[s:e]),
                               "actions": np.array(act_trace[s:e])}); break
    if len(cycles) < 3: return None, None
    # Phase-align and average
    N = N_PHASE
    all_obs = []
    for cyc in cycles:
        L = cyc["obs"].shape[0]; t_old = np.linspace(0, 1, L); t_new = np.linspace(0, 1, N)
        ao = np.zeros((N, cyc["obs"].shape[1]))
        for d in range(cyc["obs"].shape[1]):
            ao[:, d] = np.interp(t_new, t_old, cyc["obs"][:, d])
        all_obs.append(ao)
    return np.mean(all_obs, axis=0), int(np.mean([c["obs"].shape[0] for c in cycles]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=1811)
    a = p.parse_args()

    print("=" * 72)
    print("E5: Reference Space Upper Bound")
    print(f"  Testing multi-dim reference bases for ILC")
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

    shift = SHIFTS["friction_070"]
    ro, ra, rl = build_reference(sp, basis, sc, tc, shift, DEVICE, a.seed, n_ep=10)
    print(f"  Transport ref: {rl} steps ({time.time()-t0:.1f}s)", flush=True)

    # Oracle
    oracle_model = PPO.load("results/hopper_friction_oracle_seed1811.zip", device=DEVICE)
    oe = DummyVecEnv([make_shifted_env(shift, a.seed, "hopper")])
    oracle_norm = VecNormalize.load("results/hopper_friction_oracle_norm_seed1811.pkl", oe)
    oracle_norm.training = False; oracle_norm.norm_reward = False; oe.close()

    # ── Build references ────────────────────────────────────────────────
    print("\n[2/3] Building reference variants...", flush=True)
    refs = {}

    # Transport self-reference
    refs["transport_self"] = ("Transport self-ref", ro, ra, rl)

    # vx delta = 0.10
    vx_ref = ro.copy(); vx_ref[:, VX_IDX] += 0.10
    refs["vx_0.10"] = ("vx +0.10", vx_ref, ra, rl)

    # PCA references
    pca_refs = make_pca_ref(ro, 4)
    for d, pca_r in enumerate(pca_refs):
        refs[f"pca_{d+1}"] = (f"PCA dim {d+1}", pca_r, ra, rl)

    # Oracle reference
    or_ref, or_len = collect_oracle_cycles(oracle_model, oracle_norm, shift, a.seed, n_ep=10)
    if or_ref is not None:
        refs["oracle_ref"] = ("Oracle ref", or_ref, ra, or_len)

    # ── Run ILC for each reference ──────────────────────────────────────
    print(f"\n[3/3] Running ILC ({N_TRIALS} warmup + {N_EVAL} eval) per reference...", flush=True)
    results = {}

    for key, (label, r_obs, r_act, r_len) in refs.items():
        sys.stdout.write(f"  {label}: "); sys.stdout.flush()
        t0 = time.time()
        wu, ev_r, ev_T = run_ilc_and_eval(sp, basis, sc, tc, shift, a.seed,
                                            r_obs, r_act, r_len, N_TRIALS, N_EVAL, ETA, label)
        print(f"Warmup last5={np.mean(wu[-5:]):.1f}  "
              f"Eval R={np.mean(ev_r):.1f}+/-{np.std(ev_r):.0f}  "
              f"T={np.mean(ev_T):.1f}  ({time.time()-t0:.1f}s)", flush=True)

        results[key] = {
            "label": label,
            "warmup_last5": float(np.mean(wu[-5:])),
            "eval_mean_return": float(np.mean(ev_r)),
            "eval_std_return": float(np.std(ev_r)),
            "eval_mean_length": float(np.mean(ev_T)),
        }

    # ── Summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  E5 Results: Reference Space Scan")
    print(f"  {'Reference':25s} {'Warmup_last5':>12s} {'Eval_R':>10s} {'Eval_T':>10s}")
    print(f"  {'-'*60}")
    for key, r in results.items():
        print(f"  {r['label']:25s} {r['warmup_last5']:>12.1f} "
              f"{r['eval_mean_return']:>10.1f} {r['eval_mean_length']:>10.1f}")

    json.dump({"config": {"n_trials": N_TRIALS, "n_eval": N_EVAL},
               "results": results},
              open("results/e5_reference_upper_bound.json", "w"), indent=2)
    print(f"\n  Saved to results/e5_reference_upper_bound.json")


if __name__ == "__main__":
    main()
