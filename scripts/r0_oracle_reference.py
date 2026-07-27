"""R0: Oracle target reference upper bound.

Answers: if KAN-B_t ILC is given a near-optimal reference trajectory,
how close can it get to oracle performance?

Gate:
  Case A: Oracle ref >> 655 → Executor is capable, main bottleneck is Router.
  Case B: Oracle ref ≈ 655 → Executor still has structural limits.

Uses PPO trained directly on friction=0.7 as the oracle reference provider.
"""
import sys, numpy as np, torch, argparse, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cpbn.generic_affine_kan import RecursiveAffineKANEstimator
from scripts.prescreen_hopper_physics_shifts import SHIFTS, make_shifted_env
from scripts.validate_hopper_joint_online_adaptation import FrozenSourcePolicy, load_cognition
from scripts.diagnose_hopper_pullback_effect import fit_distilled_source_counterfactual_context, load_source_twin
from scripts.ilc1_ablation import *
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

DEVICE = torch.device("cuda")
SHIFT = SHIFTS["friction_070"]
N_TRIALS = 40
ETA = 0.3
ORACLE_STEPS = 200_000

# ═══════════════════════════════════════════════════════════════════════
# Train / load Oracle
# ═══════════════════════════════════════════════════════════════════════

def get_oracle_policy(seed):
    """Train or load PPO on friction=0.7."""
    ckpt_path = f"results/hopper_friction_oracle_seed{seed}.zip"
    norm_path = f"results/hopper_friction_oracle_norm_seed{seed}.pkl"
    if Path(ckpt_path).exists():
        print(f"  Loading existing oracle from {ckpt_path}", flush=True)
        env = DummyVecEnv([make_shifted_env(SHIFT, seed, "hopper")])
        norm = VecNormalize.load(norm_path, env)
        model = PPO.load(ckpt_path, device=DEVICE)
        env.close()
        return model, norm
    print(f"  Training friction Oracle PPO ({ORACLE_STEPS} steps)...", flush=True)
    env = DummyVecEnv([make_shifted_env(SHIFT, seed, "hopper")])
    env = VecNormalize(env, norm_obs=True, norm_reward=False)
    model = PPO("MlpPolicy", env, verbose=0, device=DEVICE, seed=seed)
    model.learn(total_timesteps=ORACLE_STEPS)
    model.save(ckpt_path)
    env.save(norm_path)
    env.close()
    # Evaluate
    eval_env = DummyVecEnv([make_shifted_env(SHIFT, seed + 1000, "hopper")])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False)
    eval_env.obs_rms = env.obs_rms
    eval_env.training = False; eval_env.norm_reward = False
    returns = []
    for _ in range(10):
        obs = eval_env.reset()
        total_r, done = 0, False
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, r, d, info = eval_env.step(a)
            total_r += r[0]; done = d[0] or info[0].get("TimeLimit.truncated", False)
        returns.append(total_r)
    eval_env.close()
    print(f"  Oracle trained: {np.mean(returns):.0f} ± {np.std(returns):.0f}", flush=True)
    return model, env  # return vecnormalize for obs_rms


def make_oracle_policy_fn(model, norm):
    """Return function: obs_np -> action_np."""
    def fn(obs):
        obs_norm = ((obs - norm.obs_rms.mean) / (norm.obs_rms.var + 1e-8)**0.5)
        a, _ = model.predict(obs_norm, deterministic=True)
        return a
    return fn


# ═══════════════════════════════════════════════════════════════════════
# Build oracle reference
# ═══════════════════════════════════════════════════════════════════════

def build_oracle_reference(oracle_fn, n_ep=10, seed=1811):
    """Collect oracle cycles and build phase-aligned reference."""
    cycles = []
    for ep in range(n_ep):
        env = make_shifted_env(SHIFT, seed + 30000 + ep * 100, "hopper")()
        obs, _ = env.reset()
        detector = CycleDetector(); detector.reset()
        obs_trace, act_trace, td_steps = [], [], []
        step = 0
        while True:
            a = oracle_fn(obs).flatten()
            next_obs, _, terminated, truncated, _ = env.step(a)
            obs_trace.append(obs.copy()); act_trace.append(a.copy())
            if detector.update(env): td_steps.append(step)
            obs = next_obs; step += 1
            if terminated or truncated: break
        env.close()
        for i in range(len(td_steps) - 1):
            s, e = td_steps[i], td_steps[i + 1]
            if 15 <= (e - s) <= 70:
                cycles.append({"obs": np.array(obs_trace[s:e]),
                               "actions": np.array(act_trace[s:e]),
                               "length": e - s})
                break
    if len(cycles) < 3:
        return None, None, 0
    N = N_PHASE
    all_obs, all_act = [], []
    for cyc in cycles:
        L = cyc["obs"].shape[0]; t_old = np.linspace(0, 1, L); t_new = np.linspace(0, 1, N)
        ao = np.zeros((N, cyc["obs"].shape[1]))
        aa = np.zeros((N, cyc["actions"].shape[1]))
        for d in range(cyc["obs"].shape[1]):
            ao[:, d] = np.interp(t_new, t_old, cyc["obs"][:, d])
        for d in range(cyc["actions"].shape[1]):
            aa[:, d] = np.interp(t_new, t_old, cyc["actions"][:, d])
        all_obs.append(ao); all_act.append(aa)
    return np.mean(all_obs, axis=0), np.mean(all_act, axis=0), int(np.mean([c["length"] for c in cycles]))


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1811)
    parser.add_argument("--n-trials", type=int, default=40)
    args = parser.parse_args()

    print("=" * 60)
    print("R0: Oracle Target Reference Upper Bound")
    print("=" * 60)

    # Load
    print("\n[1/4] Loading components...", flush=True)
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

    # Train oracle
    print("\n[2/4] Oracle policy...", flush=True)
    oracle_model, oracle_norm = get_oracle_policy(args.seed)
    oracle_fn = make_oracle_policy_fn(oracle_model, oracle_norm)

    # Build references
    print("\n[3/4] Building references...", flush=True)
    ref_tr_obs, ref_tr_act, ref_tr_len = build_reference(sp, basis, sc, tc, SHIFT, DEVICE, args.seed, n_ep=10)
    print(f"  Transport ref: {ref_tr_len} steps", flush=True)

    ref_or_obs, ref_or_act, ref_or_len = build_oracle_reference(oracle_fn, n_ep=10, seed=args.seed)
    if ref_or_obs is None:
        print("  ABORT: Cannot build oracle reference.")
        return
    print(f"  Oracle ref: {ref_or_len} steps", flush=True)

    refs = {
        "Transport_ref": (ref_tr_obs, ref_tr_act, ref_tr_len),
        "Oracle_ref": (ref_or_obs, ref_or_act, ref_or_len),
        "delta_0.30": (make_progressive_ref(ref_tr_obs, None, 0.30), ref_tr_act, ref_tr_len),
    }

    # Run ILC
    print(f"\n[4/4] ILC trials ({args.n_trials} each)...", flush=True)

    all_results = {}
    for ref_name, (r_obs, r_act, r_len) in refs.items():
        label = f"kan/{ref_name}"
        sys.stdout.write(f"  {label}: "); sys.stdout.flush()
        results = run_ilc_experiment(sp, basis, sc, tc, SHIFT, DEVICE, args.seed,
                                     args.n_trials, r_obs, r_act, r_len, "kan", None, eta=ETA)
        returns = [r["return"] for r in results]
        errors = [r["error"] for r in results if r["error"] is not None]
        mr, f5, l5 = np.mean(returns), np.mean(returns[:5]), np.mean(returns[-5:])
        me = np.mean(errors) if errors else float('nan')
        print(f"R={mr:.1f} f5={f5:.1f} l5={l5:.1f} E={me:.4f}", flush=True)
        all_results[ref_name] = {"mean_return": float(mr), "first5": float(f5),
                                  "last5": float(l5), "mean_error": float(me) if not np.isnan(me) else None,
                                  "returns": [float(r) for r in returns]}

    # Summary
    print(f"\n{'='*60}")
    print(f"  {'Reference':20s} {'Return':>10s} {'First5':>10s} {'Last5':>10s}")
    print(f"  {'-'*50}")
    for name, res in all_results.items():
        print(f"  {name:20s} {res['mean_return']:>10.1f} {res['first5']:>10.1f} {res['last5']:>10.1f}")

    tr_r = all_results.get("Transport_ref", {}).get("mean_return", 571)
    or_r = all_results.get("Oracle_ref", {}).get("mean_return", 0)
    print(f"\n  Transport baseline: {tr_r:.1f}")
    print(f"  Oracle ref return: {or_r:.1f}")
    if or_r > 655 + 20:
        print(f"  => PASS: Executor can benefit from better reference. Router is the bottleneck.")
    elif or_r > 655:
        print(f"  => MARGINAL: Oracle ref slightly better than delta_0.30.")
    else:
        print(f"  => FAIL: Executor cannot improve beyond ~655 even with oracle reference.")

    json.dump({"results": all_results, "transport_baseline": float(tr_r)},
              open("results/r0_oracle_reference.json", "w"), indent=2)
    print("  Saved to results/r0_oracle_reference.json")


if __name__ == "__main__":
    main()
