"""ILC-1 with online KAN — fresh ref per seed, shared KAN across conditions.

Fixes:
  1. Build Transport reference FRESH for each seed (no stale npz)
  2. Share KAN estimator across conditions within a seed (accumulate knowledge)
  3. u_table resets per condition (ILC starts fresh for each ref/B_t combo)
"""
import sys, numpy as np, torch, argparse, json
from pathlib import Path
sys.path.insert(0, ".")
from cpbn.generic_affine_kan import RecursiveAffineKANEstimator
from scripts.prescreen_hopper_physics_shifts import SHIFTS, make_shifted_env
from scripts.validate_hopper_joint_online_adaptation import FrozenSourcePolicy, load_cognition
from scripts.diagnose_hopper_pullback_effect import fit_distilled_source_counterfactual_context, load_source_twin
from scripts.ilc1_ablation import *

DEVICE = torch.device("cuda")
SHIFT = SHIFTS["friction_070"]
N_TRIALS = 40
ETA = 0.3
ILC_UPDATE_EVERY = 5  # average over N cycles before each ILC update

CONDITIONS = [
    ("kan_online", "tr"),
    ("kan_online", "delta_0.15"),
    ("kan_online", "delta_0.30"),
    ("proportional", "tr"),
    ("shuffled", "tr"),
    ("shuffled", "delta_0.15"),
    ("negative", "tr"),
]


class OnlineKAN:
    """Per-condition online KAN with quality-gated updates.

    Only updates from episodes that survive near full length,
    preventing crash data from poisoning the model.
    """

    def __init__(self, basis, tc_init, ridge=50.0, forget=0.995,
                 min_return_for_update=300, min_length_for_update=100):
        self.basis = basis
        self.estimator = RecursiveAffineKANEstimator(basis, tc_init, ridge=ridge, forgetting_factor=forget)
        self.min_return = min_return_for_update
        self.min_length = min_length_for_update
        self.n_updates = 0
        self.n_skipped = 0

    def context(self):
        return self.estimator.context()

    def maybe_update(self, obs_list, act_list, next_obs_list, episode_return, episode_length):
        """Update KAN only if episode was successful enough."""
        if episode_return >= self.min_return and episode_length >= self.min_length:
            batch_s = torch.tensor(np.stack(obs_list), device=DEVICE, dtype=torch.float32)
            batch_a = torch.tensor(np.stack(act_list), device=DEVICE, dtype=torch.float32)
            batch_acc = torch.tensor(np.stack(next_obs_list) - np.stack(obs_list),
                                     device=DEVICE, dtype=torch.float32)
            self.estimator.update(batch_s, batch_a, batch_acc)
            self.n_updates += 1
            return True
        self.n_skipped += 1
        return False


def make_bt_func(bt_type, online_kan, basis):
    """Create B_t function from online KAN."""
    if bt_type == "kan_online":
        def f(args, s_np):
            s_t = torch.as_tensor(s_np, device=DEVICE, dtype=torch.float32).unsqueeze(0)
            return get_kan_bt(basis, online_kan.context(), s_t)
        return f
    elif bt_type == "shuffled":
        def f(args, s_np):
            s_t = torch.as_tensor(s_np, device=DEVICE, dtype=torch.float32).unsqueeze(0)
            B = get_kan_bt(basis, online_kan.context(), s_t)
            return shuffle_bt(B)
        return f
    elif bt_type == "negative":
        def f(args, s_np):
            s_t = torch.as_tensor(s_np, device=DEVICE, dtype=torch.float32).unsqueeze(0)
            return -get_kan_bt(basis, online_kan.context(), s_t)
        return f
    elif bt_type == "proportional":
        return None
    else:
        raise ValueError(f"Unknown: {bt_type}")


def run_ilc_condition(sp, basis, sc, tc_init, shift, seed, n_trials,
                      ref_obs, ref_act, ref_cyc_len, bt_type, eta=0.3):
    """Run one ILC condition with independent online KAN. Quality-gated updates."""
    a_dim = basis.action_dim
    u_table = np.zeros((N_PHASE, a_dim), dtype=np.float32)
    online_kan = OnlineKAN(basis, tc_init, ridge=50.0, forget=0.995)
    bt_func = make_bt_func(bt_type, online_kan, basis)
    cycles_buffer = []  # accumulate cycles for averaged ILC updates
    results = []

    for trial in range(n_trials):
        env = make_shifted_env(shift, seed + trial * 100, "hopper")()
        obs, _ = env.reset()
        total_r = 0.0
        detector = CycleDetector(); detector.reset()
        obs_trace, act_trace, next_obs_trace = [], [], []
        td_steps = []
        step = 0
        ctx = online_kan.context()

        while True:
            s_t = torch.as_tensor(obs, device=DEVICE, dtype=torch.float32).unsqueeze(0)
            nominal = sp.action(s_t)
            s_eff = sc.acceleration(basis, s_t, nominal)
            a_tr = ctx.transport_action(basis, s_t, desired_effect=s_eff,
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
            obs_trace.append(obs.copy())
            act_trace.append(a_final.copy())
            next_obs_trace.append(next_obs.copy())
            if detector.update(env):
                td_steps.append(step)
            obs = next_obs; step += 1
            if terminated or truncated:
                break

        env.close()

        # Quality-gated KAN update
        updated = online_kan.maybe_update(obs_trace, act_trace, next_obs_trace, total_r, step)
        if updated:
            bt_func = make_bt_func(bt_type, online_kan, basis)

        trial_res = {"trial": trial, "return": float(total_r), "length": step}

        # Extract cycle + ILC update
        cycle_data = None
        for i in range(len(td_steps) - 1):
            s, e = td_steps[i], td_steps[i + 1]
            if 25 <= (e - s) <= 70:
                cycle_data = {"obs": np.array(obs_trace[s:e]),
                              "actions": np.array(act_trace[s:e]),
                              "length": e - s}
                break

        if cycle_data is not None:
            L = cycle_data["obs"].shape[0]
            t_old = np.linspace(0, 1, L)
            t_new = np.linspace(0, 1, N_PHASE)
            aligned = np.zeros((N_PHASE, cycle_data["obs"].shape[1]))
            for d in range(cycle_data["obs"].shape[1]):
                aligned[:, d] = np.interp(t_new, t_old, cycle_data["obs"][:, d])
            task_dims = [0, 2, 6, 3, 4]
            tw = np.array([2.0, 1.0, 1.0, 0.5, 0.5])
            err = sum(np.sum((tw * (ref_obs[i, task_dims] - aligned[i, task_dims])) ** 2)
                     for i in range(N_PHASE))
            trial_res["error"] = float(np.sqrt(err / N_PHASE))
            trial_res["residual_norm"] = float(np.linalg.norm(u_table))

            # Accumulate cycles, ILC update every N cycles (phase-align + average)
            cycles_buffer.append(cycle_data)
            if len(cycles_buffer) >= ILC_UPDATE_EVERY:
                # Phase-align each cycle to N_PHASE, then average
                aligned_cycles = []
                for cyc in cycles_buffer:
                    L = cyc["obs"].shape[0]
                    t_old = np.linspace(0, 1, L); t_new = np.linspace(0, 1, N_PHASE)
                    aligned = np.zeros((N_PHASE, cyc["obs"].shape[1]))
                    for d in range(cyc["obs"].shape[1]):
                        aligned[:, d] = np.interp(t_new, t_old, cyc["obs"][:, d])
                    aligned_cycles.append(aligned)
                avg_cycle_obs = np.mean(aligned_cycles, axis=0)
                u_table, _ = ilc_update_with_bt(u_table, avg_cycle_obs, ref_obs, bt_func, None, eta=eta)
                cycles_buffer = []
        else:
            trial_res["error"] = None
            trial_res["residual_norm"] = float(np.linalg.norm(u_table))

        results.append(trial_res)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=str, default="1811,1911,2011")
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    print(f"ILC-1 Online KAN (v2): {len(seeds)} seeds × {len(CONDITIONS)} cond × {N_TRIALS} trials")
    print(f"  Fixes: fresh ref per seed, shared KAN across conditions")

    # Load once
    print("[1/3] Loading shared components...", flush=True)
    sp = FrozenSourcePolicy("results/hopper_source_sb3_ppo_continued_seed1811.zip",
        "results/hopper_source_sb3_vecnorm_continued_seed1811.pkl", DEVICE, seeds[0], env="hopper")
    st = load_source_twin("results/hopper_source_affine_twin_cloud_seed1811.pt", DEVICE)
    basis, sc, _, _ = load_cognition(argparse.Namespace(
        cognition_checkpoint="results/hopper_source_control_sobolev_calibrated_seed1811.pt", device="cuda"), DEVICE)

    all_results = {}

    for seed in seeds:
        print(f"\n[2/3] Seed {seed}: initial KAN fit + building reference...", flush=True)
        fa = argparse.Namespace(target="friction_070", seed=seed, env="hopper", device="cuda",
            cognition_warmup=1024, warmup_noise=0.3, transform_ridge=10.0, drift_ridge=100.0,
            drift_spectral_eta=0.0, drift_spectral_beta=1.0, drift_spectral_mode="max",
            drift_smooth_lambda=0.0, diagonal_transform=False)
        tc_init, _ = fit_distilled_source_counterfactual_context(sp, basis, sc, fa, DEVICE, st)

        # Build reference FRESH for this seed
        ref_obs, ref_act, ref_cyc_len = build_reference(sp, basis, sc, tc_init, SHIFT, DEVICE, seed, n_ep=10)
        print(f"  Reference: {ref_cyc_len} steps/cycle", flush=True)

        refs = {
            "tr": (ref_obs, ref_act, ref_cyc_len),
            "delta_0.15": (make_progressive_ref(ref_obs, None, 0.15), ref_act, ref_cyc_len),
            "delta_0.30": (make_progressive_ref(ref_obs, None, 0.30), ref_act, ref_cyc_len),
        }

        for bt_type, ref_name in CONDITIONS:
            r_obs, r_act, r_len = refs[ref_name]
            label = f"s{seed}/{bt_type}/{ref_name}"
            sys.stdout.write(f"  {label}: "); sys.stdout.flush()

            results = run_ilc_condition(
                sp, basis, sc, tc_init, SHIFT, seed, N_TRIALS,
                r_obs, r_act, r_len, bt_type, eta=ETA)

            returns = [r["return"] for r in results]
            errors = [r["error"] for r in results if r["error"] is not None]
            n_cyc = sum(1 for r in results if r["error"] is not None)
            mr, f5, l5 = np.mean(returns), np.mean(returns[:5]), np.mean(returns[-5:])
            me = np.mean(errors) if errors else float('nan')

            print(f"R={mr:.1f} f5={f5:.1f} l5={l5:.1f} E={me:.4f} cyc={n_cyc}/{N_TRIALS}", flush=True)

            all_results[label] = {
                "seed": seed, "bt_type": bt_type, "ref": ref_name,
                "mean_return": float(mr), "first5_return": float(f5), "last5_return": float(l5),
                "mean_error": float(me) if not np.isnan(me) else None,
                "n_cycles": n_cyc, "returns": returns,
                "ref_cyc_len": ref_cyc_len,
            }

    # Summary
    print(f"\n[3/3] Summary across {len(seeds)} seeds")
    print(f"  {'B_t':>18s} {'Ref':>12s} | {'MeanR':>8s} {'F5':>8s} {'L5':>8s} {'ΔLearn':>8s}")
    print(f"  {'-'*68}")

    for bt in ["kan_online", "proportional", "shuffled", "negative"]:
        for ref in ["tr", "delta_0.15", "delta_0.30"]:
            br = [v for k, v in all_results.items() if v["bt_type"] == bt and v["ref"] == ref]
            if not br: continue
            avg = np.mean([r["mean_return"] for r in br])
            f5 = np.mean([r["first5_return"] for r in br])
            l5 = np.mean([r["last5_return"] for r in br])
            print(f"  {bt:>18s} {ref:>12s} | {avg:>8.1f} {f5:>8.1f} {l5:>8.1f} {l5-f5:>+8.1f}")

    json.dump({"config": {"seeds": seeds, "n_trials": N_TRIALS, "conditions": CONDITIONS},
               "results": all_results},
              open("results/ilc1_online_kan.json", "w"), indent=2)
    print("  Saved to results/ilc1_online_kan.json")


if __name__ == "__main__":
    main()
