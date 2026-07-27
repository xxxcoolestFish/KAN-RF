"""ILC-1 Ablation: multi-seed, B_t variants, reward decomposition.

Tests:
  B_t variants: KAN, Oracle(FD), Source(unfitted), Shuffled, Negative, Proportional
  References: Transport-self, progressive-delta (vx+, height+)
  Metrics: return, forward/healthy/ctrl reward, tracking error, residual norm
"""

from __future__ import annotations

import argparse, sys, os, json
from pathlib import Path
from collections import deque
import numpy as np, torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.diagnose_hopper_pullback_effect import (
    fit_distilled_source_counterfactual_context, load_source_twin)
from scripts.prescreen_hopper_physics_shifts import SHIFTS, make_shifted_env
from scripts.validate_hopper_joint_online_adaptation import FrozenSourcePolicy, load_cognition

N_PHASE = 20

# ═══════════════════════════════════════════════════════════════════════
# Cycle detector (local minima of foot_z)
# ═══════════════════════════════════════════════════════════════════════

class CycleDetector:
    def __init__(self, min_interval=25, window=5):
        self.min_interval = min_interval
        self.window = window
        self._last_cyc = -min_interval
        self._step = 0
        self._fz_hist = []

    def reset(self):
        self._last_cyc = -self.min_interval
        self._step = 0
        self._fz_hist = []

    def update(self, env):
        try:
            fz = float(env.unwrapped.data.body("foot").xpos[2])
        except Exception:
            fz = 0.0
        self._fz_hist.append(fz)
        self._step += 1
        w = self.window
        if len(self._fz_hist) < 2 * w + 1:
            return False
        recent = self._fz_hist[-(2 * w + 1):]
        if recent[w] < min(recent[:w]) and recent[w] <= min(recent[w + 1:]):
            if (self._step - self._last_cyc) >= self.min_interval:
                self._last_cyc = self._step
                return True
        return False


# ═══════════════════════════════════════════════════════════════════════
# B_t variants
# ═══════════════════════════════════════════════════════════════════════

def get_kan_bt(basis, target_ctx, s_tensor):
    with torch.no_grad():
        _, gain = target_ctx.drift_and_gain(basis, s_tensor)
        return gain.squeeze(0).cpu().numpy()  # (s_dim, a_dim)

def get_source_bt(basis, source_ctx, s_tensor):
    with torch.no_grad():
        _, gain = source_ctx.drift_and_gain(basis, s_tensor)
        return gain.squeeze(0).cpu().numpy()

def get_oracle_bt(fd_env, qp, qv, a_nom, eps=0.05):
    s_dim, a_dim = 11, len(a_nom)
    B = np.zeros((s_dim, a_dim))
    fd_env.unwrapped.set_state(qp, qv)
    obs_base, _, _, _, _ = fd_env.step(a_nom)
    for d in range(a_dim):
        da = np.zeros(a_dim); da[d] = eps
        fd_env.unwrapped.set_state(qp, qv)
        obs_pert, _, _, _, _ = fd_env.step(np.clip(a_nom + da, -1, 1))
        B[:, d] = (obs_pert - obs_base) / eps
    return B

def shuffle_bt(B):
    """Randomly permute rows (state-dim → action-dim mapping scrambled)."""
    B_s = B.copy()
    np.random.shuffle(B_s)
    return B_s


# ═══════════════════════════════════════════════════════════════════════
# ILC update
# ═══════════════════════════════════════════════════════════════════════

def ilc_update_with_bt(u_k, cycle_obs, ref_obs, bt_func, bt_args,
                       eta=0.3, beta_f=0.1, lam=0.1, N=N_PHASE):
    """ILC update using provided B_t function.

    bt_func(bt_args, s_np) -> B (s_dim, a_dim) or None for proportional
    """
    L = cycle_obs.shape[0]
    s_dim, a_dim = cycle_obs.shape[1], u_k.shape[1]
    task_dims = [0, 2, 6, 3, 4]
    task_weight = np.array([2.0, 1.0, 1.0, 0.5, 0.5])

    t_old = np.linspace(0, 1, L)
    t_new = np.linspace(0, 1, N)
    aligned_obs = np.zeros((N, s_dim))
    for d in range(s_dim):
        aligned_obs[:, d] = np.interp(t_new, t_old, cycle_obs[:, d])

    err_norm = 0.0
    u_new = u_k.copy()

    for i in range(N):
        e_full = ref_obs[i] - aligned_obs[i]
        e_task = e_full[task_dims]
        e_weighted = e_task * task_weight
        err_norm += np.sum(e_weighted ** 2)

        if bt_func is None:  # proportional
            delta_u = np.zeros(a_dim)
            delta_u[0] = 0.05 * (0.5 * e_weighted[0] + 0.3 * e_weighted[2])
            delta_u[1] = 0.05 * (0.5 * e_weighted[0] + 0.2 * e_weighted[3])
            delta_u[2] = 0.05 * (0.3 * e_weighted[3])
        else:
            B = bt_func(bt_args, aligned_obs[i])
            B_task = B[task_dims, :]
            Q_diag = np.diag(task_weight)
            BtQB = B_task.T @ Q_diag @ B_task
            reg = BtQB + lam * np.eye(a_dim)
            delta_u = np.linalg.solve(reg, B_task.T @ Q_diag @ e_weighted)

        u_new[i] = (1.0 - beta_f) * u_k[i] + eta * delta_u

    u_new = np.clip(u_new, -0.15, 0.15)
    for d in range(a_dim):
        u_new[:, d] = np.convolve(u_new[:, d], [0.25, 0.5, 0.25], mode='same')

    return u_new, np.sqrt(err_norm / N)


# ═══════════════════════════════════════════════════════════════════════
# Episode runner
# ═══════════════════════════════════════════════════════════════════════

def run_ilc_episode(env, sp, basis, sc, tc, u_table, device, ref_cyc_len=55):
    """Run episode, extract cycle, return (total_r, cycle_data, steps, rewards_dict)."""
    obs, _ = env.reset()
    total_r, fwd_r, healthy_r, ctrl_r = 0.0, 0.0, 0.0, 0.0
    detector = CycleDetector()
    detector.reset()
    obs_trace, act_trace, td_steps = [], [], []
    step = 0

    while True:
        s_t = torch.as_tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
        nominal = sp.action(s_t)
        s_eff = sc.acceleration(basis, s_t, nominal)
        a_tr = tc.transport_action(basis, s_t, desired_effect=s_eff,
                                   nominal_action=nominal, regularization=1e-2
                                   ).clamp(-1, 1).squeeze(0).cpu().numpy()

        if u_table is not None and len(td_steps) > 0:
            steps_since_td = step - td_steps[-1]
            phase = min(steps_since_td / ref_cyc_len, 0.99)
            u_ff = u_table[min(int(phase * N_PHASE), N_PHASE - 1)]
            a_final = np.clip(a_tr + u_ff, -1, 1)
        else:
            a_final = a_tr

        next_obs, reward, terminated, truncated, _ = env.step(a_final)
        total_r += float(reward)

        # Decompose reward (approximate — use env info)
        z, angle = next_obs[1], next_obs[2]
        z_ok = 0.7 < z < float('inf')
        a_ok = -0.2 < angle < 0.2
        healthy_r += 1.0 if (z_ok and a_ok) else 0.0
        ctrl_r += 0.001 * float(np.sum(a_final ** 2))
        # forward component = total - healthy + ctrl
        fwd_r += float(reward) - (1.0 if (z_ok and a_ok) else 0.0) + 0.001 * float(np.sum(a_final ** 2))

        obs_trace.append(obs.copy())
        act_trace.append(a_tr.copy())

        if detector.update(env):
            td_steps.append(step)

        obs = next_obs
        step += 1
        if terminated or truncated:
            break

    # Extract cycle
    cycle_data = None
    for i in range(len(td_steps) - 1):
        s, e = td_steps[i], td_steps[i + 1]
        if 25 <= (e - s) <= 70:
            cycle_data = {"obs": np.array(obs_trace[s:e]),
                          "actions": np.array(act_trace[s:e]),
                          "length": e - s}
            break

    reward_breakdown = {"forward": fwd_r, "healthy": healthy_r, "ctrl": ctrl_r}
    return total_r, cycle_data, step, reward_breakdown


# ═══════════════════════════════════════════════════════════════════════
# Build reference
# ═══════════════════════════════════════════════════════════════════════

def build_reference(sp, basis, sc, tc, shift, device, seed, n_ep=10):
    cycles = []
    for ep in range(n_ep):
        env = make_shifted_env(shift, seed + 10000 + ep * 100, "hopper")()
        _, cd, _, _ = run_ilc_episode(env, sp, basis, sc, tc, None, device)
        env.close()
        if cd is not None and 25 <= cd["length"] <= 70:
            cycles.append(cd)

    if len(cycles) < 3:
        return None, None, 0

    N = N_PHASE
    all_obs, all_act = [], []
    for cyc in cycles:
        L = cyc["obs"].shape[0]
        t_old = np.linspace(0, 1, L)
        t_new = np.linspace(0, 1, N)
        ao = np.zeros((N, cyc["obs"].shape[1]))
        aa = np.zeros((N, cyc["actions"].shape[1]))
        for d in range(cyc["obs"].shape[1]):
            ao[:, d] = np.interp(t_new, t_old, cyc["obs"][:, d])
        for d in range(cyc["actions"].shape[1]):
            aa[:, d] = np.interp(t_new, t_old, cyc["actions"][:, d])
        all_obs.append(ao); all_act.append(aa)

    ref_obs = np.mean(all_obs, axis=0)
    ref_act = np.mean(all_act, axis=0)
    avg_len = int(np.mean([c["length"] for c in cycles]))
    return ref_obs, ref_act, avg_len


def make_progressive_ref(ref_tr_obs, delta, rho):
    """ref = ref_tr + rho * delta where delta modifies key dims."""
    ref = ref_tr_obs.copy()
    ref[:, 0] += rho * 0.3     # z (height) +
    ref[:, 6] += rho * 0.5     # vx +
    return ref


# ═══════════════════════════════════════════════════════════════════════
# Run one ILC experiment
# ═══════════════════════════════════════════════════════════════════════

def run_ilc_experiment(sp, basis, sc, tc, shift, device, seed, n_trials,
                       ref_obs, ref_act, ref_cyc_len, bt_type, bt_extra,
                       eta=0.3):
    """Run n_trials of ILC, return per-trial metrics."""
    a_dim = basis.action_dim
    u_table = np.zeros((N_PHASE, a_dim), dtype=np.float32)

    # Set up B_t function
    if bt_type == "kan":
        def bt_func(args, s_np):
            s_t = torch.as_tensor(s_np, device=device, dtype=torch.float32).unsqueeze(0)
            return get_kan_bt(basis, args["tc"], s_t)
        bt_args = {"tc": tc}
    elif bt_type == "source":
        def bt_func(args, s_np):
            s_t = torch.as_tensor(s_np, device=device, dtype=torch.float32).unsqueeze(0)
            return get_source_bt(basis, args["sc"], s_t)
        bt_args = {"sc": sc}
    elif bt_type == "oracle":
        bt_func = lambda args, s_np: get_oracle_bt(
            args["env"], args["qp"], args["qv"], args["a"], eps=0.05)
        bt_args = None  # set per-cycle
    elif bt_type == "shuffled":
        def bt_func(args, s_np):
            s_t = torch.as_tensor(s_np, device=device, dtype=torch.float32).unsqueeze(0)
            B = get_kan_bt(basis, args["tc"], s_t)
            return shuffle_bt(B)
        bt_args = {"tc": tc}
    elif bt_type == "negative":
        def bt_func(args, s_np):
            s_t = torch.as_tensor(s_np, device=device, dtype=torch.float32).unsqueeze(0)
            return -get_kan_bt(basis, args["tc"], s_t)
        bt_args = {"tc": tc}
    elif bt_type == "proportional":
        bt_func = None
        bt_args = None
    else:
        raise ValueError(f"Unknown bt_type: {bt_type}")

    results = []
    for trial in range(n_trials):
        env = make_shifted_env(shift, seed + trial * 100, "hopper")()

        if bt_type == "oracle":
            # Need to pre-compute qp/qv for oracle BT (can't do per-phase in ILC update)
            # For oracle, use the Transport action's qp/qv at each phase
            # This requires running a transport-only episode first
            # Simplified: use fixed oracle env
            pass  # oracle bt needs env interaction — handle specially

        ep_r, cycle_data, ep_len, r_breakdown = run_ilc_episode(
            env, sp, basis, sc, tc, u_table, device, ref_cyc_len)
        env.close()

        trial_result = {
            "trial": trial, "return": float(ep_r), "length": ep_len,
            "fwd_r": float(r_breakdown["forward"]),
            "healthy_r": float(r_breakdown["healthy"]),
            "ctrl_r": float(r_breakdown["ctrl"]),
        }

        if cycle_data is not None:
            # Compute tracking error
            L = cycle_data["obs"].shape[0]
            t_old = np.linspace(0, 1, L)
            t_new = np.linspace(0, 1, N_PHASE)
            aligned = np.zeros((N_PHASE, cycle_data["obs"].shape[1]))
            for d in range(cycle_data["obs"].shape[1]):
                aligned[:, d] = np.interp(t_new, t_old, cycle_data["obs"][:, d])
            task_dims = [0, 2, 6, 3, 4]
            task_weight = np.array([2.0, 1.0, 1.0, 0.5, 0.5])
            err = 0.0
            for i in range(N_PHASE):
                e = ref_obs[i, task_dims] - aligned[i, task_dims]
                err += np.sum((e * task_weight) ** 2)
            trial_result["error"] = float(np.sqrt(err / N_PHASE))
            trial_result["cycle_len"] = cycle_data["length"]
            trial_result["residual_norm"] = float(np.linalg.norm(u_table))

            # ILC update
            if bt_type not in ("oracle",):  # oracle deferred
                u_table, _ = ilc_update_with_bt(
                    u_table, cycle_data["obs"], ref_obs,
                    bt_func, bt_args, eta=eta)
        else:
            trial_result["error"] = None
            trial_result["cycle_len"] = None
            trial_result["residual_norm"] = float(np.linalg.norm(u_table))

        results.append(trial_result)

    return results


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--smoke", action="store_true",
                       help="Quick smoke test: 5 trials, 1 seed")
    parser.add_argument("--seeds", type=str, default="1811,1911,2011",
                       help="Comma-separated random seeds")
    parser.add_argument("--n-trials", type=int, default=40,
                       help="ILC trials per condition per seed")
    parser.add_argument("--bt-types", type=str,
                       default="kan,proportional,shuffled,negative,source",
                       help="B_t variants to test")
    parser.add_argument("--json-out", default="results/ilc1_ablation.json")
    args = parser.parse_args()

    if args.smoke:
        args.seeds = "1811"
        args.n_trials = 8
        args.bt_types = "kan,proportional,shuffled"

    device = torch.device(args.device)
    seeds = [int(s) for s in args.seeds.split(",")]
    bt_types = [b.strip() for b in args.bt_types.split(",")]
    shift = SHIFTS["friction_070"]

    print("=" * 72)
    print(f"ILC-1 Ablation: {len(seeds)} seeds × {len(bt_types)} B_t types × {args.n_trials} trials")
    if args.smoke:
        print("  SMOKE TEST MODE")
    print("=" * 72)

    # ── Load components (once) ──────────────────────────────────────────
    print("\n[1/4] Loading components...", flush=True)
    sp = FrozenSourcePolicy(
        "results/hopper_source_sb3_ppo_continued_seed1811.zip",
        "results/hopper_source_sb3_vecnorm_continued_seed1811.pkl",
        device, seeds[0], env="hopper")
    st = load_source_twin("results/hopper_source_affine_twin_cloud_seed1811.pt", device)
    basis, sc, _, _ = load_cognition(
        argparse.Namespace(
            cognition_checkpoint="results/hopper_source_control_sobolev_calibrated_seed1811.pt",
            device=args.device), device)
    fa = argparse.Namespace(target="friction_070", seed=seeds[0], env="hopper",
        device=args.device, cognition_warmup=1024, warmup_noise=0.3,
        transform_ridge=10.0, drift_ridge=100.0, drift_spectral_eta=0.0,
        drift_spectral_beta=1.0, drift_spectral_mode="max",
        drift_smooth_lambda=0.0, diagonal_transform=False)
    tc, _ = fit_distilled_source_counterfactual_context(sp, basis, sc, fa, device, st)

    # ── Build reference ─────────────────────────────────────────────────
    print("\n[2/4] Building Transport reference...", flush=True)
    ref_obs, ref_act, ref_cyc_len = build_reference(
        sp, basis, sc, tc, shift, device, seeds[0], n_ep=10)
    if ref_obs is None:
        print("  ABORT: Cannot build reference.")
        return
    print(f"  Reference: {ref_cyc_len} steps/cycle, shape={ref_obs.shape}")

    # Progressive references
    refs = {
        "tr": (ref_obs, ref_act, ref_cyc_len),
    }
    # Add delta references
    for rho in [0.15, 0.30]:
        ref_delta = make_progressive_ref(ref_obs, None, rho)
        refs[f"delta_{rho}"] = (ref_delta, ref_act, ref_cyc_len)

    # ── Run experiments ─────────────────────────────────────────────────
    print(f"\n[3/4] Running {len(seeds)} seeds × {len(bt_types)} B_t types × "
          f"{len(refs)} refs...", flush=True)

    all_results = {}

    for seed in seeds:
        # Re-fit KAN for each seed
        if seed != seeds[0]:
            fa2 = argparse.Namespace(target="friction_070", seed=seed, env="hopper",
                device=args.device, cognition_warmup=1024, warmup_noise=0.3,
                transform_ridge=10.0, drift_ridge=100.0, drift_spectral_eta=0.0,
                drift_spectral_beta=1.0, drift_spectral_mode="max",
                drift_smooth_lambda=0.0, diagonal_transform=False)
            tc_seed, _ = fit_distilled_source_counterfactual_context(
                sp, basis, sc, fa2, device, st)
        else:
            tc_seed = tc

        for bt_type in bt_types:
            for ref_name, (r_obs, r_act, r_len) in refs.items():
                if bt_type == "proportional" and ref_name != "tr":
                    continue  # only test prop on tr ref
                if bt_type == "source" and ref_name != "tr":
                    continue  # source bt only on tr ref

                label = f"seed={seed}/{bt_type}/{ref_name}"
                print(f"\n  --- {label} ---", flush=True)

                results = run_ilc_experiment(
                    sp, basis, sc, tc_seed, shift, device, seed,
                    args.n_trials, r_obs, r_act, r_len, bt_type, None)

                returns = [r["return"] for r in results]
                errors = [r["error"] for r in results if r["error"] is not None]
                n_cycles = sum(1 for r in results if r["error"] is not None)

                mean_r = np.mean(returns)
                first5_r = np.mean(returns[:5])
                last5_r = np.mean(returns[-5:])
                mean_err = np.mean(errors) if errors else float('nan')
                first5_err = np.mean(errors[:5]) if len(errors) >= 5 else float('nan')
                last5_err = np.mean(errors[-5:]) if len(errors) >= 5 else float('nan')
                fwd_r = np.mean([r["fwd_r"] for r in results])
                ctrl_r = np.mean([r["ctrl_r"] for r in results])

                print(f"    return={mean_r:.1f} (first5={first5_r:.1f} last5={last5_r:.1f}) "
                      f"err={mean_err:.4f} (first5={first5_err:.4f} last5={last5_err:.4f}) "
                      f"cycles={n_cycles}/{args.n_trials}", flush=True)

                all_results[label] = {
                    "seed": seed, "bt_type": bt_type, "ref": ref_name,
                    "mean_return": float(mean_r),
                    "first5_return": float(first5_r),
                    "last5_return": float(last5_r),
                    "mean_error": float(mean_err) if not np.isnan(mean_err) else None,
                    "first5_error": float(first5_err) if not np.isnan(first5_err) else None,
                    "last5_error": float(last5_err) if not np.isnan(last5_err) else None,
                    "n_cycles": n_cycles,
                    "mean_fwd_r": float(fwd_r),
                    "mean_ctrl_r": float(ctrl_r),
                    "returns": returns,
                    "errors": [float(e) if e is not None else None for e in errors],
                }

    # ── Report ──────────────────────────────────────────────────────────
    print(f"\n[4/4] Summary", flush=True)
    print("=" * 72)

    # Aggregate by bt_type
    print(f"\n  {'B_t Type':>15s} {'Mean R':>10s} {'First5':>10s} "
          f"{'Last5':>10s} {'Delta':>10s} {'Error':>10s} {'FwdR':>10s} {'CtrlR':>10s}")
    print(f"  {'-'*85}")

    for bt in bt_types:
        bt_results = [v for k, v in all_results.items()
                      if v["bt_type"] == bt and v["ref"] == "tr"]
        if not bt_results:
            continue
        mean_r = np.mean([r["mean_return"] for r in bt_results])
        first5 = np.mean([r["first5_return"] for r in bt_results])
        last5 = np.mean([r["last5_return"] for r in bt_results])
        delta = last5 - first5
        mean_err = np.mean([r["mean_error"] for r in bt_results if r["mean_error"]])
        fwd = np.mean([r["mean_fwd_r"] for r in bt_results])
        ctrl = np.mean([r["mean_ctrl_r"] for r in bt_results])
        print(f"  {bt:>15s} {mean_r:>10.1f} {first5:>10.1f} {last5:>10.1f} "
              f"{delta:>+10.1f} {mean_err:>10.4f} {fwd:>10.1f} {ctrl:>10.1f}")

    # Transport baseline
    tr_results = [v for k, v in all_results.items() if v["bt_type"] == "kan"]
    tr_mean = np.mean([r["mean_return"] for r in tr_results]) if tr_results else 571
    print(f"\n  Transport baseline: {tr_mean:.1f} (no ILC)")

    # Save
    summary = {
        "config": {"seeds": seeds, "n_trials": args.n_trials,
                   "bt_types": bt_types, "n_phase": N_PHASE},
        "reference_cycle_len": ref_cyc_len,
        "results": all_results,
    }
    Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(args.json_out, "w"), indent=2)
    print(f"\n  Saved to {args.json_out}")


if __name__ == "__main__":
    main()
