"""ILC-1: Cycle-level reference tracking with KAN-B_t learning gain.

Trial-to-trial ILC: each episode provides one cycle, error computed
against reference, feedforward u_k(φ) updated for next episode.

Comparisons:
  A. Transport only (no ILC)
  B. Transport + proportional ILC (scalar gain)
  C. Transport + KAN-B_t ILC (KAN provides learning gain matrix)

References:
  1. Transport self-reference (validation: should stay near zero error)
  2. Progressive blend: (1-β)*y_Tr + β*y_source
"""

from __future__ import annotations

import argparse, sys, os, json
from pathlib import Path
from collections import deque

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.diagnose_hopper_pullback_effect import (
    fit_distilled_source_counterfactual_context,
    load_source_twin,
)
from scripts.prescreen_hopper_physics_shifts import SHIFTS, make_shifted_env
from scripts.validate_hopper_joint_online_adaptation import (
    FrozenSourcePolicy,
    load_cognition,
)

N_PHASE = 20  # uniform phase grid points per cycle


# ═══════════════════════════════════════════════════════════════════════
# Improved cycle detector with hysteresis
# ═══════════════════════════════════════════════════════════════════════

class ImprovedCycleDetector:
    """Detect gait cycles from foot_z local minima.

    On friction_070, Transport produces a flight-dominant gait where
    the foot touches ground only once per episode. Cycles are defined
    by the periodic oscillation of foot_z during flight.

    Cycle boundary = local minimum of foot_z (trough between oscillations).
    """

    def __init__(self, min_interval=25, max_interval=50, window=5):
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.window = window  # half-window for local min detection
        self._last_cyc_step = -min_interval
        self._step = 0
        self._foot_z_history = []

    def reset(self):
        self._last_cyc_step = -self.min_interval
        self._step = 0
        self._foot_z_history = []

    def update(self, env):
        """Call after env.step(). Returns True at cycle boundary (foot_z local min)."""
        try:
            foot_z = float(env.unwrapped.data.body("foot").xpos[2])
        except Exception:
            foot_z = 0.0
        self._foot_z_history.append(foot_z)
        self._step += 1

        # Need enough history for local min detection
        w = self.window
        if len(self._foot_z_history) < 2 * w + 1:
            return False

        # Check if middle point is local minimum
        recent = self._foot_z_history[-(2 * w + 1):]
        mid_val = recent[w]
        left_vals = recent[:w]
        right_vals = recent[w+1:]

        is_local_min = (mid_val < np.min(left_vals) and mid_val <= np.min(right_vals))

        is_cyc = False
        if is_local_min:
            if (self._step - self._last_cyc_step) >= self.min_interval:
                is_cyc = True
                self._last_cyc_step = self._step

        return is_cyc


# ═══════════════════════════════════════════════════════════════════════
# Reference collection
# ═══════════════════════════════════════════════════════════════════════

def collect_transport_reference(source_policy, basis, source_context, target_ctx,
                                shift, device, n_episodes=10, seed=1811):
    """Collect Transport cycles and build phase-aligned reference."""
    detector = ImprovedCycleDetector()
    all_aligned_obs = []
    all_aligned_actions = []

    for ep in range(n_episodes):
        env = make_shifted_env(shift, seed + ep * 100, "hopper")()
        obs, _ = env.reset()
        detector.reset()
        obs_trace, act_trace = [], []

        while True:
            s_t = torch.as_tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
            nominal = source_policy.action(s_t)
            s_eff = source_context.acceleration(basis, s_t, nominal)
            a_tr = target_ctx.transport_action(
                basis, s_t, desired_effect=s_eff,
                nominal_action=nominal, regularization=1e-2,
            ).clamp(-1, 1).squeeze(0).cpu().numpy()
            next_obs, _, terminated, truncated, _ = env.step(a_tr)
            obs_trace.append(obs.copy())
            act_trace.append(a_tr.copy())
            detector.update(env)
            obs = next_obs
            if terminated or truncated:
                break
        env.close()

        obs_arr = np.array(obs_trace)
        act_arr = np.array(act_trace)

        # Find touchdown indices
        td_idx = []
        detector2 = ImprovedCycleDetector()
        detector2.reset()
        for i in range(len(obs_trace)):
            # Replay foot detection
            pass  # Can't replay easily — use stored foot_z

        # Use the detector's internal state to find TD
        # Simpler: run detector again on the stored obs
        # Actually let's just use foot_z from the env during collection
        # We can store foot_z alongside obs

        # For now, find cycles using a simple heuristic:
        # touchdown = foot_z crosses below threshold after being above
        pass  # Will implement inline

    return None  # placeholder


def build_transport_reference_from_cycles(cycles, N=N_PHASE):
    """Phase-align and average cycles to create reference trajectory.

    Args:
        cycles: list of {"obs": (L, s_dim), "actions": (L, a_dim)}

    Returns:
        ref_obs: (N, s_dim), ref_actions: (N, a_dim)
    """
    all_obs = []
    all_act = []
    for cyc in cycles:
        L = cyc["obs"].shape[0]
        t_old = np.linspace(0, 1, L)
        t_new = np.linspace(0, 1, N)
        aligned_obs = np.zeros((N, cyc["obs"].shape[1]))
        aligned_act = np.zeros((N, cyc["actions"].shape[1]))
        for d in range(cyc["obs"].shape[1]):
            aligned_obs[:, d] = np.interp(t_new, t_old, cyc["obs"][:, d])
        for d in range(cyc["actions"].shape[1]):
            aligned_act[:, d] = np.interp(t_new, t_old, cyc["actions"][:, d])
        all_obs.append(aligned_obs)
        all_act.append(aligned_act)

    ref_obs = np.mean(all_obs, axis=0)
    ref_act = np.mean(all_act, axis=0)
    return ref_obs, ref_act


# ═══════════════════════════════════════════════════════════════════════
# ILC update
# ═══════════════════════════════════════════════════════════════════════

def compute_kan_bt(basis, target_ctx, s_tensor):
    """Extract KAN B_t = dF/da at state."""
    with torch.no_grad():
        _, gain = target_ctx.drift_and_gain(basis, s_tensor)
        return gain.squeeze(0)  # (s_dim, a_dim)


def compute_true_bt(fd_env, qp, qv, a_nominal, eps=0.05):
    """Finite-difference true B_t at state."""
    s_dim = 11  # Hopper obs dim
    a_dim = len(a_nominal)
    B = np.zeros((s_dim, a_dim))
    fd_env.unwrapped.set_state(qp, qv)
    # Baseline next state
    obs_base, _, _, _, _ = fd_env.step(a_nominal)
    for dim in range(a_dim):
        da = np.zeros(a_dim)
        da[dim] = eps
        a_pert = np.clip(a_nominal + da, -1, 1)
        fd_env.unwrapped.set_state(qp, qv)
        obs_pert, _, _, _, _ = fd_env.step(a_pert)
        B[:, dim] = (obs_pert - obs_base) / eps
    return B


def ilc_update(u_k, cycle_obs, cycle_actions, ref_obs, ref_actions,
               basis, target_ctx, device, eta=0.3, beta_forget=0.1,
               lam=0.1, N=N_PHASE):
    """One ILC update: compute phase-aligned error, map to action correction.

    Args:
        u_k: (N, a_dim) current feedforward table
        cycle_obs: (L, s_dim) actual cycle observations
        cycle_actions: (L, a_dim) actual cycle actions
        ref_obs: (N, s_dim) reference observations
        ref_actions: (N, a_dim) reference actions
        eta: learning rate
        beta_forget: forgetting factor
        lam: regularization for (B^T B + lam I)^{-1}

    Returns:
        u_{k+1}: (N, a_dim) updated feedforward table
        error_norm: scalar tracking error
    """
    L = cycle_obs.shape[0]
    s_dim = cycle_obs.shape[1]
    a_dim = cycle_actions.shape[1]

    # Phase-align actual cycle to reference grid
    t_old = np.linspace(0, 1, L)
    t_new = np.linspace(0, 1, N)
    aligned_obs = np.zeros((N, s_dim))
    aligned_act = np.zeros((N, a_dim))
    for d in range(s_dim):
        aligned_obs[:, d] = np.interp(t_new, t_old, cycle_obs[:, d])
    for d in range(a_dim):
        aligned_act[:, d] = np.interp(t_new, t_old, cycle_actions[:, d])

    # Task-weighted error (subset of state dimensions)
    # Use: z, cos(a), vx, thigh, leg (key locomotion dimensions)
    task_dims = [0, 2, 6, 3, 4]  # z, cos(angle), vx, thigh, leg
    task_weight = np.array([2.0, 1.0, 1.0, 0.5, 0.5])

    error_norm = 0.0
    u_new = u_k.copy()

    for i in range(N):
        # Error at this phase (task dimensions only)
        e_full = ref_obs[i] - aligned_obs[i]
        e_task = e_full[task_dims]
        e_weighted = e_task * task_weight
        error_norm += np.sum(e_weighted ** 2)

        # KAN B_t at this phase's state
        s_t = torch.as_tensor(aligned_obs[i], device=device,
                             dtype=torch.float32).unsqueeze(0)
        B = compute_kan_bt(basis, target_ctx, s_t).cpu().numpy()  # (s_dim, a_dim)
        B_task = B[task_dims, :]  # (task_dim, a_dim)

        # Learning gain: (B^T Q B + lam I)^{-1} B^T Q
        Q_diag = np.diag(task_weight)
        BtQB = B_task.T @ Q_diag @ B_task  # (a_dim, a_dim)
        reg_matrix = BtQB + lam * np.eye(a_dim)
        gain = np.linalg.solve(reg_matrix, B_task.T @ Q_diag @ e_weighted)

        # Update with forgetting
        u_new[i] = (1.0 - beta_forget) * u_k[i] + eta * gain

    # Clip feedforward
    u_new = np.clip(u_new, -0.15, 0.15)

    # Smooth across phases
    for d in range(a_dim):
        u_new[:, d] = np.convolve(u_new[:, d], [0.25, 0.5, 0.25], mode='same')

    return u_new, np.sqrt(error_norm / N)


def proportional_ilc_update(u_k, cycle_obs, ref_obs, ref_actions,
                           eta_p=0.05, beta_forget=0.1, N=N_PHASE):
    """Simple proportional ILC (no KAN B_t)."""
    L = cycle_obs.shape[0]
    s_dim = cycle_obs.shape[1]
    a_dim = ref_actions.shape[1]
    task_dims = [0, 2, 6, 3, 4]
    task_weight = np.array([2.0, 1.0, 1.0, 0.5, 0.5])

    t_old = np.linspace(0, 1, L)
    t_new = np.linspace(0, 1, N)
    aligned_obs = np.zeros((N, s_dim))
    for d in range(s_dim):
        aligned_obs[:, d] = np.interp(t_new, t_old, cycle_obs[:, d])

    error_norm = 0.0
    u_new = u_k.copy()

    for i in range(N):
        e_full = ref_obs[i] - aligned_obs[i]
        e_task = e_full[task_dims]
        e_weighted = e_task * task_weight
        error_norm += np.sum(e_weighted ** 2)

        # Heuristic: map task error to action using fixed PD-like gains
        # z error → thigh/leg action; vx error → thigh action
        delta_u = np.zeros(a_dim)
        delta_u[0] = eta_p * (0.5 * e_weighted[0] + 0.3 * e_weighted[2])  # thigh
        delta_u[1] = eta_p * (0.5 * e_weighted[0] + 0.2 * e_weighted[3])  # leg
        delta_u[2] = eta_p * (0.3 * e_weighted[3])  # foot

        u_new[i] = (1.0 - beta_forget) * u_k[i] + delta_u

    u_new = np.clip(u_new, -0.15, 0.15)
    return u_new, np.sqrt(error_norm / N)


# ═══════════════════════════════════════════════════════════════════════
# Episode runner with ILC feedforward
# ═══════════════════════════════════════════════════════════════════════

def run_ilc_episode(env, source_policy, basis, source_context, target_ctx,
                    u_table, device, ref_cycle_len=55):
    """Run episode with Transport + ILC feedforward, extract one cycle.

    Args:
        u_table: (N_PHASE, a_dim) current ILC feedforward
        ref_cycle_len: expected cycle length for phase normalization

    Returns:
        total_return, cycle_data, episode_length
    """
    obs, _ = env.reset()
    total_r = 0.0
    detector = ImprovedCycleDetector()
    detector.reset()

    obs_trace = []
    act_trace = []
    foot_z_trace = []
    td_steps = []

    step = 0
    while True:
        s_t = torch.as_tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
        nominal = source_policy.action(s_t)
        s_eff = source_context.acceleration(basis, s_t, nominal)
        a_tr = target_ctx.transport_action(
            basis, s_t, desired_effect=s_eff,
            nominal_action=nominal, regularization=1e-2,
        ).clamp(-1, 1).squeeze(0).cpu().numpy()

        # ILC feedforward: interpolate u at current phase
        if u_table is not None and len(td_steps) > 0:
            # Phase = steps since last touchdown / ref_cycle_len
            steps_since_td = step - td_steps[-1]
            phase = min(steps_since_td / ref_cycle_len, 0.99)
            phi_idx = int(phase * N_PHASE)
            u_ff = u_table[min(phi_idx, N_PHASE - 1)]
            a_final = np.clip(a_tr + u_ff, -1, 1)
        else:
            a_final = a_tr

        next_obs, reward, terminated, truncated, _ = env.step(a_final)
        total_r += float(reward)

        obs_trace.append(obs.copy())
        act_trace.append(a_tr.copy())  # store transport action (not corrected)

        try:
            foot_z_trace.append(float(env.unwrapped.data.body("foot").xpos[2]))
        except Exception:
            foot_z_trace.append(0.5)

        is_td = detector.update(env)
        if is_td:
            td_steps.append(step)

        obs = next_obs
        step += 1
        if terminated or truncated:
            break

    # Extract best cycle: from first clean touchdown pair
    cycle_data = None
    for i in range(len(td_steps) - 1):
        start, end = td_steps[i], td_steps[i + 1]
        cyc_len = end - start
        if 25 <= cyc_len <= 70:  # valid oscillation cycle
            cycle_data = {
                "obs": np.array(obs_trace[start:end]),
                "actions": np.array(act_trace[start:end]),
                "foot_z": np.array(foot_z_trace[start:end]),
                "start_step": start,
                "length": cyc_len,
            }
            break

    return total_r, cycle_data, step


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=1811)
    parser.add_argument("--n-ilc-episodes", type=int, default=40,
                       help="Number of ILC trials (episodes)")
    parser.add_argument("--n-ref-episodes", type=int, default=10,
                       help="Episodes for building Transport reference")
    parser.add_argument("--eta", type=float, default=0.3, help="ILC learning rate")
    parser.add_argument("--beta-blend", type=float, default=0.3,
                       help="Blend ratio for progressive reference (0=Tr, 1=source)")
    parser.add_argument("--json-out", default="results/ilc1_reference_tracking.json")
    args = parser.parse_args()

    device = torch.device(args.device)
    shift = SHIFTS["friction_070"]

    print("=" * 72)
    print("ILC-1: Cycle-Level Reference Tracking")
    print(f"  n_ilc_episodes={args.n_ilc_episodes}, eta={args.eta}")
    print(f"  blend_beta={args.beta_blend}")
    print("=" * 72)

    # ── Load ────────────────────────────────────────────────────────────
    print("\n[1/4] Loading components...", flush=True)
    source_policy = FrozenSourcePolicy(
        "results/hopper_source_sb3_ppo_continued_seed1811.zip",
        "results/hopper_source_sb3_vecnorm_continued_seed1811.pkl",
        device, args.seed, env="hopper",
    )
    source_twin = load_source_twin(
        "results/hopper_source_affine_twin_cloud_seed1811.pt", device,
    )
    basis, source_context, _, _ = load_cognition(
        argparse.Namespace(
            cognition_checkpoint="results/hopper_source_control_sobolev_calibrated_seed1811.pt",
            device=args.device,
        ),
        device,
    )
    fit_args = argparse.Namespace(
        target="friction_070", seed=args.seed, env="hopper", device=args.device,
        cognition_warmup=1024, warmup_noise=0.3,
        transform_ridge=10.0, drift_ridge=100.0,
        drift_spectral_eta=0.0, drift_spectral_beta=1.0,
        drift_spectral_mode="max", drift_smooth_lambda=0.0, diagonal_transform=False,
    )
    target_ctx, _ = fit_distilled_source_counterfactual_context(
        source_policy, basis, source_context, fit_args, device, source_twin,
    )

    # ── Build reference ─────────────────────────────────────────────────
    print(f"\n[2/4] Building references ({args.n_ref_episodes} episodes each)...",
          flush=True)

    # Collect Transport cycles for reference
    tr_cycles = []
    for ep in range(args.n_ref_episodes):
        env = make_shifted_env(shift, args.seed + 10000 + ep * 100, "hopper")()
        _, cycle_data, _ = run_ilc_episode(
            env, source_policy, basis, source_context, target_ctx,
            None, device, ref_cycle_len=36,
        )
        env.close()
        if cycle_data is not None and 25 <= cycle_data["length"] <= 70:
            tr_cycles.append(cycle_data)

    if len(tr_cycles) < 3:
        print(f"  ABORT: Only {len(tr_cycles)} valid Transport cycles. Cannot build reference.")
        return

    ref_obs_tr, ref_act_tr = build_transport_reference_from_cycles(tr_cycles)
    ref_cycle_len = int(np.mean([c["length"] for c in tr_cycles]))
    print(f"  Transport reference: {len(tr_cycles)} cycles, "
          f"avg_len={ref_cycle_len}, ref_shape={ref_obs_tr.shape}", flush=True)

    # Collect source cycles
    src_cycles = []
    src_shift = SHIFTS["source"]
    for ep in range(args.n_ref_episodes):
        env = make_shifted_env(src_shift, args.seed + 20000 + ep * 100, "hopper")()
        obs, _ = env.reset()
        obs_trace, act_trace = [], []
        detector = ImprovedCycleDetector()
        detector.reset()
        td_steps = []
        step = 0
        while True:
            s_t = torch.as_tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
            a_src = source_policy.action(s_t).cpu().numpy().squeeze(0)
            next_obs, _, terminated, truncated, _ = env.step(a_src)
            obs_trace.append(obs.copy())
            act_trace.append(a_src.copy())
            is_td = detector.update(env)
            if is_td:
                td_steps.append(step)
            obs = next_obs
            step += 1
            if terminated or truncated:
                break
        env.close()

        obs_arr = np.array(obs_trace)
        act_arr = np.array(act_trace)
        for i in range(len(td_steps) - 1):
            start, end = td_steps[i], td_steps[i + 1]
            if 25 <= (end - start) <= 70:
                src_cycles.append({
                    "obs": obs_arr[start:end],
                    "actions": act_arr[start:end],
                    "length": end - start,
                })
                break

    if len(src_cycles) >= 3:
        ref_obs_src, ref_act_src = build_transport_reference_from_cycles(src_cycles)
        src_cycle_len = int(np.mean([c["length"] for c in src_cycles]))
        print(f"  Source reference: {len(src_cycles)} cycles, "
              f"avg_len={src_cycle_len}", flush=True)
    else:
        print(f"  WARNING: Only {len(src_cycles)} source cycles. Using Transport as fallback.")
        ref_obs_src, ref_act_src = ref_obs_tr, ref_act_tr
        src_cycle_len = ref_cycle_len

    # Progressive blend reference
    beta = args.beta_blend
    ref_obs_blend = (1.0 - beta) * ref_obs_tr + beta * ref_obs_src
    ref_act_blend = (1.0 - beta) * ref_act_tr + beta * ref_act_src
    blend_cycle_len = int((1.0 - beta) * ref_cycle_len + beta * src_cycle_len)

    # ── Run ILC experiments ─────────────────────────────────────────────
    print(f"\n[3/4] Running ILC trials ({args.n_ilc_episodes} episodes each)...",
          flush=True)

    configs = [
        ("Transport only", "transport", None, ref_obs_tr, ref_act_tr, ref_cycle_len,
         False),
        ("Proportional ILC (Tr ref)", "prop_ilc_tr", ref_obs_tr, ref_obs_tr,
         ref_act_tr, ref_cycle_len, True),
        ("KAN-B_t ILC (Tr ref)", "kan_ilc_tr", ref_obs_tr, ref_obs_tr,
         ref_act_tr, ref_cycle_len, True),
        ("KAN-B_t ILC (blend ref)", "kan_ilc_blend", ref_obs_blend, ref_obs_blend,
         ref_act_blend, blend_cycle_len, True),
    ]

    all_results = {}

    for label, key, u_init_ref, ref_obs, ref_act, cyc_len, use_ilc in configs:
        print(f"\n  --- {label} ---", flush=True)

        # Initialize feedforward table
        if u_init_ref is not None:
            u_table = np.zeros((N_PHASE, basis.action_dim), dtype=np.float32)
        else:
            u_table = np.zeros((N_PHASE, basis.action_dim), dtype=np.float32)

        returns = []
        errors = []
        cycle_lengths = []
        n_cycles_found = 0

        for ep in range(args.n_ilc_episodes):
            env = make_shifted_env(shift, args.seed + ep * 100, "hopper")()

            if key == "transport":
                ep_r, cycle_data, ep_len = run_ilc_episode(
                    env, source_policy, basis, source_context, target_ctx,
                    None, device, ref_cycle_len=cyc_len,
                )
            else:
                ep_r, cycle_data, ep_len = run_ilc_episode(
                    env, source_policy, basis, source_context, target_ctx,
                    u_table, device, ref_cycle_len=cyc_len,
                )

            env.close()
            returns.append(ep_r)

            if cycle_data is not None:
                n_cycles_found += 1
                cycle_lengths.append(cycle_data["length"])

                # Compute tracking error
                L = cycle_data["obs"].shape[0]
                t_old = np.linspace(0, 1, L)
                t_new = np.linspace(0, 1, N_PHASE)
                aligned_obs = np.zeros((N_PHASE, cycle_data["obs"].shape[1]))
                for d in range(cycle_data["obs"].shape[1]):
                    aligned_obs[:, d] = np.interp(
                        t_new, t_old, cycle_data["obs"][:, d])
                task_dims = [0, 2, 6, 3, 4]
                task_weight = np.array([2.0, 1.0, 1.0, 0.5, 0.5])
                err = 0.0
                for i in range(N_PHASE):
                    e = ref_obs[i, task_dims] - aligned_obs[i, task_dims]
                    err += np.sum((e * task_weight) ** 2)
                errors.append(np.sqrt(err / N_PHASE))

                # ILC update
                if use_ilc and key.startswith("kan_ilc"):
                    u_table, _ = ilc_update(
                        u_table, cycle_data["obs"], cycle_data["actions"],
                        ref_obs, ref_act,
                        basis, target_ctx, device,
                        eta=args.eta,
                    )
                elif use_ilc and key.startswith("prop_ilc"):
                    u_table, _ = proportional_ilc_update(
                        u_table, cycle_data["obs"], ref_obs, ref_act,
                        eta_p=0.05,
                    )

            if (ep + 1) % 10 == 0:
                mean_r = np.mean(returns)
                mean_err = np.mean(errors[-10:]) if errors else float('nan')
                print(f"    ep {ep+1}: return={mean_r:.1f}, "
                      f"err={mean_err:.4f}, "
                      f"cycles_found={n_cycles_found}",
                      flush=True)

        mean_r = np.mean(returns)
        mean_err = np.mean(errors) if errors else float('nan')
        mean_cyc_len = np.mean(cycle_lengths) if cycle_lengths else 0
        print(f"    Result: return={mean_r:.1f}, "
              f"error={mean_err:.4f}, "
              f"cycles={n_cycles_found}/{args.n_ilc_episodes}, "
              f"mean_cycle_len={mean_cyc_len:.1f}",
              flush=True)

        all_results[label] = {
            "key": key,
            "mean_return": float(mean_r),
            "returns": [float(r) for r in returns],
            "mean_error": float(mean_err) if not np.isnan(mean_err) else None,
            "errors": [float(e) for e in errors],
            "n_cycles_found": n_cycles_found,
            "mean_cycle_len": float(mean_cyc_len),
        }

    # ── Report ──────────────────────────────────────────────────────────
    print(f"\n[4/4] Summary", flush=True)
    print("=" * 72)
    print(f"\n  {'Condition':35s} {'Return':>10s} {'Error':>10s} "
          f"{'Cycles':>8s}")
    print(f"  {'-'*65}")

    for label, res in all_results.items():
        err_str = f"{res['mean_error']:.4f}" if res['mean_error'] else "N/A"
        print(f"  {label:35s} {res['mean_return']:>10.1f} {err_str:>10s} "
              f"{res['n_cycles_found']:>8d}")

    # Check: does error decrease over trials for ILC methods?
    print(f"\n  Error convergence (first 10 vs last 10 trials):")
    for label, res in all_results.items():
        if res['errors'] and len(res['errors']) >= 20:
            first10 = np.mean(res['errors'][:10])
            last10 = np.mean(res['errors'][-10:])
            delta = last10 - first10
            print(f"    {label:35s}: {first10:.4f} → {last10:.4f} "
                  f"({delta:+.4f})")

    # Save
    summary = {
        "config": {"n_ilc_episodes": args.n_ilc_episodes,
                   "eta": args.eta, "beta_blend": args.beta_blend,
                   "n_phase": N_PHASE},
        "reference": {
            "transport_cycle_len": ref_cycle_len,
            "source_cycle_len": src_cycle_len,
            "blend_cycle_len": blend_cycle_len,
            "n_tr_cycles": len(tr_cycles),
            "n_src_cycles": len(src_cycles),
        },
        "results": all_results,
    }
    Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(args.json_out, "w"), indent=2)
    print(f"\n  Saved to {args.json_out}")


if __name__ == "__main__":
    main()
