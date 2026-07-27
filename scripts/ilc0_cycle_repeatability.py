"""ILC-0: Cycle repeatability audit.

Tests whether Hopper gait cycles under Transport on friction_070 are
repeatable enough for phase-domain ILC.

Key questions:
  1. Contact event detection reliability
  2. Cycle length mean and variance
  3. State variance after phase alignment
  4. Number of complete cycles per episode
  5. Cycle-to-cycle state difference at key phases
"""

from __future__ import annotations

import argparse, sys, os, json
from pathlib import Path

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


# ═══════════════════════════════════════════════════════════════════════
# Cycle detector
# ═══════════════════════════════════════════════════════════════════════

class CycleDetector:
    """Detect gait cycles from foot contact events.

    A cycle is defined as touchdown → next touchdown (one full hop).
    """

    def __init__(self, foot_z_threshold=0.12, min_cycle_len=10, max_cycle_len=60):
        self.threshold = foot_z_threshold
        self.min_cycle_len = min_cycle_len
        self.max_cycle_len = max_cycle_len
        self._prev_in_contact = False
        self._foot_z_trace = []
        self._torso_z_trace = []

    def reset(self):
        self._prev_in_contact = False
        self._foot_z_trace = []
        self._torso_z_trace = []

    def update(self, env):
        """Call after env.step(). Returns True at touchdown (cycle boundary)."""
        try:
            foot_z = float(env.unwrapped.data.body("foot").xpos[2])
        except Exception:
            foot_z = 0.0

        try:
            torso_z = float(env.unwrapped.data.body("torso").xpos[2])
        except Exception:
            torso_z = 0.0

        self._foot_z_trace.append(foot_z)
        self._torso_z_trace.append(torso_z)

        in_contact = foot_z < self.threshold
        is_touchdown = in_contact and not self._prev_in_contact
        self._prev_in_contact = in_contact
        return is_touchdown


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=1811)
    parser.add_argument("--n-episodes", type=int, default=10)
    parser.add_argument("--json-out", default="results/ilc0_cycle_repeatability.json")
    args = parser.parse_args()

    device = torch.device(args.device)
    shift = SHIFTS["friction_070"]

    print("=" * 72)
    print("ILC-0: Cycle Repeatability Audit")
    print("=" * 72)

    # Load
    print("\n[1/2] Loading components...", flush=True)
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

    # ── Collect cycles ──────────────────────────────────────────────────
    print(f"\n[2/2] Collecting cycles from {args.n_episodes} episodes...", flush=True)

    all_cycles = []       # list of (obs_array, action_array, foot_z_array)
    all_cycle_lengths = []
    all_touchdown_states = []
    episode_summaries = []

    for ep in range(args.n_episodes):
        env = make_shifted_env(shift, args.seed + ep * 100, "hopper")()
        obs, _ = env.reset()
        detector = CycleDetector()
        detector.reset()

        # Trace for current episode
        ep_obs = []
        ep_actions = []
        ep_foot_z = []
        ep_torso_z = []
        touchdown_indices = []

        step = 0
        while True:
            s_t = torch.as_tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
            nominal = source_policy.action(s_t)
            s_eff = source_context.acceleration(basis, s_t, nominal)
            a_tr = target_ctx.transport_action(
                basis, s_t, desired_effect=s_eff,
                nominal_action=nominal, regularization=1e-2,
            ).clamp(-1, 1).squeeze(0).cpu().numpy()

            next_obs, reward, terminated, truncated, _ = env.step(a_tr)

            ep_obs.append(obs.copy())
            ep_actions.append(a_tr.copy())

            is_td = detector.update(env)
            try:
                ep_foot_z.append(float(env.unwrapped.data.body("foot").xpos[2]))
                ep_torso_z.append(float(env.unwrapped.data.body("torso").xpos[2]))
            except Exception:
                ep_foot_z.append(0.0)
                ep_torso_z.append(0.0)

            if is_td:
                touchdown_indices.append(step)

            obs = next_obs
            step += 1
            if terminated or truncated:
                break

        env.close()

        ep_obs = np.array(ep_obs)
        ep_actions = np.array(ep_actions)
        ep_foot_z = np.array(ep_foot_z)
        ep_torso_z = np.array(ep_torso_z)

        # Extract cycles: touchdown[i] → touchdown[i+1]
        n_cycles = len(touchdown_indices) - 1
        ep_cycle_lengths = []
        for i in range(n_cycles):
            start = touchdown_indices[i]
            end = touchdown_indices[i + 1]
            cycle_len = end - start
            if 8 <= cycle_len <= 50:  # reasonable hop cycle length
                ep_cycle_lengths.append(cycle_len)
                all_cycle_lengths.append(cycle_len)
                all_cycles.append({
                    "obs": ep_obs[start:end],
                    "actions": ep_actions[start:end],
                    "foot_z": ep_foot_z[start:end],
                    "torso_z": ep_torso_z[start:end],
                    "length": cycle_len,
                })
                all_touchdown_states.append(ep_obs[start])

        episode_summaries.append({
            "episode": ep,
            "total_steps": step,
            "n_touchdowns": len(touchdown_indices),
            "n_cycles": n_cycles,
            "valid_cycles": len(ep_cycle_lengths),
            "cycle_lengths": ep_cycle_lengths,
            "mean_cycle_len": float(np.mean(ep_cycle_lengths)) if ep_cycle_lengths else 0,
            "std_cycle_len": float(np.std(ep_cycle_lengths)) if ep_cycle_lengths else 0,
        })

        print(f"  ep {ep+1}: {step} steps, {len(touchdown_indices)} touchdowns, "
              f"{len(ep_cycle_lengths)} valid cycles, "
              f"mean_len={np.mean(ep_cycle_lengths) if ep_cycle_lengths else 0:.1f}",
              flush=True)

    n_total_cycles = len(all_cycle_lengths)
    if n_total_cycles < 5:
        print(f"\n  ABORT: Only {n_total_cycles} valid cycles. Contact detection may be wrong.")
        return

    # ── Analysis ────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"Cycle Repeatability Analysis ({n_total_cycles} cycles)")

    cycle_lengths = np.array(all_cycle_lengths)
    print(f"\n  Cycle Length: mean={cycle_lengths.mean():.1f}, "
          f"std={cycle_lengths.std():.1f}, "
          f"min={cycle_lengths.min()}, max={cycle_lengths.max()}, "
          f"CV={cycle_lengths.std()/cycle_lengths.mean():.3f}")

    # Phase-align cycles: resample each cycle to N uniform phase points
    N_PHASE = 20
    all_aligned = []  # (n_cycles, N_PHASE, s_dim)
    all_aligned_foot_z = []
    all_aligned_torso_z = []

    for cycle in all_cycles:
        t = np.linspace(0, 1, cycle["length"])
        t_new = np.linspace(0, 1, N_PHASE)
        aligned = np.zeros((N_PHASE, cycle["obs"].shape[1]))
        foot_aligned = np.zeros(N_PHASE)
        torso_aligned = np.zeros(N_PHASE)
        for d in range(cycle["obs"].shape[1]):
            aligned[:, d] = np.interp(t_new, t, cycle["obs"][:, d])
        foot_aligned = np.interp(t_new, t, cycle["foot_z"])
        torso_aligned = np.interp(t_new, t, cycle["torso_z"])
        all_aligned.append(aligned)
        all_aligned_foot_z.append(foot_aligned)
        all_aligned_torso_z.append(torso_aligned)

    all_aligned = np.array(all_aligned)  # (n_cycles, N_PHASE, s_dim)
    all_aligned_foot_z = np.array(all_aligned_foot_z)
    all_aligned_torso_z = np.array(all_aligned_torso_z)

    # Per-phase state variance
    phase_std = all_aligned.std(axis=0)  # (N_PHASE, s_dim)
    mean_phase_std = phase_std.mean(axis=1)  # average over state dims
    mean_state_std = phase_std.mean()  # scalar

    print(f"\n  Phase-aligned state variability:")
    print(f"    Mean per-dim state std across phases: {mean_state_std:.4f}")
    print(f"    Max per-phase state std: {phase_std.max():.4f}")
    print(f"    Phase with max variability: {mean_phase_std.argmax()} "
          f"(std={mean_phase_std.max():.4f})")
    print(f"    Phase with min variability: {mean_phase_std.argmin()} "
          f"(std={mean_phase_std.min():.4f})")

    # Touchdown state variability
    td_states = np.array(all_touchdown_states)
    td_std = td_states.std(axis=0)
    print(f"\n  Touchdown state std: mean={td_std.mean():.4f}, "
          f"max={td_std.max():.4f}")
    # Per-dim
    dim_names = ["z", "sin(a)", "cos(a)", "thigh", "leg", "foot",
                 "vx", "vz", "v_ang", "v_thigh", "v_leg", "v_foot"]
    print(f"    Per-dim touchdown std:")
    for i, (name, s) in enumerate(zip(dim_names[:td_std.shape[0]], td_std)):
        print(f"      {name:12s}: {s:.4f}")

    # Foot height trajectory (phase-aligned)
    foot_mean = all_aligned_foot_z.mean(axis=0)
    foot_std = all_aligned_foot_z.std(axis=0)
    torso_mean = all_aligned_torso_z.mean(axis=0)
    torso_std = all_aligned_torso_z.std(axis=0)

    print(f"\n  Foot z trajectory (phase-aligned):")
    print(f"    Mean range: [{foot_mean.min():.3f}, {foot_mean.max():.3f}]")
    print(f"    Mean std across cycles: {foot_std.mean():.4f}")
    print(f"  Torso z trajectory (phase-aligned):")
    print(f"    Mean range: [{torso_mean.min():.3f}, {torso_mean.max():.3f}]")
    print(f"    Mean std across cycles: {torso_std.mean():.4f}")

    # ── Verdict ─────────────────────────────────────────────────────────
    cv = cycle_lengths.std() / cycle_lengths.mean()
    print(f"\n  {'='*60}")

    checks = []
    checks.append(("CV < 0.3 (cycle length consistent)", cv < 0.3, f"CV={cv:.3f}"))
    checks.append(("Mean state std < 0.3", mean_state_std < 0.3,
                   f"std={mean_state_std:.4f}"))
    checks.append(("Touchdown state std < 0.3", td_std.mean() < 0.3,
                   f"std={td_std.mean():.4f}"))
    checks.append(("> 3 cycles/ep avg", n_total_cycles / args.n_episodes > 3,
                   f"{n_total_cycles/args.n_episodes:.1f} cycles/ep"))
    checks.append(("N cycles >= 20", n_total_cycles >= 20,
                   f"n={n_total_cycles}"))

    all_ok = True
    for label, ok, detail in checks:
        status = "[OK]" if ok else "[FAIL]"
        print(f"  {status} {label}: {detail}")
        if not ok:
            all_ok = False

    if all_ok:
        print(f"\n  => PASS: Gait cycles are sufficiently repeatable for ILC.")
    else:
        print(f"\n  => PARTIAL: Some repeatability concerns. ILC may need")
        print(f"     phase-aware averaging or larger learning gain damping.")

    # Save
    summary = {
        "n_episodes": args.n_episodes,
        "n_total_cycles": int(n_total_cycles),
        "cycle_length": {
            "mean": float(cycle_lengths.mean()),
            "std": float(cycle_lengths.std()),
            "min": int(cycle_lengths.min()),
            "max": int(cycle_lengths.max()),
            "cv": float(cv),
        },
        "phase_aligned": {
            "n_phase_points": N_PHASE,
            "mean_state_std": float(mean_state_std),
            "max_phase_std": float(phase_std.max()),
            "foot_z_range": [float(foot_mean.min()), float(foot_mean.max())],
            "foot_z_mean_std": float(foot_std.mean()),
            "torso_z_range": [float(torso_mean.min()), float(torso_mean.max())],
            "torso_z_mean_std": float(torso_std.mean()),
        },
        "touchdown_state_std_mean": float(td_std.mean()),
        "touchdown_state_std_per_dim": {dim_names[i]: float(td_std[i])
                                        for i in range(len(td_std))},
        "all_checks_pass": all_ok,
        "episode_summaries": episode_summaries,
    }
    Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(args.json_out, "w"), indent=2)
    print(f"\n  Saved to {args.json_out}")


if __name__ == "__main__":
    main()
