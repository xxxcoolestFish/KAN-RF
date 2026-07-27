"""Phase 1.5: Phase-gated sparse gradient correction.

Tests whether applying gradient correction ONLY at key gait phases
(once per cycle) avoids the distribution shift that kills every-step correction.

Hypothesis: single-step gradient is locally correct (Phase 0A), but every-step
application shifts state distribution beyond source critic's reliable region.
Sparse per-cycle intervention may preserve distribution while retaining benefit.

Conditions:
  1. Transport only (baseline)
  2. Every-step correction (Phase 1 dead end)
  3. Correction at MID_STANCE only (foot planted, torso lowest)
  4. Correction at TOUCHDOWN only (foot just hit ground)
  5. Correction at MID_STANCE + TOUCHDOWN (twice per cycle)
  6. Shuffle: random step in cycle (ablation)
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
# Phase detector for Hopper
# ═══════════════════════════════════════════════════════════════════════

class HopperPhaseDetector:
    """Detect gait phase events from MuJoCo foot position.

    Uses foot z-position to detect:
      - TOUCHDOWN: foot just made contact (z crosses below threshold)
      - MID_STANCE: foot planted, torso at local z-minimum
      - LIFTOFF: foot just left ground (z crosses above threshold)
      - MID_FLIGHT: torso at local z-maximum
    """

    def __init__(self, foot_z_threshold=0.12):
        self.threshold = foot_z_threshold
        self._prev_in_contact = False
        self._foot_z_history = []
        self._torso_z_history = []
        self._max_history = 30

    def reset(self):
        self._prev_in_contact = False
        self._foot_z_history = []
        self._torso_z_history = []

    def update(self, env):
        """Get current phase events. Call AFTER env.step().

        Returns:
            events: set of {"touchdown", "liftoff", "mid_stance", "mid_flight"}
        """
        try:
            foot_z = float(env.unwrapped.data.body("foot").xpos[2])
        except Exception:
            foot_z = 0.0

        try:
            torso_z = float(env.unwrapped.data.body("torso").xpos[2])
        except Exception:
            torso_z = 0.0

        self._foot_z_history.append(foot_z)
        self._torso_z_history.append(torso_z)
        if len(self._foot_z_history) > self._max_history:
            self._foot_z_history.pop(0)
            self._torso_z_history.pop(0)

        in_contact = foot_z < self.threshold
        events = set()

        # Edge detection for touchdown / liftoff
        if in_contact and not self._prev_in_contact:
            events.add("touchdown")
        if not in_contact and self._prev_in_contact:
            events.add("liftoff")

        self._prev_in_contact = in_contact

        # Local extrema detection for mid-stance / mid-flight
        if len(self._torso_z_history) >= 5:
            recent = self._torso_z_history[-5:]
            mid_idx = len(recent) // 2
            if (recent[mid_idx] < recent[0] and recent[mid_idx] < recent[-1]
                    and in_contact):
                events.add("mid_stance")
            if (recent[mid_idx] > recent[0] and recent[mid_idx] > recent[-1]
                    and not in_contact):
                events.add("mid_flight")

        return events


# ═══════════════════════════════════════════════════════════════════════
# Gradient corrector (same as Phase 1)
# ═══════════════════════════════════════════════════════════════════════

class ValueGradientCorrector:
    def __init__(self, source_policy, basis, target_ctx, device):
        self.source_policy = source_policy
        self.basis = basis
        self.target_ctx = target_ctx
        self.device = device

    def compute_correction(self, s_tensor, a_transport_tensor):
        with torch.no_grad():
            s_effect = self.target_ctx.acceleration(self.basis, s_tensor, a_transport_tensor)
            s_next = s_tensor + s_effect

        mean = self.source_policy.mean
        var = self.source_policy.variance
        s_next_grad = s_next.detach().clone().requires_grad_(True)
        s_norm = ((s_next_grad - mean) / (var + 1e-8).sqrt()).clamp(-10, 10)
        features = self.source_policy.model.policy.features_extractor(s_norm)
        latent = self.source_policy.model.policy.mlp_extractor(features)
        if isinstance(latent, tuple):
            _, latent_vf = latent
        else:
            latent_vf = latent
        value = self.source_policy.model.policy.value_net(latent_vf).squeeze(-1)
        grad_v = torch.autograd.grad(value.sum(), s_next_grad)[0].detach()

        with torch.no_grad():
            _, gain = self.target_ctx.drift_and_gain(self.basis, s_tensor)
            B = gain.squeeze(0)
        g = B.T @ grad_v.squeeze(0)
        g_norm = g.norm()
        d = g / g_norm if g_norm > 1e-8 else torch.zeros_like(g)

        return d.cpu().numpy(), {"grad_norm": float(g_norm), "v_s_next": float(value)}


# ═══════════════════════════════════════════════════════════════════════
# Episode runners
# ═══════════════════════════════════════════════════════════════════════

def run_episode_phase_gated(env, source_policy, basis, source_context, target_ctx,
                            corrector, alpha, eps_a, device, gate_events):
    """Run episode with gradient correction ONLY at specified phase events.

    Phase flow:
      1. Detect phase BEFORE computing action
      2. Apply correction if phase matches gate
      3. Step environment
      4. Update phase detector from new state
    """
    obs, _ = env.reset()
    total_r = 0.0
    detector = HopperPhaseDetector()
    detector.reset()
    # Initialize phase by running one detector update on reset state
    detector.update(env)
    corrections_applied = 0
    total_steps = 0
    # Don't correct on the very first step (no phase context yet)
    apply_correction_next = False

    while True:
        s_t = torch.as_tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
        nominal = source_policy.action(s_t)
        s_eff = source_context.acceleration(basis, s_t, nominal)
        a_tr_tensor = target_ctx.transport_action(
            basis, s_t, desired_effect=s_eff,
            nominal_action=nominal, regularization=1e-2,
        ).clamp(-1, 1)
        a_tr = a_tr_tensor.squeeze(0).cpu().numpy()

        # Apply correction if previous step's phase event matched the gate
        if alpha > 0 and apply_correction_next:
            d, c_info = corrector.compute_correction(s_t, a_tr_tensor)
            da = alpha * d
            da = np.clip(da, -eps_a, eps_a)
            a_final = np.clip(a_tr + da, -1, 1)
            corrections_applied += 1
        else:
            a_final = a_tr

        next_obs, reward, terminated, truncated, _ = env.step(a_final)
        total_r += float(reward)
        total_steps += 1

        # Update phase and check if NEXT step should be corrected
        events = detector.update(env)
        if gate_events == "every":
            apply_correction_next = True
        elif gate_events is not None and len(gate_events) > 0:
            apply_correction_next = bool(events & gate_events)
        else:
            apply_correction_next = False

        obs = next_obs
        if terminated or truncated:
            break

    return total_r, total_steps, corrections_applied


def run_episode_transport(env, source_policy, basis, source_context, target_ctx, device):
    """Pure Transport baseline."""
    obs, _ = env.reset()
    total_r = 0.0
    steps = 0
    while True:
        s_t = torch.as_tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
        nominal = source_policy.action(s_t)
        s_eff = source_context.acceleration(basis, s_t, nominal)
        a_tr = target_ctx.transport_action(
            basis, s_t, desired_effect=s_eff,
            nominal_action=nominal, regularization=1e-2,
        ).clamp(-1, 1).squeeze(0).cpu().numpy()
        next_obs, reward, terminated, truncated, _ = env.step(a_tr)
        total_r += float(reward)
        steps += 1
        obs = next_obs
        if terminated or truncated:
            break
    return total_r, steps, 0


def run_episode_every_step(env, source_policy, basis, source_context, target_ctx,
                           corrector, alpha, eps_a, device):
    """Every-step correction (Phase 1 approach)."""
    obs, _ = env.reset()
    total_r = 0.0
    corrections = 0
    steps = 0
    while True:
        s_t = torch.as_tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
        nominal = source_policy.action(s_t)
        s_eff = source_context.acceleration(basis, s_t, nominal)
        a_tr_tensor = target_ctx.transport_action(
            basis, s_t, desired_effect=s_eff,
            nominal_action=nominal, regularization=1e-2,
        ).clamp(-1, 1)
        a_tr = a_tr_tensor.squeeze(0).cpu().numpy()

        if alpha > 0:
            d, c_info = corrector.compute_correction(s_t, a_tr_tensor)
            da = alpha * d
            da = np.clip(da, -eps_a, eps_a)
            a_final = np.clip(a_tr + da, -1, 1)
            corrections += 1
        else:
            a_final = a_tr

        next_obs, reward, terminated, truncated, _ = env.step(a_final)
        total_r += float(reward)
        steps += 1
        obs = next_obs
        if terminated or truncated:
            break
    return total_r, steps, corrections


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=1811)
    parser.add_argument("--n-episodes", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=0.05,
                       help="Gradient step size (same as Phase 1 best)")
    parser.add_argument("--eps-a", type=float, default=0.10,
                       help="Action trust region")
    parser.add_argument("--json-out", default="results/phase1p5_phase_gated.json")
    args = parser.parse_args()

    device = torch.device(args.device)
    shift = SHIFTS["friction_070"]

    print("=" * 72)
    print("Phase 1.5: Phase-Gated Sparse Gradient Correction")
    print(f"  alpha={args.alpha}, eps_a={args.eps_a}")
    print("=" * 72)

    # ── Load ─────────────────────────────────────────────────────────────
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
    corrector = ValueGradientCorrector(source_policy, basis, target_ctx, device)

    # ── Test conditions ──────────────────────────────────────────────────
    conditions = [
        ("Transport (baseline)", "transport", None),
        ("Every-step correction", "every_step", None),
        ("Mid-stance only", "mid_stance", {"mid_stance"}),
        ("Touchdown only", "touchdown", {"touchdown"}),
        ("Liftoff only", "liftoff", {"liftoff"}),
        ("Mid-stance + Touchdown", "stance_touchdown", {"mid_stance", "touchdown"}),
    ]

    print(f"\n[2/2] Testing {len(conditions)} conditions, {args.n_episodes} episodes each...",
          flush=True)

    all_results = {}

    for label, key, gate_events in conditions:
        print(f"\n  --- {label} ---", flush=True)
        returns = []
        lengths = []
        total_corrections = []

        for ep in range(args.n_episodes):
            env = make_shifted_env(shift, args.seed + ep * 100, "hopper")()

            if key == "transport":
                ep_r, ep_len, n_corr = run_episode_transport(
                    env, source_policy, basis, source_context, target_ctx, device)
            elif key == "every_step":
                ep_r, ep_len, n_corr = run_episode_every_step(
                    env, source_policy, basis, source_context, target_ctx,
                    corrector, args.alpha, args.eps_a, device)
            else:
                ep_r, ep_len, n_corr = run_episode_phase_gated(
                    env, source_policy, basis, source_context, target_ctx,
                    corrector, args.alpha, args.eps_a, device, gate_events)

            env.close()
            returns.append(ep_r)
            lengths.append(ep_len)
            total_corrections.append(n_corr)

            if (ep + 1) % 10 == 0:
                print(f"    ep {ep+1}/{args.n_episodes}: "
                      f"mean_return={np.mean(returns):.1f}, "
                      f"corr/ep={np.mean(total_corrections):.1f}", flush=True)

        mean_r = np.mean(returns)
        std_r = np.std(returns)
        mean_len = np.mean(lengths)
        mean_corr = np.mean(total_corrections)
        corr_per_step = mean_corr / max(mean_len, 1)

        print(f"    Result: {mean_r:.1f} ± {std_r:.1f}, "
              f"len={mean_len:.1f}, corr/ep={mean_corr:.1f} "
              f"({corr_per_step:.1%} of steps)", flush=True)

        all_results[label] = {
            "key": key,
            "mean_return": float(mean_r),
            "std_return": float(std_r),
            "returns": [float(r) for r in returns],
            "mean_length": float(mean_len),
            "mean_corrections_per_ep": float(mean_corr),
            "correction_fraction": float(corr_per_step),
        }

    # ── Report ───────────────────────────────────────────────────────────
    transport_r = all_results.get("Transport (baseline)", {}).get("mean_return", 570)

    print(f"\n{'='*72}")
    print(f"  {'Condition':30s} {'Return':>10s} {'vs Tr':>10s} "
          f"{'Corr/ep':>10s} {'Corr%':>8s}")
    print(f"  {'-'*68}")

    for label, res in all_results.items():
        mr = res["mean_return"]
        vs_tr = mr - transport_r
        n_corr = res["mean_corrections_per_ep"]
        corr_pct = res["correction_fraction"] * 100
        print(f"  {label:30s} {mr:>10.1f} {vs_tr:>+10.1f} "
              f"{n_corr:>10.1f} {corr_pct:>7.1f}%")

    best = max(all_results.values(), key=lambda r: r["mean_return"])
    print(f"\n  Best: {best['key']} ({best['mean_return']:.1f})")
    if best["mean_return"] > transport_r + 5:
        print(f"  => Meaningful improvement over Transport!")
    elif best["mean_return"] > transport_r + 2:
        print(f"  => Marginal improvement (possible noise).")
    else:
        print(f"  => No improvement over Transport.")

    if best["mean_return"] > 672:
        print(f"  => BEATS source policy (672)!")

    # Save
    summary = {
        "config": {"alpha": args.alpha, "eps_a": args.eps_a,
                   "n_episodes": args.n_episodes},
        "transport_baseline": float(transport_r),
        "source_policy_baseline": 672,
        "results": all_results,
        "best_condition": best["key"],
        "best_return": float(best["mean_return"]),
    }
    Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(args.json_out, "w"), indent=2)
    print(f"\n  Saved to {args.json_out}")


if __name__ == "__main__":
    main()
