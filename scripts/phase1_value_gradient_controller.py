"""Phase 1: One-step value-gradient controller.

Closed-loop test of:
    d_t = normalize(B_KAN^T * grad(V_source)(s'_KAN))
    a_t = a_transport + alpha * d_t

Compared against:
    - Transport only (alpha=0, current best reward-free: 572)
    - Source policy (no adaptation: 672, the threshold to beat)

Key constraints:
    - NO KAN multi-step rollout
    - NO multi-step return backprop
    - NO imagined terminal value
    - ONLY single-step B_t + source critic gradient
"""

from __future__ import annotations

import argparse, sys, os, json
from pathlib import Path
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cpbn.generic_affine_kan import RecursiveAffineKANEstimator
from scripts.diagnose_hopper_pullback_effect import (
    fit_distilled_source_counterfactual_context,
    load_source_twin,
)
from scripts.prescreen_hopper_physics_shifts import SHIFTS, make_shifted_env
from scripts.validate_hopper_joint_online_adaptation import (
    FrozenSourcePolicy,
    load_cognition,
)

# Hopper-v5 exact params
FORWARD_WEIGHT = 1.0
HEALTHY_REWARD = 1.0
CTRL_COST_WEIGHT = 0.001
Z_LO, Z_HI = 0.7, float('inf')
ANG_LO, ANG_HI = -0.2, 0.2
DT = 0.008 * 4
GAMMA = 0.99


# ═══════════════════════════════════════════════════════════════════════
# Reward
# ═══════════════════════════════════════════════════════════════════════

def hopper_reward_np(obs, action, prev_obs=None):
    fwd = obs[5] if prev_obs is None else (obs[0] - prev_obs[0]) / DT
    z_ok = Z_LO < obs[1] < Z_HI
    a_ok = ANG_LO < obs[2] < ANG_HI
    healthy = 1.0 if (z_ok and a_ok) else 0.0
    ctrl = CTRL_COST_WEIGHT * float(np.sum(action ** 2))
    return FORWARD_WEIGHT * fwd + HEALTHY_REWARD * healthy - ctrl


# ═══════════════════════════════════════════════════════════════════════
# Gradient correction
# ═══════════════════════════════════════════════════════════════════════

class ValueGradientCorrector:
    """Computes normalized one-step value-gradient action correction.

    d = normalize(B^T * grad(V_source)(s'_KAN))

    Uses source PPO critic for value gradient.
    Uses KAN for B_t = dF/da.
    """

    def __init__(self, source_policy, basis, target_ctx, device):
        self.source_policy = source_policy
        self.basis = basis
        self.target_ctx = target_ctx
        self.device = device

        # Cache for diagnostics
        self.last_gradient_norm = 0.0
        self.last_v_value = 0.0
        self.last_correction_norm = 0.0

    def compute_correction(self, s_tensor, a_transport_tensor):
        """Compute normalized gradient correction direction.

        Args:
            s_tensor: (1, s_dim) current state on device
            a_transport_tensor: (1, a_dim) transport action on device

        Returns:
            d: (a_dim,) numpy array, unit-norm correction direction
            info: dict with diagnostic values
        """
        # KAN-predicted next state under transport action
        with torch.no_grad():
            s_effect = self.target_ctx.acceleration(self.basis, s_tensor, a_transport_tensor)
            s_next = s_tensor + s_effect

        # Value gradient at KAN-predicted next state (needs autograd)
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

        grad_v = torch.autograd.grad(
            value.sum(), s_next_grad, create_graph=False, retain_graph=False,
        )[0].detach()

        # B_t = G(s) from KAN (no gradient needed for this)
        with torch.no_grad():
            _, gain = self.target_ctx.drift_and_gain(self.basis, s_tensor)
            B = gain.squeeze(0)  # (s_dim, a_dim)

        # Action gradient: B^T * grad(V)
        g = (B.T @ grad_v.squeeze(0))  # (a_dim,)

        # Normalize to unit direction
        g_norm = g.norm()
        if g_norm > 1e-8:
            d = g / g_norm
        else:
            d = torch.zeros_like(g)

        # Cache diagnostics
        self.last_gradient_norm = float(g_norm)
        self.last_v_value = float(value)
        self.last_correction_norm = float(d.norm())

        return d.cpu().numpy(), {
            "gradient_norm": float(g_norm),
            "v_s_next": float(value),
            "correction_norm": float(d.norm()),
        }


# ═══════════════════════════════════════════════════════════════════════
# Episode runner
# ═══════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=1811)
    parser.add_argument("--n-episodes", type=int, default=20,
                       help="Episodes per configuration")
    parser.add_argument("--alphas", type=str, default="0,0.02,0.05,0.10,0.15",
                       help="Comma-separated alpha values to test (0=Transport baseline)")
    parser.add_argument("--eps-a", type=float, default=0.15,
                       help="Action trust region (max correction magnitude)")
    parser.add_argument("--online-kan", action="store_true", default=True,
                       help="Update KAN online during episodes")
    parser.add_argument("--kan-warmup", type=int, default=1024,
                       help="Warmup steps before online KAN update")
    parser.add_argument("--json-out", default="results/phase1_value_gradient.json")
    args = parser.parse_args()

    device = torch.device(args.device)
    shift = SHIFTS["friction_070"]
    alphas = [float(x) for x in args.alphas.split(",")]

    print("=" * 72)
    print("Phase 1: One-Step Value-Gradient Controller")
    print(f"  Alphas: {alphas}")
    print(f"  eps_a: {args.eps_a}, online_kan: {args.online_kan}")
    print("=" * 72)

    # ── Load components ─────────────────────────────────────────────────
    print("\n[1/3] Loading components...", flush=True)

    source_policy = FrozenSourcePolicy(
        "results/hopper_source_sb3_ppo_continued_seed1811.zip",
        "results/hopper_source_sb3_vecnorm_continued_seed1811.pkl",
        device, args.seed, env="hopper",
    )
    source_twin = load_source_twin(
        "results/hopper_source_affine_twin_cloud_seed1811.pt", device,
    )
    basis, source_context, estimator, _ = load_cognition(
        argparse.Namespace(
            cognition_checkpoint="results/hopper_source_control_sobolev_calibrated_seed1811.pt",
            device=args.device,
        ),
        device,
    )

    # Initial target KAN (offline fit to friction)
    print("  Fitting initial target KAN...", flush=True)
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

    # Gradient corrector
    corrector = ValueGradientCorrector(source_policy, basis, target_ctx, device)

    # ── Run experiments ─────────────────────────────────────────────────
    print(f"\n[2/3] Running {args.n_episodes} episodes per alpha...", flush=True)

    all_results = {}

    for alpha in alphas:
        label = f"alpha={alpha:.3f}" if alpha > 0 else "Transport (alpha=0)"
        print(f"\n  --- {label} ---", flush=True)

        episode_returns = []
        episode_lengths = []
        all_diag = []

        for ep in range(args.n_episodes):
            env = make_shifted_env(shift, args.seed + ep * 100, "hopper")()

            ep_r, ep_len, diag = run_episode(
                env, source_policy, basis, source_context, target_ctx,
                corrector, alpha, args.eps_a, device,
            )

            env.close()
            episode_returns.append(ep_r)
            episode_lengths.append(ep_len)
            all_diag.append(diag)

            if (ep + 1) % 5 == 0:
                mean_r = np.mean(episode_returns)
                print(f"    ep {ep+1}/{args.n_episodes}: "
                      f"mean_return={mean_r:.1f}, len={ep_len}", flush=True)

        mean_r = np.mean(episode_returns)
        std_r = np.std(episode_returns)
        mean_len = np.mean(episode_lengths)

        # Aggregate correction statistics
        all_corr_mags = []
        all_grad_norms = []
        for d in all_diag:
            all_corr_mags.extend(d.get("correction_magnitudes", []))
            all_grad_norms.extend(d.get("gradient_norms", []))

        avg_corr = np.mean(all_corr_mags) if all_corr_mags else 0
        avg_grad = np.mean(all_grad_norms) if all_grad_norms else 0

        print(f"    Result: {mean_r:.1f} ± {std_r:.1f}, "
              f"avg_len={mean_len:.1f}, "
              f"avg_corr={avg_corr:.4f}, avg_grad={avg_grad:.4f}", flush=True)

        all_results[label] = {
            "alpha": alpha,
            "mean_return": float(mean_r),
            "std_return": float(std_r),
            "returns": [float(r) for r in episode_returns],
            "mean_length": float(mean_len),
            "avg_correction_magnitude": float(avg_corr),
            "avg_gradient_norm": float(avg_grad),
        }

    # ── Report ───────────────────────────────────────────────────────────
    print(f"\n[3/3] Summary", flush=True)
    print("=" * 72)
    print(f"\n  {'Alpha':>10s}  {'Mean Return':>12s}  {'Std':>8s}  "
          f"{'vs Transport':>14s}  {'vs Source(672)':>16s}")
    print(f"  {'-'*70}")

    transport_r = all_results.get("Transport (alpha=0)", {}).get("mean_return", 572)

    for label, res in all_results.items():
        alpha = res["alpha"]
        mr = res["mean_return"]
        std = res["std_return"]
        vs_tr = mr - transport_r
        vs_src = mr - 672
        print(f"  {alpha:>10.3f}  {mr:>12.1f}  {std:>8.1f}  "
              f"{vs_tr:>+14.1f}  {vs_src:>+16.1f}")

    # Highlight best
    best_alpha = max(all_results.values(), key=lambda r: r["mean_return"])
    best_r = best_alpha["mean_return"]
    print(f"\n  Best: alpha={best_alpha['alpha']:.3f}, return={best_r:.1f}")
    if best_r > transport_r + 20:
        print(f"  => Improvement over Transport: {best_r - transport_r:+.1f}")
    if best_r > 672:
        print(f"  => BEATS source policy (672)! First reward-free method to do so.")
    else:
        print(f"  => Still below source policy ({672 - best_r:.0f} gap).")

    # ── Save ─────────────────────────────────────────────────────────────
    summary = {
        "config": {"eps_a": args.eps_a, "alphas": alphas,
                   "online_kan": args.online_kan, "n_episodes": args.n_episodes},
        "transport_baseline": float(transport_r),
        "source_policy_baseline": 672,
        "results": all_results,
        "best_alpha": best_alpha["alpha"],
        "best_return": float(best_r),
    }
    Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(args.json_out, "w"), indent=2)
    print(f"\n  Results saved to {args.json_out}")


def run_episode(env, source_policy, basis, source_context, target_ctx,
                corrector, alpha, eps_a, device):
    """Run one episode with value-gradient-corrected actions.

    Uses fixed target_ctx (offline-fitted target KAN).
    No online KAN updates during the episode.
    """
    obs, _ = env.reset()
    total_r = 0.0
    prev_obs = None
    step_count = 0
    diag = {
        "correction_magnitudes": [],
        "gradient_norms": [],
        "v_values": [],
    }

    while True:
        s_t = torch.as_tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)

        # Transport
        nominal = source_policy.action(s_t)
        s_eff = source_context.acceleration(basis, s_t, nominal)
        a_tr_tensor = target_ctx.transport_action(
            basis, s_t, desired_effect=s_eff,
            nominal_action=nominal, regularization=1e-2,
        ).clamp(-1, 1)
        a_tr = a_tr_tensor.squeeze(0).cpu().numpy()

        # Gradient correction
        if alpha > 0:
            d, c_info = corrector.compute_correction(s_t, a_tr_tensor)
            da = alpha * d
            da = np.clip(da, -eps_a, eps_a)
            a_final = np.clip(a_tr + da, -1, 1)
            diag["correction_magnitudes"].append(float(np.linalg.norm(da)))
            diag["gradient_norms"].append(float(c_info["gradient_norm"]))
            diag["v_values"].append(float(c_info["v_s_next"]))
        else:
            a_final = a_tr

        next_obs, reward, terminated, truncated, _ = env.step(a_final)
        total_r += float(reward)
        step_count += 1

        prev_obs = obs.copy()
        obs = next_obs

        if terminated or truncated:
            break

    return total_r, step_count, diag


if __name__ == "__main__":
    main()
