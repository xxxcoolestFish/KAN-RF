"""Test planner teacher upper bound on friction_070.

Uses CEM optimization with KAN world model to plan reward-aware actions
in the imagined target world. NO target reward feedback — only source
reward function evaluated on KAN-simulated states.

Answers: can a reward-aware KAN planner beat transport_action (671)?
"""

from __future__ import annotations

import argparse, sys, os
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cpbn.cognitive_pca import CognitivePCA, PCARanges
from cpbn.generic_affine_kan import AffineKANContext
from scripts.diagnose_hopper_pullback_effect import (
    fit_distilled_source_counterfactual_context,
    load_source_twin,
)
from scripts.prescreen_hopper_physics_shifts import ENVS, SHIFTS, make_shifted_env
from scripts.validate_hopper_joint_online_adaptation import (
    FrozenSourcePolicy,
    load_cognition,
)


def hopper_reward(state, action, prev_state=None):
    """Compute Hopper-v5 reward from state and action.

    Components: forward_velocity + healthy_reward - ctrl_cost
    """
    # Forward velocity = delta_x (state[0] is x-position)
    if prev_state is not None:
        fwd_vel = state[0] - prev_state[0]
    else:
        fwd_vel = state[5]  # velocity_x is state[5] in Hopper

    # Healthy: z-height (state[1]) > 0.7 and angles (state[2:5]) in range
    height_ok = 1.0 if state[1] > 0.7 else 0.0
    angle_ok = 1.0 if np.all(np.abs(state[2:5]) < 1.0) else 0.0
    healthy = height_ok * angle_ok

    # Control cost
    ctrl_cost = 0.001 * np.sum(action ** 2)

    return fwd_vel + 1.0 * healthy - ctrl_cost


def cem_planner(
    state, z_value, source_policy, source_context, basis, pca, device,
    horizon=8, n_iter=3, n_candidates=128, n_elite=16,
    init_std=0.3, final_std=0.05,
):
    """CEM planner using KAN world model.

    Optimizes action sequence to maximize cumulative reward in the
    KAN-imagined target physics world.
    """
    s_dim = basis.state_dim
    a_dim = basis.action_dim
    n_feature = basis.feature_dim

    # Build drifted KAN context for this z
    delta_W = pca.decode(z_value)
    dw_t = torch.tensor(delta_W, device=device, dtype=torch.float32)
    source_blocks = source_context.coefficients.clone().reshape(
        1 + a_dim, n_feature, -1,
    )
    db = source_blocks.clone()
    db[0] = source_blocks[0] + dw_t.reshape(n_feature, -1)
    drifted_ctx = AffineKANContext(
        db.reshape_as(source_context.coefficients)
    )

    # Source action as initialization
    s0 = torch.tensor(state, device=device, dtype=torch.float32).unsqueeze(0)
    nominal_seq = source_policy.action(s0).squeeze(0).cpu().numpy()  # (a_dim,)
    mean_seq = np.tile(nominal_seq, (horizon, 1))  # (H, a_dim)
    std = init_std

    best_seq = mean_seq.copy()
    best_reward = -float('inf')

    for it in range(n_iter):
        # Sample candidates
        candidates = np.random.randn(n_candidates, horizon, a_dim) * std
        candidates += mean_seq[None, :, :]
        candidates = np.clip(candidates, -1.0, 1.0)

        rewards = np.zeros(n_candidates)
        for c in range(n_candidates):
            s_t = s0.clone()
            total_r = 0.0
            alive = True
            for t in range(horizon):
                a_t = torch.tensor(candidates[c, t:t+1], device=device, dtype=torch.float32)
                effect = drifted_ctx.acceleration(basis, s_t, a_t)
                s_next = s_t + effect

                s_np = s_next.squeeze(0).cpu().numpy()
                a_np = candidates[c, t]
                s_prev_np = s_t.squeeze(0).cpu().numpy()
                r_t = hopper_reward(s_np, a_np, s_prev_np)
                total_r += r_t

                # Early termination
                if s_np[1] < 0.3 or np.abs(s_np[2:5]).max() > 2.0:
                    total_r -= 5.0
                    alive = False

                s_t = s_next
                if not alive:
                    break
            rewards[c] = total_r

        # Select elite
        elite_idx = np.argsort(rewards)[-n_elite:]
        elite = candidates[elite_idx]

        # Update best
        if rewards[elite_idx[-1]] > best_reward:
            best_reward = rewards[elite_idx[-1]]
            best_seq = elite[-1].copy()

        # Refit distribution
        mean_seq = elite.mean(axis=0)
        std = max(final_std, std * 0.5)

    return torch.tensor(best_seq[0:1], device=device, dtype=torch.float32)


def rollout_teacher(source_policy, basis, source_context, pca, z_val, shift,
                    device, n_episodes=10, planner_mode="transport"):
    """Run teacher controller in closed-loop on target environment."""
    fit_args = argparse.Namespace(
        target="friction_070", seed=1811, env="hopper", device="cuda",
        cognition_warmup=1024, warmup_noise=0.3,
        transform_ridge=10.0, drift_ridge=100.0,
        drift_spectral_eta=0.0, drift_spectral_beta=1.0,
        drift_spectral_mode="max", drift_smooth_lambda=0.0, diagonal_transform=False,
    )
    source_twin = load_source_twin(
        "results/hopper_source_affine_twin_cloud_seed1811.pt", device,
    )
    target_ctx, _ = fit_distilled_source_counterfactual_context(
        source_policy, basis, source_context, fit_args, device, source_twin,
    )

    returns = []
    for ep in range(n_episodes):
        env = make_shifted_env(shift, 1911 + ep * 100, "hopper")()
        obs, _ = env.reset(seed=1911 + ep * 100)
        total = 0.0
        while True:
            s_t = torch.tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
            nominal = source_policy.action(s_t)
            source_effect = source_context.acceleration(basis, s_t, nominal)

            if planner_mode == "cem":
                a = cem_planner(
                    obs, z_val, source_policy, source_context, basis, pca, device,
                ).squeeze(0).cpu().numpy()
            elif planner_mode == "transport":
                a = target_ctx.transport_action(
                    basis, s_t, desired_effect=source_effect,
                    nominal_action=nominal, regularization=1e-2,
                ).clamp(-1, 1).squeeze(0).cpu().numpy()
            else:  # source
                a = nominal.squeeze(0).cpu().numpy()

            obs, r, t, tr, _ = env.step(a)
            total += float(r)
            if t or tr: break
        env.close()
        returns.append(total)
    return float(np.mean(returns)), float(np.std(returns))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=1811)
    parser.add_argument("--n-episodes", type=int, default=10)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    source_policy = FrozenSourcePolicy(
        "results/hopper_source_sb3_ppo_continued_seed1811.zip",
        "results/hopper_source_sb3_vecnorm_continued_seed1811.pkl",
        device, args.seed, env="hopper",
    )
    basis, source_context, _, _ = load_cognition(
        argparse.Namespace(
            cognition_checkpoint="results/hopper_source_control_sobolev_calibrated_seed1811.pt",
            device=args.device,
        ),
        device,
    )
    pca = CognitivePCA(
        "results/cppe_pca_model.npz", k=5,
        pc_ranges=PCARanges(
            mins=np.load("results/cppe_pca_model.npz")["pc_mins"],
            maxs=np.load("results/cppe_pca_model.npz")["pc_maxs"],
        ),
    )
    z_friction = np.load("results/cppe_pca_model.npz")["z_values"][2, :5].astype(np.float32)

    shift = SHIFTS["friction_070"]

    for label, mode in [
        ("Source policy (no adapt)", "source"),
        ("Transport teacher", "transport"),
        ("CEM planner teacher", "cem"),
    ]:
        r, s = rollout_teacher(
            source_policy, basis, source_context, pca, z_friction,
            shift, device, n_episodes=args.n_episodes, planner_mode=mode,
        )
        print(f"{label:30s}: {r:8.1f} +/- {s:.1f}")


if __name__ == "__main__":
    main()
