"""Phase 1: True-dynamics MPC with EXACT Hopper-v5 reward.

Answers: can constrained MPC with correct task reward beat Transport?
"""

from __future__ import annotations

import argparse, sys, os, json
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.prescreen_hopper_physics_shifts import SHIFTS, make_shifted_env
from scripts.validate_hopper_joint_online_adaptation import FrozenSourcePolicy


# Hopper-v5 exact params (verified from gymnasium)
FORWARD_WEIGHT = 1.0
HEALTHY_REWARD = 1.0
CTRL_COST_WEIGHT = 0.001
Z_RANGE = (0.7, float('inf'))
ANGLE_RANGE = (-0.2, 0.2)
DT = 0.008 * 4  # model.opt.timestep * frame_skip


def is_healthy(obs):
    """Exact Hopper-v5 healthy check."""
    z = obs[1]
    angle = obs[2]  # torso angle
    return (Z_RANGE[0] < z < Z_RANGE[1]) and (ANGLE_RANGE[0] < angle < ANGLE_RANGE[1])


def is_healthy_torch(z, angle):
    """Torch version for differentiable MPC."""
    return (z > Z_RANGE[0]) & (z < Z_RANGE[1]) & (angle > ANGLE_RANGE[0]) & (angle < ANGLE_RANGE[1])


def hopper_reward(obs, action, prev_obs=None):
    """Exact Hopper-v5 reward: forward_vel + healthy - ctrl_cost."""
    if prev_obs is not None:
        fwd_vel = (obs[0] - prev_obs[0]) / DT
    else:
        fwd_vel = obs[5]  # x_velocity from observation

    healthy = 1.0 if is_healthy(obs) else 0.0
    ctrl = CTRL_COST_WEIGHT * np.sum(action ** 2)
    return FORWARD_WEIGHT * fwd_vel + HEALTHY_REWARD * healthy - ctrl


def hopper_reward_smooth(obs, action, prev_obs_t, sigma_z=0.02, sigma_a=0.05):
    """Smooth, differentiable Hopper reward surrogate for optimization.

    Uses sigmoid for healthy (continuous approximation).
    """
    fwd_vel = (obs[..., 0] - prev_obs_t[..., 0]) / DT if prev_obs_t is not None else obs[..., 5]

    z, angle = obs[..., 1], obs[..., 2]
    healthy_z = torch.sigmoid((z - Z_RANGE[0]) / sigma_z) * torch.sigmoid((Z_RANGE[1] - z) / sigma_z)
    healthy_a = torch.sigmoid((angle - ANGLE_RANGE[0]) / sigma_a) * torch.sigmoid((ANGLE_RANGE[1] - angle) / sigma_a)
    healthy = healthy_z * healthy_a

    ctrl = CTRL_COST_WEIGHT * (action ** 2).sum(dim=-1)
    return FORWARD_WEIGHT * fwd_vel + HEALTHY_REWARD * healthy - ctrl


def true_mpc_hopper(fd_env, qp, qv, a_transport, H=4, n_cand=256, beta=0.01, eps_a=0.15):
    """True-dynamics MPC: optimize exact Hopper reward with action trust region."""
    a_dim = len(a_transport)

    best_reward = -float('inf')
    best_first_a = a_transport.copy()

    for _ in range(n_cand):
        da = np.random.randn(H, a_dim) * eps_a
        da = np.clip(da, -eps_a * 2, eps_a * 2)  # trust region

        total_r = 0.0
        alive = True

        fd_env.unwrapped.set_state(qp, qv)
        prev_obs = None
        for t in range(H):
            a_t = np.clip(a_transport + da[t], -1, 1)
            obs, _, terminated, truncated, _ = fd_env.step(a_t)
            r_t = hopper_reward(obs, a_t, prev_obs)
            total_r += r_t
            prev_obs = obs.copy()
            if terminated or truncated:
                alive = False
                break

        # Action cost penalty
        total_r -= beta * float(np.sum(da ** 2))

        if total_r > best_reward and alive:
            best_reward = total_r
            best_first_a = np.clip(a_transport + da[0], -1, 1)

    return best_first_a, best_reward


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=1811)
    parser.add_argument("--n-episodes", type=int, default=5)
    parser.add_argument("--H", type=int, default=4)
    parser.add_argument("--n-cand", type=int, default=256)
    parser.add_argument("--eps-a", type=float, default=0.15)
    args = parser.parse_args()

    device = torch.device(args.device)
    shift = SHIFTS["friction_070"]

    print(f"Phase 1: Exact Hopper Reward MPC (H={args.H}, cand={args.n_cand}, eps_a={args.eps_a})")

    # Load source policy for Transport nominal
    source_policy = FrozenSourcePolicy(
        "results/hopper_source_sb3_ppo_continued_seed1811.zip",
        "results/hopper_source_sb3_vecnorm_continued_seed1811.pkl",
        device, args.seed, env="hopper",
    )

    # Load KAN for transport
    import argparse as argparse_mod
    from scripts.diagnose_hopper_pullback_effect import fit_distilled_source_counterfactual_context, load_source_twin
    from scripts.validate_hopper_joint_online_adaptation import load_cognition

    print("Loading KAN for Transport...", flush=True)
    source_twin = load_source_twin("results/hopper_source_affine_twin_cloud_seed1811.pt", device)
    basis, source_context, _, _ = load_cognition(
        argparse_mod.Namespace(cognition_checkpoint="results/hopper_source_control_sobolev_calibrated_seed1811.pt", device="cuda"),
        device,
    )
    fit_args = argparse_mod.Namespace(
        target="friction_070", seed=args.seed, env="hopper", device="cuda",
        cognition_warmup=1024, warmup_noise=0.3, transform_ridge=10.0, drift_ridge=100.0,
        drift_spectral_eta=0.0, drift_spectral_beta=1.0, drift_spectral_mode="max",
        drift_smooth_lambda=0.0, diagonal_transform=False,
    )
    target_ctx, _ = fit_distilled_source_counterfactual_context(
        source_policy, basis, source_context, fit_args, device, source_twin,
    )

    # Collect transport trajectory records
    fd_env = make_shifted_env(shift, args.seed, "hopper")()
    obs, _ = fd_env.reset(seed=args.seed)
    records = []
    for _ in range(500):
        s_t = torch.as_tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
        nominal = source_policy.action(s_t)
        s_eff = source_context.acceleration(basis, s_t, nominal)
        a_tr = target_ctx.transport_action(basis, s_t, desired_effect=s_eff, nominal_action=nominal, regularization=1e-2).clamp(-1,1).squeeze(0).cpu().numpy()
        qp = fd_env.unwrapped.data.qpos.copy()
        qv = fd_env.unwrapped.data.qvel.copy()
        records.append({"qp": qp, "qv": qv, "obs": obs.copy(), "a_tr": a_tr})
        obs, _, t, tr, _ = fd_env.step(a_tr)
        if t or tr: break
    print(f"  Transport trajectory: {len(records)} states")

    # Compare Transport vs True-MPC per step
    n_test = min(30, len(records))
    idx_test = np.linspace(0, len(records) - 1, n_test, dtype=int)

    tr_rewards = []; mpc_rewards = []
    for i in idx_test:
        rec = records[i]

        # Transport
        fd_env.unwrapped.set_state(rec["qp"], rec["qv"])
        tr_total = 0.0; prev = None
        for _ in range(args.H):
            obs, r, t, tr, _ = fd_env.step(rec["a_tr"])
            tr_total += hopper_reward(obs, rec["a_tr"], prev)
            prev = obs.copy()
            if t or tr: break
        tr_rewards.append(tr_total)

        # True-MPC
        mpc_a, _ = true_mpc_hopper(fd_env, rec["qp"], rec["qv"], rec["a_tr"],
                                    H=args.H, n_cand=args.n_cand, eps_a=args.eps_a)
        fd_env.unwrapped.set_state(rec["qp"], rec["qv"])
        mpc_total = 0.0; prev = None
        for _ in range(args.H):
            obs, r, t, tr, _ = fd_env.step(mpc_a)
            mpc_total += hopper_reward(obs, mpc_a, prev)
            prev = obs.copy()
            if t or tr: break
        mpc_rewards.append(mpc_total)

    fd_env.close()

    tr_mean, mpc_mean = np.mean(tr_rewards), np.mean(mpc_rewards)
    imp = np.array(mpc_rewards) - np.array(tr_rewards)

    print(f"\n=== Phase 1 Results (Exact Hopper Reward) ===")
    print(f"  Transport H-step reward:  {tr_mean:.4f}")
    print(f"  True-MPC H-step reward:   {mpc_mean:.4f}")
    print(f"  Mean improvement:         {np.mean(imp):+.4f}")
    print(f"  Fraction improved:        {np.mean(imp > 0):.1%}")

    if mpc_mean > tr_mean * 1.05:
        print(f"\n  Verdict: PASS — exact reward MPC beats Transport")
    elif mpc_mean > tr_mean:
        print(f"\n  Verdict: MARGINAL — slight improvement")
    else:
        print(f"\n  Verdict: FAIL — even with exact reward and true dynamics, short-horizon MPC struggles")

    json.dump({"tr_mean": tr_mean, "mpc_mean": mpc_mean, "improved": float(np.mean(imp > 0))},
              open("results/phase1_exact_reward.json", "w"), indent=2)


if __name__ == "__main__":
    main()
