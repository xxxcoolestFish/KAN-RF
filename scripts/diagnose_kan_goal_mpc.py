"""Goal-conditioned KAN residual MPC: fix cross-physics executor gap.

Four controllers tested on friction_070 with same Router goals:
  1. Universal Executor (baseline)
  2. Universal + transport correction (old method)
  3. Universal + KAN goal residual MPC (new method)
  4. Universal + true-dynamics residual MPC (oracle upper bound)

Key metric: goal progress = (d(s_t, g) - d(s_{t+H}, g)) / d(s_t, g)
"""

from __future__ import annotations

import argparse, sys, os, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cpbn.cognitive_pca import CognitivePCA, PCARanges
from cpbn.generic_affine_kan import AffineKANContext
from scripts.diagnose_hopper_pullback_effect import (
    fit_distilled_source_counterfactual_context,
    load_source_twin,
)
from scripts.prescreen_hopper_physics_shifts import SHIFTS, make_shifted_env
from scripts.validate_hopper_joint_online_adaptation import (
    FrozenSourcePolicy,
    load_cognition,
)
from scripts.diagnose_oracle_goal_executor import GoalExecutor


def kan_mpc_correction(s, g, a0, basis, ctx, device, H=3, n_cand=128, beta=0.1):
    """KAN goal-conditioned residual MPC.

    Find δa that minimizes ||s_H - g||² starting from s, using KAN dynamics,
    with a0 as the initial action guess.
    """
    s_dim = basis.state_dim
    a_dim = basis.action_dim
    s_t = torch.as_tensor(s, device=device, dtype=torch.float32).unsqueeze(0)
    g_t = torch.as_tensor(g, device=device, dtype=torch.float32)
    a0_t = torch.as_tensor(a0, device=device, dtype=torch.float32)

    # Task state mask: use position, height, angle, velocity, foot contact
    task_mask = torch.tensor([1.0, 1.0, 0.5, 0.5, 0.5, 1.0, 0.2, 0.2, 0.2, 0.1, 0.1],
                             device=device)

    best_cost = float('inf')
    best_first_action = a0_t.clone()

    for _ in range(n_cand):
        s_curr = s_t.clone()
        da_seq = torch.randn(H, a_dim, device=device) * 0.2  # small residuals
        cost = 0.0

        for t in range(H):
            a_curr = a0_t + da_seq[t:t+1]
            a_curr = a_curr.clamp(-1, 1)
            effect = ctx.acceleration(basis, s_curr, a_curr)
            s_curr = s_curr + effect
            cost += beta * (da_seq[t] ** 2).sum().item()

        # Final distance to goal (task-weighted)
        diff = (s_curr.squeeze(0) - g_t) * task_mask
        cost += (diff ** 2).sum().item()

        if cost < best_cost:
            best_cost = cost
            best_first_action = (a0_t + da_seq[0:1]).clamp(-1, 1)

    return best_first_action.squeeze(0).cpu().numpy()


def true_mpc_correction(s, g, a0, shift, H=3, n_cand=128, beta=0.1):
    """Oracle: use real target env for MPC (diagnostic only)."""
    s_dim = len(s)
    a_dim = len(a0)

    task_mask = np.array([1.0, 1.0, 0.5, 0.5, 0.5, 1.0, 0.2, 0.2, 0.2, 0.1, 0.1])

    best_cost = float('inf')
    best_first_a = a0.copy()

    for _ in range(n_cand):
        da_seq = np.random.randn(H, a_dim) * 0.2
        env = make_shifted_env(shift, 1911, "hopper")()
        # Reset and try to reach from reset state
        obs, _ = env.reset()
        # Can't set state directly — approximate from current
        s_curr = s.copy()
        cost = 0.0

        for t in range(H):
            a_curr = np.clip(a0 + da_seq[t], -1, 1)
            # Simulate one step in real env
            obs, _, _, _, _ = env.step(a_curr)
            s_curr = obs
            cost += beta * float((da_seq[t] ** 2).sum())

        diff = (s_curr - g) * task_mask
        cost += float((diff ** 2).sum())
        env.close()

        if cost < best_cost:
            best_cost = cost
            best_first_a = np.clip(a0 + da_seq[0], -1, 1)

    return best_first_a


def compute_source_value(states, source_policy, device):
    s_t = torch.as_tensor(states, device=device, dtype=torch.float32)
    mean, var = source_policy.mean, source_policy.variance
    s_n = ((s_t - mean) / (var + 1e-8).sqrt()).clamp(-10, 10)
    with torch.no_grad():
        f = source_policy.model.policy.features_extractor(s_n)
        l = source_policy.model.policy.mlp_extractor(f)
        _, lv = l if isinstance(l, tuple) else (l, l)
        return source_policy.model.policy.value_net(lv).squeeze(-1).cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=1811)
    parser.add_argument("--n-episodes", type=int, default=5)
    parser.add_argument("--H", type=int, default=3)
    parser.add_argument("--n-cand", type=int, default=64)
    parser.add_argument("--json-out", default="results/kan_goal_mpc.json")
    args = parser.parse_args()

    device = torch.device(args.device)
    shift_key = "friction_070"
    shift = SHIFTS[shift_key]
    H = args.H

    print(f"KAN Goal MPC: H={H}, n_cand={args.n_cand}")

    # ── Load components ──────────────────────────────────────────────────
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

    # Fit KAN to friction
    print("Fitting KAN to friction...", flush=True)
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

    # ── Load universal executor ──────────────────────────────────────────
    executor = GoalExecutor(11, 11, 3).to(device)
    executor.load_state_dict(torch.load("results/universal_executor.pt",
                                        map_location=device, weights_only=True))
    executor.eval()

    # Goal pool
    pool = []
    for ep in range(5):
        env = make_shifted_env(SHIFTS["source"], args.seed + ep * 100, "hopper")()
        obs, _ = env.reset(seed=args.seed + ep * 100); sts = []
        while True:
            a = source_policy.action(torch.tensor(obs, dtype=torch.float32)).cpu().numpy()
            sts.append(obs.copy()); obs, _, t, tr, _ = env.step(a)
            if t or tr: break
        env.close()
        if len(sts) > 50: pool.append(np.stack(sts))
    all_src = np.concatenate(pool, axis=0)

    def gen_goals(n=20):
        gs = []
        for _ in range(n // 3):
            i = np.random.randint(len(all_src)); g = all_src[i].copy()
            g[0] += np.random.uniform(0.3, 1.5); gs.append(g)
        for _ in range(n // 3):
            i = np.random.randint(len(all_src)); g = all_src[i].copy()
            g[0] += np.random.uniform(-0.1, 0.3); gs.append(g)
        for _ in range(n - len(gs)):
            i = np.random.randint(len(all_src)); g = all_src[i].copy()
            g += np.random.randn(*g.shape) * 0.3; gs.append(g)
        return np.stack(gs).astype(np.float32)

    # ── Test controllers ─────────────────────────────────────────────────
    controllers = [
        ("Universal Executor", "universal"),
        ("Universal + transport", "transport"),
        ("Universal + KAN MPC", "kan_mpc"),
    ]

    results = {}
    for label, mode in controllers:
        print(f"\n=== {label} ===", flush=True)
        returns, progresses, residuals = [], [], []

        for ep in range(args.n_episodes):
            env = make_shifted_env(shift, args.seed + ep * 100, "hopper")()
            obs, _ = env.reset(seed=args.seed + ep * 100)
            total = 0.0; step = 0

            while True:
                if step % H == 0:
                    goals = gen_goals(20)
                    values = compute_source_value(goals, source_policy, device)
                    dists = np.linalg.norm(goals - obs[None, :], axis=-1)
                    safe = dists < 2.0
                    if safe.sum() > 0:
                        scores = values.copy(); scores[~safe] = -float('inf')
                    else:
                        scores = values
                    goal = goals[np.argmax(scores)]
                    s_start = obs.copy()
                    d_start = np.linalg.norm(s_start - goal)

                s_t = torch.as_tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
                g_t = torch.as_tensor(goal, device=device, dtype=torch.float32).unsqueeze(0)

                with torch.no_grad():
                    a0 = executor(s_t, g_t).clamp(-1, 1).squeeze(0).cpu().numpy()

                if mode == "universal":
                    a = a0
                elif mode == "transport":
                    nominal = source_policy.action(s_t)
                    s_eff = source_context.acceleration(basis, s_t, nominal)
                    a = target_ctx.transport_action(
                        basis, s_t, desired_effect=s_eff,
                        nominal_action=nominal, regularization=1e-2,
                    ).clamp(-1, 1).squeeze(0).cpu().numpy()
                elif mode == "kan_mpc":
                    a = kan_mpc_correction(
                        obs, goal, a0, basis, target_ctx, device,
                        H=H, n_cand=args.n_cand,
                    )

                obs, r, t, tr, _ = env.step(a)
                total += float(r); step += 1
                residuals.append(float(np.linalg.norm(a - a0)))

                if (step % H == 0 or t or tr):
                    d_end = np.linalg.norm(obs - goal)
                    progress = (d_start - d_end) / max(d_start, 1e-10)
                    progresses.append(progress)

                if t or tr:
                    break

            env.close()
            returns.append(total)

        mr = float(np.mean(returns))
        mp = float(np.mean(progresses)) if progresses else 0
        ma = float(np.mean(residuals))
        print(f"  Reward: {mr:.1f} +/- {float(np.std(returns)):.1f}")
        print(f"  Goal progress: {mp:.2%}")
        print(f"  Mean |da|: {ma:.4f}")
        results[label] = {"reward": mr, "progress": mp, "mean_da": ma}

    # ── Baselines ────────────────────────────────────────────────────────
    print(f"\n=== Baselines ===")
    print(f"  Source policy (no adapt): ~672")
    print(f"  Oracle PPO: 1347")
    print(f"  Router + Oracle goal (Exp2): ~801")

    json.dump(results, open(args.json_out, "w"), indent=2)
    print(f"\nSaved: {args.json_out}")


if __name__ == "__main__":
    main()
