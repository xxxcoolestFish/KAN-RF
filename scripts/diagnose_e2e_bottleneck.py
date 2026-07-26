"""Diagnose where 73 comes from: executor, router, or closed-loop accumulation?

Three quick tests:
  A: Can executor reach router-chosen goals? (single H-step)
  B: KAN reachability + Oracle Router (isolates KAN)
  C: No replanning — one goal, execute to end (isolates replanning)
"""

from __future__ import annotations

import argparse, json, sys, os, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.prescreen_hopper_physics_shifts import SHIFTS, make_shifted_env
from scripts.validate_hopper_joint_online_adaptation import FrozenSourcePolicy
from scripts.diagnose_oracle_goal_executor import GoalExecutor, collect_oracle_trajectories


def compute_source_value(states, source_policy, device):
    s_t = torch.as_tensor(states, device=device, dtype=torch.float32)
    mean, var = source_policy.mean, source_policy.variance
    s_n = ((s_t - mean) / (var + 1e-8).sqrt()).clamp(-10, 10)
    with torch.no_grad():
        f = source_policy.model.policy.features_extractor(s_n)
        l = source_policy.model.policy.mlp_extractor(f)
        _, lv = l if isinstance(l, tuple) else (l, l)
        return source_policy.model.policy.value_net(lv).squeeze(-1).cpu().numpy()


def generate_goals(pool, n=20):
    all_s = np.concatenate([s for s, _ in pool], axis=0)
    goals = []
    for _ in range(n // 3):
        i = np.random.randint(len(all_s)); g = all_s[i].copy()
        g[0] += np.random.uniform(0.3, 1.5); goals.append(g)
    for _ in range(n // 3):
        i = np.random.randint(len(all_s)); g = all_s[i].copy()
        g[0] += np.random.uniform(-0.1, 0.3); goals.append(g)
    for _ in range(n - len(goals)):
        i = np.random.randint(len(all_s)); g = all_s[i].copy()
        g += np.random.randn(*g.shape) * 0.3; goals.append(g)
    return np.stack(goals).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=1811)
    parser.add_argument("--n-steps", type=int, default=200)
    parser.add_argument("--H", type=int, default=4)
    args = parser.parse_args()

    device = torch.device(args.device)
    shift = SHIFTS["friction_070"]
    H = args.H

    # ── Load ─────────────────────────────────────────────────────────────
    source_policy = FrozenSourcePolicy(
        "results/hopper_source_sb3_ppo_continued_seed1811.zip",
        "results/hopper_source_sb3_vecnorm_continued_seed1811.pkl",
        device, args.seed, env="hopper",
    )

    # Train executor
    oracle_traj = collect_oracle_trajectories(shift, n_episodes=10, seed=args.seed)
    all_s, all_g, all_a = [], [], []
    for st, ac in oracle_traj:
        for t in range(len(st) - 10):
            all_s.append(st[t]); all_g.append(st[10+t]); all_a.append(ac[t])
    X_s = torch.tensor(np.stack(all_s), dtype=torch.float32)
    X_g = torch.tensor(np.stack(all_g), dtype=torch.float32)
    Y_a = torch.tensor(np.stack(all_a), dtype=torch.float32)
    s_dim, a_dim = X_s.shape[1], Y_a.shape[1]
    executor = GoalExecutor(s_dim, s_dim, a_dim).to(device)
    opt = torch.optim.Adam(executor.parameters(), lr=1e-3)
    loader = DataLoader(TensorDataset(X_s, X_g, Y_a), batch_size=256, shuffle=True)
    for _ in range(150):
        for sb, gb, ab in loader:
            sb, gb, ab = sb.to(device), gb.to(device), ab.to(device)
            (F.mse_loss(executor(sb, gb), ab)).backward()
            opt.step(); opt.zero_grad()

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
        if len(sts) > 50: pool.append((np.stack(sts), np.zeros((len(sts), 3))))

    oracle_states = oracle_traj[0][0]

    # ═══ Test A: Single-step goal error ═══
    print("=== Test A: Single H-step goal error ===")
    env = make_shifted_env(shift, args.seed, "hopper")()
    obs, _ = env.reset(seed=args.seed)
    errors = []
    for step in range(args.n_steps):
        goals = generate_goals(pool, n=20)
        values = compute_source_value(goals, source_policy, device)
        dists = np.linalg.norm(goals - obs[None, :], axis=-1)
        safe = dists < 2.0
        if safe.sum() > 0:
            scores = values.copy(); scores[~safe] = -float('inf')
        else:
            scores = values
        goal = goals[np.argmax(scores)]

        # Execute H steps towards goal
        s_start = obs.copy()
        for h in range(H):
            st = torch.as_tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
            gt = torch.as_tensor(goal, device=device, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                a = executor(st, gt).clamp(-1, 1).squeeze(0).cpu().numpy()
            obs, _, t, tr, _ = env.step(a)
            if t or tr: break
        errors.append(float(np.linalg.norm(obs - goal)))

    env.close()
    print(f"  Mean goal error: {np.mean(errors):.4f}")
    print(f"  Goal error < 1.0: {np.mean(np.array(errors) < 1.0):.1%}")
    print(f"  Goal error < 2.0: {np.mean(np.array(errors) < 2.0):.1%}")

    # ═══ Test B: Oracle Router with KAN reachability ═══
    print("\n=== Test B: Oracle goal (from oracle trajectory) ===")
    returns = []
    for ep in range(5):
        env = make_shifted_env(shift, args.seed + ep * 100, "hopper")()
        obs, _ = env.reset(seed=args.seed + ep * 100)
        total = 0.0; step = 0
        while True:
            # Oracle goal: future state from oracle trajectory
            g = oracle_states[min(step + H, len(oracle_states) - 1)]
            for h in range(H):
                st = torch.as_tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
                gt = torch.as_tensor(g, device=device, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    a = executor(st, gt).clamp(-1, 1).squeeze(0).cpu().numpy()
                obs, r, t, tr, _ = env.step(a)
                total += float(r); step += 1
                if t or tr: break
            if t or tr: break
        env.close(); returns.append(total)
    print(f"  Oracle goal executor: {np.mean(returns):.1f} +/- {np.std(returns):.1f}")

    # ═══ Test C: No replanning ═══
    print("\n=== Test C: Single goal, no replanning ===")
    returns = []
    for ep in range(5):
        env = make_shifted_env(shift, args.seed + ep * 100, "hopper")()
        obs, _ = env.reset(seed=args.seed + ep * 100)
        total = 0.0
        goals = generate_goals(pool, n=20)
        values = compute_source_value(goals, source_policy, device)
        dists = np.linalg.norm(goals - obs[None, :], axis=-1)
        safe = dists < 2.0
        if safe.sum() > 0:
            scores = values.copy(); scores[~safe] = -float('inf')
        else:
            scores = values
        goal = goals[np.argmax(scores)]

        while True:
            st = torch.as_tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
            gt = torch.as_tensor(goal, device=device, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                a = executor(st, gt).clamp(-1, 1).squeeze(0).cpu().numpy()
            obs, r, t, tr, _ = env.step(a)
            total += float(r)
            if t or tr: break
        env.close(); returns.append(total)
    print(f"  Single goal (no replan): {np.mean(returns):.1f} +/- {np.std(returns):.1f}")


if __name__ == "__main__":
    main()
