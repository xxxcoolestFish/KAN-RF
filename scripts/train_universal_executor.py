"""Train a universal goal-conditioned executor: E(s, g) -> a.

Training data: (s, g) pairs from source trajectories with diverse goals:
  - Same-state: g = s (maintain)
  - Near-future: g = s_{t+H} (source trajectory goals)
  - Perturbed: g = s_{t+H} + noise (robustness)
  - Random walk: g = nearby random state

This covers the Router's goal distribution.
"""

from __future__ import annotations

import argparse, json, sys, os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.prescreen_hopper_physics_shifts import SHIFTS, make_shifted_env
from scripts.validate_hopper_joint_online_adaptation import FrozenSourcePolicy
from scripts.diagnose_oracle_goal_executor import GoalExecutor


def collect_diverse_training_data(source_policy, device, n_episodes=20, H_range=(1, 16)):
    """Collect (s, g, a) pairs with diverse goals from source trajectories.

    Goal types:
      - Same state: g = s
      - Near future: g = s_{t+H} for random H
      - Perturbed future: g = s_{t+H} + noise
      - Forward push: g = s with increased x and velocity
      - Stance hold: g = s with same position, different phase
    """
    all_s, all_g, all_a = [], [], []

    for ep in range(n_episodes):
        env = make_shifted_env(SHIFTS["source"], 1811 + ep * 100, "hopper")()
        obs, _ = env.reset(seed=1811 + ep * 100)
        traj_s, traj_a = [], []

        while True:
            s_t = torch.tensor(obs, device=device, dtype=torch.float32)
            a = source_policy.action(s_t).cpu().numpy()
            traj_s.append(obs.copy())
            traj_a.append(a.copy())
            obs, _, t, tr, _ = env.step(a)
            if t or tr: break
        env.close()

        if len(traj_s) < 100:
            continue

        traj_s = np.stack(traj_s)
        traj_a = np.stack(traj_a)

        for t in range(len(traj_s) - max(H_range)):
            s_t = traj_s[t]
            a_t = traj_a[t]

            # 1. Same state goal
            all_s.append(s_t); all_g.append(s_t.copy()); all_a.append(a_t)

            # 2. Near future goals at various horizons
            for H in range(H_range[0], H_range[1] + 1, 2):
                if t + H < len(traj_s):
                    g = traj_s[t + H].copy()
                    all_s.append(s_t); all_g.append(g); all_a.append(a_t)

                    # 3. Perturbed future
                    g_pert = g + np.random.randn(*g.shape) * np.array(
                        [0.2, 0.05, 0.05, 0.05, 0.05, 0.5, 0.1, 0.1, 0.1, 0.05, 0.05]
                    )
                    all_s.append(s_t); all_g.append(g_pert); all_a.append(a_t)

            # 4. Forward push goals
            g_fwd = s_t.copy()
            g_fwd[0] += np.random.uniform(0.2, 1.5)  # forward
            g_fwd[5] += np.random.uniform(0.5, 3.0)  # faster
            all_s.append(s_t); all_g.append(g_fwd); all_a.append(a_t)

            # 5. Stance goals (stay in place, vary height/angle)
            g_stance = s_t.copy()
            g_stance[1] += np.random.uniform(-0.1, 0.15)  # height change
            g_stance[2] += np.random.uniform(-0.1, 0.1)   # angle
            all_s.append(s_t); all_g.append(g_stance); all_a.append(a_t)

    return (np.stack(all_s).astype(np.float32),
            np.stack(all_g).astype(np.float32),
            np.stack(all_a).astype(np.float32))


def evaluate_executor(executor, source_policy, shift, device, n_episodes=10, H=4):
    """Test executor with Router-style goals on friction_070."""
    # Train oracle trajectory first
    from scripts.diagnose_oracle_goal_executor import collect_oracle_trajectories
    oracle_traj = collect_oracle_trajectories(shift, n_episodes=10, seed=1811)
    oracle_states = oracle_traj[0][0]

    # Source goal pool
    pool = []
    for ep in range(5):
        env = make_shifted_env(SHIFTS["source"], 1811 + ep * 100, "hopper")()
        obs, _ = env.reset(seed=1811 + ep * 100)
        sts = []
        while True:
            a = source_policy.action(torch.tensor(obs, dtype=torch.float32)).cpu().numpy()
            sts.append(obs.copy()); obs, _, t, tr, _ = env.step(a)
            if t or tr: break
        env.close()
        if len(sts) > 50: pool.append(np.stack(sts))
    all_src = np.concatenate(pool, axis=0)

    def compute_v(states):
        s_t = torch.as_tensor(states, device=device, dtype=torch.float32)
        mean, var = source_policy.mean, source_policy.variance
        s_n = ((s_t - mean) / (var + 1e-8).sqrt()).clamp(-10, 10)
        with torch.no_grad():
            f = source_policy.model.policy.features_extractor(s_n)
            l = source_policy.model.policy.mlp_extractor(f)
            _, lv = l if isinstance(l, tuple) else (l, l)
            return source_policy.model.policy.value_net(lv).squeeze(-1).cpu().numpy()

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

    # --- Router-style test ---
    returns = []
    for ep in range(n_episodes):
        env = make_shifted_env(shift, 1811 + ep * 100, "hopper")()
        obs, _ = env.reset(seed=1811 + ep * 100)
        total = 0.0; step = 0
        while True:
            if step % H == 0:
                goals = gen_goals(20)
                values = compute_v(goals)
                dists = np.linalg.norm(goals - obs[None, :], axis=-1)
                safe = dists < 2.0
                if safe.sum() > 0:
                    scores = values.copy(); scores[~safe] = -float('inf')
                    goal = goals[np.argmax(scores)]
                else:
                    goal = goals[np.argmax(values)]

            st = torch.as_tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
            gt = torch.as_tensor(goal, device=device, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                a = executor(st, gt).clamp(-1, 1).squeeze(0).cpu().numpy()
            obs, r, t, tr, _ = env.step(a)
            total += float(r); step += 1
            if t or tr: break
        env.close(); returns.append(total)
    return float(np.mean(returns)), float(np.std(returns))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=1811)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--n-episodes", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    device = torch.device(args.device)
    shift = SHIFTS["friction_070"]

    print("Loading source policy...", flush=True)
    source_policy = FrozenSourcePolicy(
        "results/hopper_source_sb3_ppo_continued_seed1811.zip",
        "results/hopper_source_sb3_vecnorm_continued_seed1811.pkl",
        device, args.seed, env="hopper",
    )

    # ── Collect diverse training data ────────────────────────────────────
    print(f"Collecting {args.n_episodes} episodes for diverse goals...", flush=True)
    X_s, X_g, Y_a = collect_diverse_training_data(source_policy, device, args.n_episodes)
    print(f"  {len(X_s)} training samples (s_dim={X_s.shape[1]}, a_dim={Y_a.shape[1]})")

    # ── Train executor ───────────────────────────────────────────────────
    executor = GoalExecutor(X_s.shape[1], X_s.shape[1], Y_a.shape[1]).to(device)
    optimizer = torch.optim.Adam(executor.parameters(), lr=args.lr)
    s_t = torch.as_tensor(X_s, device=device)
    g_t = torch.as_tensor(X_g, device=device)
    a_t = torch.as_tensor(Y_a, device=device)
    loader = DataLoader(TensorDataset(s_t, g_t, a_t), batch_size=512, shuffle=True)

    print(f"Training {args.epochs} epochs...", flush=True)
    for epoch in range(args.epochs):
        total_loss = 0.0
        for sb, gb, ab in loader:
            pred = executor(sb, gb)
            loss = F.mse_loss(pred, ab)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += float(loss)
        if epoch % 50 == 0:
            print(f"  epoch {epoch:3d}: loss={total_loss/len(loader):.6f}")

    # ── Evaluate ─────────────────────────────────────────────────────────
    print("\n=== Evaluation on friction_070 ===")
    r, s = evaluate_executor(executor, source_policy, shift, device, n_episodes=10)
    print(f"  Universal Executor + Router goals: {r:.1f} +/- {s:.1f}")

    # Compare with source policy
    src_rets = []
    for ep in range(10):
        env = make_shifted_env(shift, 1811 + ep * 100, "hopper")()
        obs, _ = env.reset(seed=1811 + ep * 100); total = 0.0
        while True:
            a = source_policy.action(torch.tensor(obs, dtype=torch.float32)).cpu().numpy()
            obs, r, t, tr, _ = env.step(a); total += float(r)
            if t or tr: break
        env.close(); src_rets.append(total)
    print(f"  Source policy (no adapt):       {np.mean(src_rets):.1f} +/- {np.std(src_rets):.1f}")
    print(f"  Oracle PPO:                      1347")

    # Save
    torch.save(executor.state_dict(), "results/universal_executor.pt")
    print("\nSaved: results/universal_executor.pt")


if __name__ == "__main__":
    main()
