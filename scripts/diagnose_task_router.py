"""Experiment 2: Task router with oracle reachability.

Key question: can source-trained value function pick good goals
when given correct reachability information?

Setup:
  - Executor from Exp 1 (goal-conditioned, works)
  - Candidates: oracle trajectory states (all reachable by construction)
  - Router: V_source(g) selects best goal
  - Groups: oracle_goal / router / random

If router ≈ oracle_goal → task knowledge transfers across physics.
If router ≈ random → task knowledge doesn't transfer (core assumption fails).
"""

from __future__ import annotations

import argparse, sys, os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.prescreen_hopper_physics_shifts import SHIFTS, make_shifted_env
from scripts.diagnose_oracle_goal_executor import GoalExecutor, collect_oracle_trajectories


def compute_source_value(states, source_policy, device):
    """Use source PPO critic V(s) to score states."""
    s_t = torch.tensor(states, device=device, dtype=torch.float32)
    mean = source_policy.mean
    var = source_policy.variance
    s_norm = ((s_t - mean) / (var + 1e-8).sqrt()).clamp(-10, 10)
    with torch.no_grad():
        features = source_policy.model.policy.features_extractor(s_norm)
        latent = source_policy.model.policy.mlp_extractor(features)
        if isinstance(latent, tuple):
            _, latent_vf = latent
        else:
            latent_vf = latent
        value = source_policy.model.policy.value_net(latent_vf)
    return value.squeeze(-1).cpu().numpy()


def get_future_states(trajectory_states, t, horizon, n_candidates=20):
    """Get candidate goals: future states from oracle trajectory + noise."""
    max_t = len(trajectory_states) - 1
    candidates = []
    # Future states at various lookaheads
    for h in range(horizon, horizon * 3, horizon // 2):
        idx = min(t + h, max_t)
        candidates.append(trajectory_states[idx].copy())
    # Perturbed versions
    for _ in range(n_candidates - len(candidates)):
        base_idx = min(t + horizon, max_t)
        noise = np.random.randn(*trajectory_states[base_idx].shape) * 0.05
        candidates.append(trajectory_states[base_idx] + noise)
    return np.stack(candidates)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shift", default="friction_070")
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--n-episodes", type=int, default=10)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    shift = SHIFTS[args.shift]
    H = args.horizon

    # ── Load source policy ──────────────────────────────────────────────
    from scripts.validate_hopper_joint_online_adaptation import FrozenSourcePolicy
    source_policy = FrozenSourcePolicy(
        "results/hopper_source_sb3_ppo_continued_seed1811.zip",
        "results/hopper_source_sb3_vecnorm_continued_seed1811.pkl",
        device, 1811, env="hopper",
    )

    # ── Collect oracle + source trajectories ─────────────────────────────
    print("Collecting trajectories...", flush=True)
    oracle_traj = collect_oracle_trajectories(shift, n_episodes=10)
    oracle_states, oracle_actions = oracle_traj[0]  # Use first trajectory

    # Source trajectories for value reference
    src_states_all = []
    for ep in range(5):
        env = make_shifted_env(SHIFTS["source"], 1811 + ep * 100, "hopper")()
        obs, _ = env.reset(seed=1811 + ep * 100)
        while True:
            a = source_policy.action(torch.tensor(obs, device=device, dtype=torch.float32)).cpu().numpy()
            src_states_all.append(obs.copy())
            obs, _, t, tr, _ = env.step(a)
            if t or tr: break
        env.close()
    src_states_all = np.stack(src_states_all)
    print(f"  Oracle traj: {len(oracle_states)} states, Source: {len(src_states_all)} states")

    # ── Train executor on oracle data ────────────────────────────────────
    print("Training executor...", flush=True)
    all_s, all_g, all_a = [], [], []
    for st, ac in oracle_traj:
        for t in range(len(st) - H):
            all_s.append(st[t]); all_g.append(st[H + t]); all_a.append(ac[t])
    X_s = torch.tensor(np.stack(all_s), dtype=torch.float32)
    X_g = torch.tensor(np.stack(all_g), dtype=torch.float32)
    Y_a = torch.tensor(np.stack(all_a), dtype=torch.float32)
    executor = GoalExecutor(X_s.shape[1], X_s.shape[1], Y_a.shape[1]).to(device)
    opt = torch.optim.Adam(executor.parameters(), lr=1e-3)
    loader = DataLoader(TensorDataset(X_s, X_g, Y_a), batch_size=256, shuffle=True)
    for _ in range(100):
        for sb, gb, ab in loader:
            sb, gb, ab = sb.to(device), gb.to(device), ab.to(device)
            loss = F.mse_loss(executor(sb, gb), ab)
            opt.zero_grad(); loss.backward(); opt.step()
    print("  Done.")

    # ── Test groups ──────────────────────────────────────────────────────
    for group_label, group_mode in [
        ("Oracle goal (H-step future)", "oracle"),
        ("Router: V_source picks goal", "router"),
        ("Random oracle state", "random"),
        ("Source policy (no adapt)", "source"),
    ]:
        returns = []

        for ep in range(args.n_episodes):
            env = make_shifted_env(shift, 1811 + ep * 100, "hopper")()
            obs, _ = env.reset(seed=1811 + ep * 100)
            total = 0.0; step = 0

            while True:
                if group_mode == "oracle":
                    goal = oracle_states[min(step + H, len(oracle_states) - 1)]
                elif group_mode == "router":
                    cand = get_future_states(oracle_states, step % (len(oracle_states) - H*3), H, 20)
                    v = compute_source_value(cand, source_policy, device)
                    goal = cand[np.argmax(v)]
                elif group_mode == "random":
                    idx = np.random.randint(0, len(oracle_states))
                    goal = oracle_states[idx]
                elif group_mode == "source":
                    # Direct source policy (no executor)
                    a = source_policy.action(torch.tensor(obs, device=device, dtype=torch.float32)).cpu().numpy()
                    obs, r, t, tr, _ = env.step(a)
                    total += float(r); step += 1
                    if t or tr: break
                    continue

                # Single-step execution with goal (matches Exp 1 loop)
                s_t = torch.tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
                g_t = torch.tensor(goal, device=device, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    a = executor(s_t, g_t).clamp(-1, 1).squeeze(0).cpu().numpy()
                obs, r, t, tr, _ = env.step(a)
                total += float(r); step += 1
                if t or tr:
                    break

            env.close()
            returns.append(total)

        print(f"  {group_label:35s}: {np.mean(returns):8.1f} +/- {np.std(returns):.1f}")

    print(f"\n  Oracle PPO (200K): 1347")


if __name__ == "__main__":
    main()
