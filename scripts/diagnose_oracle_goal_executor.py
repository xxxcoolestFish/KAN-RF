"""Experiment 1: Oracle local goal executor test.

Can a goal-conditioned executor reach oracle-provided local goals
in target physics, trained only with self-supervised goal-reaching error?

If NO -> three-network architecture is dead.
If YES -> proceed to Experiment 2.
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


def collect_oracle_trajectories(shift, n_episodes=10, seed=1911):
    """Collect trajectories from oracle PPO on target shift."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    venv = DummyVecEnv([lambda: make_shifted_env(shift, seed + i, "hopper")() for i in range(4)])
    venv = VecNormalize(venv, training=True, norm_obs=True, norm_reward=True)

    model = PPO("MlpPolicy", venv, n_steps=2048//4, batch_size=256, n_epochs=5,
                learning_rate=3e-4, device="cuda", verbose=0)
    model.learn(total_timesteps=200_000, progress_bar=False)

    # Collect trajectories with normalized env for prediction, raw states for goals
    eval_env = DummyVecEnv([lambda: make_shifted_env(shift, seed, "hopper")()])
    eval_env = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=False)
    eval_env.obs_rms = venv.obs_rms

    trajectories = []
    raw_env_for_state = None
    for ep in range(n_episodes):
        raw_env_for_state = make_shifted_env(shift, seed + ep * 100, "hopper")()
        raw_obs, _ = raw_env_for_state.reset(seed=seed + ep * 100)
        traj_states, traj_actions = [], []
        while True:
            norm_obs = venv.normalize_obs(raw_obs.reshape(1, -1))
            a, _ = model.predict(norm_obs, deterministic=True)
            traj_states.append(raw_obs.copy())  # raw state for goal
            traj_actions.append(a[0].copy())
            raw_obs, _, t, tr, _ = raw_env_for_state.step(a[0])
            if t or tr:
                break
        raw_env_for_state.close()
        if len(traj_states) > 50:
            trajectories.append((np.stack(traj_states), np.stack(traj_actions)))
    venv.close()
    return trajectories


class GoalExecutor(nn.Module):
    """MLP: (s_t, g_t) -> a_t. Learns to reach goal g_t from state s_t."""

    def __init__(self, s_dim, g_dim, a_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(s_dim + g_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, a_dim),
        )

    def forward(self, s, g):
        return self.net(torch.cat([s, g], dim=-1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shift", default="friction_070")
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--n-episodes", type=int, default=10)
    args = parser.parse_args()

    device = torch.device(args.device)
    shift = SHIFTS[args.shift]
    H = args.horizon

    # ── Collect oracle trajectories ──────────────────────────────────────
    print(f"Training oracle PPO on {args.shift}...", flush=True)
    trajectories = collect_oracle_trajectories(shift, n_episodes=args.n_episodes)

    # ── Build training data: (s_t, g_t) -> a_t ──────────────────────────
    all_s, all_g, all_a = [], [], []
    for states, actions in trajectories:
        for t in range(len(states) - H):
            all_s.append(states[t])
            all_g.append(states[t + H])  # oracle future state as goal
            all_a.append(actions[t])

    X_s = torch.tensor(np.stack(all_s), dtype=torch.float32)
    X_g = torch.tensor(np.stack(all_g), dtype=torch.float32)
    Y_a = torch.tensor(np.stack(all_a), dtype=torch.float32)
    print(f"Training data: {len(X_s)} samples (s_dim={X_s.shape[1]}, a_dim={Y_a.shape[1]})")

    # ── Train executor ───────────────────────────────────────────────────
    s_dim = X_s.shape[1]
    a_dim = Y_a.shape[1]
    executor = GoalExecutor(s_dim, s_dim, a_dim).to(device)
    optimizer = torch.optim.Adam(executor.parameters(), lr=args.lr)
    dataset = TensorDataset(X_s, X_g, Y_a)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)

    print(f"Training {args.epochs} epochs...", flush=True)
    for epoch in range(args.epochs):
        total_loss = 0.0
        for s_b, g_b, a_b in loader:
            s_b, g_b, a_b = s_b.to(device), g_b.to(device), a_b.to(device)
            pred = executor(s_b, g_b)
            loss = F.mse_loss(pred, a_b)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss)
        if epoch % 50 == 0:
            print(f"  epoch {epoch:3d}: loss={total_loss/len(loader):.6f}", flush=True)

    # ── Test closed-loop with oracle goals ───────────────────────────────
    print(f"\n=== Closed-loop test ===", flush=True)

    # Collect one oracle trajectory for goal sequence
    oracle_states, oracle_actions = trajectories[0]

    returns = []
    for ep in range(args.n_episodes):
        env = make_shifted_env(shift, 1911 + ep * 100, "hopper")()
        obs, _ = env.reset(seed=1911 + ep * 100)
        total = 0.0
        step = 0
        while step < len(oracle_states) - H:
            s_t = torch.tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
            g_t = torch.tensor(oracle_states[min(step + H, len(oracle_states) - 1)],
                               device=device, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                a = executor(s_t, g_t).clamp(-1, 1).squeeze(0).cpu().numpy()
            obs, r, t, tr, _ = env.step(a)
            total += float(r)
            step += 1
            if t or tr:
                break
        env.close()
        returns.append(total)
    print(f"  Executor (oracle goals): {np.mean(returns):.1f} +/- {np.std(returns):.1f}")

    # ── Baseline: source policy ──────────────────────────────────────────
    from scripts.validate_hopper_joint_online_adaptation import FrozenSourcePolicy
    sp = FrozenSourcePolicy(
        "results/hopper_source_sb3_ppo_continued_seed1811.zip",
        "results/hopper_source_sb3_vecnorm_continued_seed1811.pkl",
        device, 1811, env="hopper",
    )
    src_returns = []
    for ep in range(args.n_episodes):
        env = make_shifted_env(shift, 1911 + ep * 100, "hopper")()
        obs, _ = env.reset(seed=1911 + ep * 100)
        total = 0.0
        while True:
            a = sp.action(torch.tensor(obs, device=device, dtype=torch.float32)).cpu().numpy()
            obs, r, t, tr, _ = env.step(a)
            total += float(r)
            if t or tr: break
        env.close()
        src_returns.append(total)
    print(f"  Source policy (no adapt): {np.mean(src_returns):.1f} +/- {np.std(src_returns):.1f}")

    # ── Oracle PPO direct ────────────────────────────────────────────────
    oracle_r = 1347  # from previous experiment
    print(f"  Oracle PPO:              {oracle_r}")


if __name__ == "__main__":
    main()
