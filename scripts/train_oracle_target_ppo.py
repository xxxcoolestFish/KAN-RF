"""Train oracle PPO on target physics shifts to establish upper bounds.

Answers: is CPPE's 252 on friction_070 close to or far from what's possible?
"""

from __future__ import annotations

import argparse, json, sys, os
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.prescreen_hopper_physics_shifts import ENVS, SHIFTS, make_shifted_env


def evaluate(model, env, n_episodes=10):
    env.training = False
    env.norm_reward = False
    returns = []
    for _ in range(n_episodes):
        obs = env.reset()
        total = 0.0
        while True:
            a, _ = model.predict(obs, deterministic=True)
            obs, r, d, i = env.step(a)
            total += float(r[0])
            if d[0]:
                break
        returns.append(total)
    return float(np.mean(returns)), float(np.std(returns))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-steps", type=int, default=200_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=1811)
    parser.add_argument("--json-out", default="results/oracle_target_ppo.json")
    args = parser.parse_args()

    results = {}
    shifts = ["source"] + ["payload_125", "friction_070", "combo_medium"]

    for shift_name in shifts:
        print(f"\n=== Oracle PPO on {shift_name} ({args.total_steps} steps) ===", flush=True)

        shift = SHIFTS[shift_name]
        vec_env = DummyVecEnv([
            lambda seed_i=i: make_shifted_env(shift, args.seed + seed_i, "hopper")()
            for i in range(args.n_envs)
        ])
        vec_env = VecNormalize(vec_env, training=True, norm_obs=True, norm_reward=True)

        model = PPO("MlpPolicy", vec_env,
                    n_steps=2048 // args.n_envs,
                    batch_size=256,
                    n_epochs=5,
                    learning_rate=3e-4,
                    device=args.device, verbose=0)
        model.learn(total_timesteps=args.total_steps, progress_bar=False)

        r, s = evaluate(model, vec_env, n_episodes=10)
        print(f"  {shift_name}: {r:.1f} +/- {s:.1f}")
        results[shift_name] = {"mean": r, "std": s}
        vec_env.close()

    # Save
    os.makedirs(os.path.dirname(args.json_out), exist_ok=True)
    json.dump(results, open(args.json_out, "w"), indent=2)
    print(f"\nSaved: {args.json_out}")

    # Summary
    print("\n=== Oracle PPO Upper Bounds ===")
    r_src = results["source"]["mean"]
    for name in ["payload_125", "friction_070", "combo_medium"]:
        r_oracle = results[name]["mean"]
        print(f"  {name}: {r_oracle:.0f} ({r_oracle / r_src:.1%} of source {r_src:.0f})")


if __name__ == "__main__":
    main()
