"""Prescreen zero-shot Hopper performance under hidden physics shifts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


SHIFTS = {
    "source": {},
    "payload_125": {"torso_mass": 1.25},
    "payload_150": {"torso_mass": 1.50},
    "friction_070": {"friction": 0.70},
    "actuator_080": {"actuator": 0.80},
    "actuator_065": {"actuator": 0.65},
    "combo_mild": {
        "torso_mass": 1.20, "friction": 0.85, "actuator": 0.85,
    },
    "combo_medium": {
        "torso_mass": 1.35, "friction": 0.70, "actuator": 0.75,
    },
}

ENVS = {
    "hopper": {"gym_id": "Hopper-v5", "mass_body": "torso"},
    "walker2d": {"gym_id": "Walker2d-v5", "mass_body": "torso"},
    "halfcheetah": {"gym_id": "HalfCheetah-v5", "mass_body": "torso"},
}


def make_shifted_env(shift, seed, env="hopper"):
    spec = ENVS[env]

    def factory():
        environment = gym.make(spec["gym_id"])
        model = environment.unwrapped.model
        if "torso_mass" in shift:
            torso_id = model.body(spec["mass_body"]).id
            model.body_mass[torso_id] *= shift["torso_mass"]
        if "friction" in shift:
            model.geom_friction[:, 0] *= shift["friction"]
        if "actuator" in shift:
            model.actuator_gear[:, 0] *= shift["actuator"]
        environment.reset(seed=seed)
        return environment
    return factory


def evaluate(model_path, norm_path, shift, args):
    base = DummyVecEnv(
        [make_shifted_env(shift, args.seed + 10000, args.env)],
    )
    environment = VecNormalize.load(norm_path, base)
    environment.training = False
    environment.norm_reward = False
    model = PPO.load(model_path, env=environment, device="cuda")
    returns = []
    lengths = []
    healthy = 0
    for _ in range(args.episodes):
        observation = environment.reset()
        total = 0.0
        length = 0
        while True:
            action, _ = model.predict(
                observation, deterministic=True,
            )
            observation, reward, done, info = environment.step(action)
            total += float(reward[0])
            length += 1
            if done[0]:
                healthy += int(
                    info[0].get("TimeLimit.truncated", False)
                )
                break
        returns.append(total)
        lengths.append(length)
    environment.close()
    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_episode_length": float(np.mean(lengths)),
        "healthy_completion_rate": healthy / args.episodes,
    }


def main(args):
    results = {}
    for name, shift in SHIFTS.items():
        results[name] = {
            "hidden_shift": shift,
            **evaluate(args.model, args.norm, shift, args),
        }
        print(name, results[name], flush=True)
    source_return = results["source"]["mean_return"]
    for result in results.values():
        result["normalized_return"] = (
            result["mean_return"] / source_return
        )
    output = {
        "experiment": "HopperPhysicsShiftPrescreen",
        "env": args.env,
        "seed": args.seed,
        "episodes": args.episodes,
        "physical_parameters_visible_to_policy": False,
        "results": results,
    }
    Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json_out).write_text(
        json.dumps(output, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1811)
    parser.add_argument(
        "--env", choices=tuple(ENVS), default="hopper",
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument(
        "--model",
        default="results/hopper_source_sb3_ppo_continued_seed1811.zip",
    )
    parser.add_argument(
        "--norm",
        default="results/hopper_source_sb3_vecnorm_continued_seed1811.pkl",
    )
    parser.add_argument(
        "--json-out",
        default="results/hopper_physics_shift_prescreen_seed1811.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
