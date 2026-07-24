"""Train a reliable Hopper-v5 source PPO with Stable-Baselines3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from scripts.prescreen_hopper_physics_shifts import SHIFTS


def make_env(seed, shift):
    def factory():
        environment = gym.make("Hopper-v5")
        model = environment.unwrapped.model
        if "torso_mass" in shift:
            torso_id = model.body("torso").id
            model.body_mass[torso_id] *= shift["torso_mass"]
        if "friction" in shift:
            model.geom_friction[:, 0] *= shift["friction"]
        if "actuator" in shift:
            model.actuator_gear[:, 0] *= shift["actuator"]
        environment.reset(seed=seed)
        return environment
    return factory


def evaluate(model, norm, shift, args):
    environment = DummyVecEnv(
        [make_env(args.seed + 10000, shift)],
    )
    evaluation = VecNormalize(
        environment,
        training=False,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
    )
    evaluation.obs_rms = norm.obs_rms
    returns = []
    lengths = []
    healthy = 0
    for _ in range(args.evaluation_episodes):
        observation = evaluation.reset()
        total_reward = 0.0
        length = 0
        while True:
            action, _ = model.predict(
                observation, deterministic=True,
            )
            observation, reward, done, info = evaluation.step(action)
            total_reward += float(reward[0])
            length += 1
            if done[0]:
                truncated = bool(
                    info[0].get("TimeLimit.truncated", False)
                )
                healthy += int(truncated)
                break
        returns.append(total_reward)
        lengths.append(length)
    evaluation.close()
    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_episode_length": float(np.mean(lengths)),
        "healthy_completion_rate": healthy / args.evaluation_episodes,
    }


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}", flush=True)
    shift = SHIFTS[args.physics_shift]
    environments = DummyVecEnv(
        [
            make_env(args.seed + index, shift)
            for index in range(args.parallel_envs)
        ],
    )
    if args.initial_norm:
        normalized = VecNormalize.load(args.initial_norm, environments)
        normalized.training = not args.freeze_normalization
        normalized.norm_reward = True
    else:
        normalized = VecNormalize(
            environments,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
            gamma=args.gamma,
        )
    if args.initial_model:
        model = PPO.load(
            args.initial_model,
            env=normalized,
            device=device,
        )
        model.learning_rate = args.learning_rate
        model.lr_schedule = get_schedule_fn(args.learning_rate)
        model.verbose = 1
    else:
        model = PPO(
            "MlpPolicy",
            normalized,
            learning_rate=args.learning_rate,
            n_steps=args.rollout_steps,
            batch_size=args.minibatch_size,
            n_epochs=args.update_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_ratio,
            ent_coef=args.entropy_coefficient,
            vf_coef=args.value_coefficient,
            max_grad_norm=args.gradient_clip,
            policy_kwargs={"net_arch": dict(pi=[64, 64], vf=[64, 64])},
            seed=args.seed,
            device=device,
            verbose=1,
        )
    model.learn(total_timesteps=args.total_transitions)
    metrics = evaluate(model, normalized, shift, args)
    print(metrics, flush=True)
    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    model.save(args.model_out)
    normalized.save(args.norm_out)
    output = {
        "experiment": "HopperSourceSB3PPO",
        "seed": args.seed,
        "device": device,
        "physics_shift": args.physics_shift,
        "hidden_shift": shift,
        "physical_parameters_visible_to_learner": False,
        "initialized_from_checkpoint": bool(args.initial_model),
        "config": vars(args),
        "evaluation": metrics,
    }
    Path(args.json_out).write_text(
        json.dumps(output, indent=2) + "\n",
        encoding="utf-8",
    )
    normalized.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1811)
    parser.add_argument(
        "--physics-shift",
        choices=tuple(SHIFTS),
        default="source",
    )
    parser.add_argument("--initial-model", default="")
    parser.add_argument("--initial-norm", default="")
    parser.add_argument(
        "--freeze-normalization",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--total-transitions", type=int, default=1_000_000)
    parser.add_argument("--parallel-envs", type=int, default=8)
    parser.add_argument("--rollout-steps", type=int, default=2048)
    parser.add_argument("--minibatch-size", type=int, default=64)
    parser.add_argument("--update-epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--entropy-coefficient", type=float, default=0.0)
    parser.add_argument("--value-coefficient", type=float, default=0.5)
    parser.add_argument("--gradient-clip", type=float, default=0.5)
    parser.add_argument("--evaluation-episodes", type=int, default=20)
    parser.add_argument(
        "--model-out",
        default="results/hopper_source_sb3_ppo_seed1811",
    )
    parser.add_argument(
        "--norm-out",
        default="results/hopper_source_sb3_vecnorm_seed1811.pkl",
    )
    parser.add_argument(
        "--json-out",
        default="results/hopper_source_sb3_ppo_seed1811.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
