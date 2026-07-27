"""Train the fixed-physics Pusher-v5 SAC task upper bound with debug logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from kanrf.pusher_oracle import object_goal_distance


def evaluate(model: SAC, episodes: int, seed: int) -> dict[str, float]:
    returns: list[float] = []
    final_distances: list[float] = []
    minimum_distances: list[float] = []
    for episode in range(episodes):
        env = gym.make("Pusher-v5")
        observation, _ = env.reset(seed=seed + episode * 101)
        total = 0.0
        distances = [object_goal_distance(env)]
        while True:
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, _ = env.step(action)
            total += float(reward)
            distances.append(object_goal_distance(env))
            if terminated or truncated:
                break
        env.close()
        returns.append(total)
        final_distances.append(distances[-1])
        minimum_distances.append(min(distances))
    return {
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "final_distance_mean": float(np.mean(final_distances)),
        "minimum_distance_mean": float(np.mean(minimum_distances)),
        "success_any_005": float(np.mean(np.asarray(minimum_distances) < 0.05)),
    }


class DebugCallback(BaseCallback):
    def __init__(
        self,
        log_every: int,
        eval_every: int,
        eval_episodes: int,
        seed: int,
    ):
        super().__init__()
        self.log_every = log_every
        self.eval_every = eval_every
        self.eval_episodes = eval_episodes
        self.seed = seed
        self.started = perf_counter()
        self.last_log = 0
        self.last_eval = 0
        self.evaluations: list[dict] = []

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_log >= self.log_every:
            self.last_log = self.num_timesteps
            elapsed = perf_counter() - self.started
            episode_infos = list(self.model.ep_info_buffer)
            recent_reward = (
                float(np.mean([item["r"] for item in episode_infos]))
                if episode_infos
                else float("nan")
            )
            logger_values = self.model.logger.name_to_value
            replay_size = self.model.replay_buffer.size()
            gpu_mb = (
                torch.cuda.memory_allocated() / 1024**2
                if torch.cuda.is_available()
                else 0.0
            )
            print(
                "TRAIN "
                f"steps={self.num_timesteps} "
                f"fps={self.num_timesteps/max(elapsed, 1e-6):.1f} "
                f"episodes={len(episode_infos)} "
                f"recent_return={recent_reward:.3f} "
                f"replay={replay_size} "
                f"actor_loss={logger_values.get('train/actor_loss', float('nan')):.5f} "
                f"critic_loss={logger_values.get('train/critic_loss', float('nan')):.5f} "
                f"ent_coef={logger_values.get('train/ent_coef', float('nan')):.5f} "
                f"gpu_mem_mb={gpu_mb:.1f}",
                flush=True,
            )

        if self.num_timesteps - self.last_eval >= self.eval_every:
            self.last_eval = self.num_timesteps
            metrics = evaluate(
                self.model,
                episodes=self.eval_episodes,
                seed=self.seed + 100_000 + self.num_timesteps,
            )
            record = {"steps": self.num_timesteps, **metrics}
            self.evaluations.append(record)
            print(
                "EVAL "
                f"steps={self.num_timesteps} "
                f"return={metrics['return_mean']:.3f}"
                f"+/-{metrics['return_std']:.3f} "
                f"final_distance={metrics['final_distance_mean']:.4f} "
                f"min_distance={metrics['minimum_distance_mean']:.4f} "
                f"success_any={metrics['success_any_005']:.2%}",
                flush=True,
            )
        return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-steps", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=1811)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--learning-starts", type=int, default=5_000)
    parser.add_argument("--buffer-size", type=int, default=500_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--train-freq", type=int, default=4)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=5_000)
    parser.add_argument("--eval-every", type=int, default=25_000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument(
        "--model-out",
        default="results/pusher_source_sac_seed1811",
    )
    parser.add_argument(
        "--json-out",
        default="results/pusher_source_sac_seed1811.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env = Monitor(gym.make("Pusher-v5"))
    env.reset(seed=args.seed)
    print(
        "CONFIG "
        f"steps={args.total_steps} seed={args.seed} device={args.device} "
        f"obs={env.observation_space.shape} action={env.action_space.shape} "
        f"learning_starts={args.learning_starts} buffer={args.buffer_size} "
        f"batch={args.batch_size} train_freq={args.train_freq} "
        f"gradient_steps={args.gradient_steps}",
        flush=True,
    )
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        gamma=0.99,
        tau=0.005,
        ent_coef="auto",
        policy_kwargs={"net_arch": [256, 256]},
        seed=args.seed,
        device=args.device,
        verbose=0,
    )
    callback = DebugCallback(
        log_every=args.log_every,
        eval_every=args.eval_every,
        eval_episodes=args.eval_episodes,
        seed=args.seed,
    )
    model.learn(total_timesteps=args.total_steps, callback=callback, progress_bar=False)
    output = Path(args.model_out)
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save(output)
    final_metrics = evaluate(model, args.eval_episodes, args.seed + 200_000)
    record = {
        "config": vars(args),
        "device": str(model.device),
        "evaluations": callback.evaluations,
        "final": final_metrics,
    }
    Path(args.json_out).write_text(
        json.dumps(record, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    env.close()
    print(f"SAVED model={output.resolve()} metrics={Path(args.json_out).resolve()}")
    print(f"FINAL {final_metrics}", flush=True)


if __name__ == "__main__":
    main()
