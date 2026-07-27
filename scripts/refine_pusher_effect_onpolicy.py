"""Refine a task-effect value on states queried by its own CEM planner."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from time import perf_counter

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

from kanrf.effect_interface import TaskEffectValue, effect_covariance_loss
from kanrf.pusher_oracle import PusherOracleCEM, object_goal_distance
from scripts.train_pusher_effect_value import (
    collect_observations,
    correlation,
    critic_values,
    load_sac,
)


def load_effect(path: str) -> tuple[TaskEffectValue, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "task_effect_value_v1":
        raise ValueError(f"unsupported checkpoint {checkpoint.get('format')}")
    model = TaskEffectValue(**checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def collect_planner_queries(
    effect: TaskEffectValue,
    episodes: int,
    max_steps: int,
    query_candidates: int,
    query_noise_scale: float,
    args: argparse.Namespace,
) -> tuple[np.ndarray, list[dict]]:
    rng = np.random.default_rng(args.seed + 4401)
    query_observations: list[np.ndarray] = []
    episode_records = []

    def effect_value(states: np.ndarray) -> np.ndarray:
        tensor = torch.as_tensor(states, dtype=torch.float32)
        with torch.no_grad():
            _, values = effect(tensor)
        return values.numpy()

    for episode in range(episodes):
        env = gym.make("Pusher-v5", max_episode_steps=max_steps)
        observation, _ = env.reset(seed=args.seed + 101 * episode)
        planner = PusherOracleCEM(
            horizon=args.horizon,
            action_repeat=args.action_repeat,
            population=args.population,
            elite_fraction=args.elite_fraction,
            iterations=args.iterations,
            discount=args.discount,
            initial_std_scale=args.initial_std_scale,
            temporal_correlation=args.temporal_correlation,
            seed=args.seed + 9000 + episode,
        )
        initial_distance = object_goal_distance(env)
        total_reward = 0.0
        for step in range(max_steps):
            plan = planner.plan(env, terminal_value_fn=effect_value)
            base = env.unwrapped
            qpos = np.asarray(base.data.qpos, dtype=np.float64).copy()
            qvel = np.asarray(base.data.qvel, dtype=np.float64).copy()
            noise = rng.normal(
                0.0,
                query_noise_scale * (planner.high - planner.low),
                size=(
                    query_candidates,
                    args.horizon,
                    planner.action_dim,
                ),
            )
            candidates = np.clip(
                plan.sequence[None, ...] + noise,
                planner.low,
                planner.high,
            )
            candidates[0] = plan.sequence
            if query_candidates > 1:
                candidates[1] = 0.0
            _, terminals, _ = planner.rollout_sequences(
                qpos,
                qvel,
                candidates,
            )
            query_observations.extend(terminals)
            query_observations.append(np.asarray(observation).copy())
            observation, reward, terminated, truncated, _ = env.step(
                plan.action
            )
            total_reward += float(reward)
            if args.debug_every > 0 and (
                step % args.debug_every == 0 or terminated or truncated
            ):
                print(
                    f"QUERY ep={episode} step={step:03d} "
                    f"queries={len(query_observations)} "
                    f"distance={object_goal_distance(env):.4f} "
                    f"action_norm={np.linalg.norm(plan.action):.3f}",
                    flush=True,
                )
            if terminated or truncated:
                break
        final_distance = object_goal_distance(env)
        episode_records.append(
            {
                "episode": episode,
                "return": total_reward,
                "initial_distance": initial_distance,
                "final_distance": final_distance,
                "progress": initial_distance - final_distance,
            }
        )
        planner.close()
        env.close()
    return np.stack(query_observations), episode_records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sac-model",
        default="results/pusher-v5-SAC-expert.zip",
    )
    parser.add_argument(
        "--input-model",
        default="results/pusher_task_effect_value_d4_capacity.pt",
    )
    parser.add_argument(
        "--output",
        default="results/pusher_task_effect_value_d4_onpolicy.pt",
    )
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--query-candidates", type=int, default=32)
    parser.add_argument("--query-noise-scale", type=float, default=0.15)
    parser.add_argument("--replay-states", type=int, default=10000)
    parser.add_argument("--replay-repeat", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--covariance-weight", type=float, default=1e-3)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--action-repeat", type=int, default=3)
    parser.add_argument("--population", type=int, default=64)
    parser.add_argument("--elite-fraction", type=float, default=0.1)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--initial-std-scale", type=float, default=0.15)
    parser.add_argument("--temporal-correlation", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=1811)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--debug-every", type=int, default=10)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    teacher = load_sac(args.sac_model, args.device)
    effect, source_checkpoint = load_effect(args.input_model)
    start = perf_counter()
    query_observations, collection_records = collect_planner_queries(
        effect,
        episodes=args.episodes,
        max_steps=args.max_steps,
        query_candidates=args.query_candidates,
        query_noise_scale=args.query_noise_scale,
        args=args,
    )
    replay_observations, _ = collect_observations(
        teacher,
        count=args.replay_states,
        num_envs=8,
        seed=args.seed + 1,
        debug_every=max(args.replay_states // 2, 1),
    )
    observations = np.concatenate(
        (
            np.repeat(
                query_observations,
                args.replay_repeat,
                axis=0,
            ),
            replay_observations,
        )
    ).astype(np.float32)
    values = critic_values(teacher, observations, batch_size=4096)
    permutation = np.random.default_rng(args.seed + 71).permutation(
        len(observations)
    )
    validation_count = max(
        1,
        int(len(observations) * args.validation_fraction),
    )
    validation_indices = permutation[:validation_count]
    training_indices = permutation[validation_count:]
    device = torch.device(args.device)
    effect = effect.to(device)
    observation_tensor = torch.as_tensor(observations, device=device)
    value_tensor = torch.as_tensor(values, device=device)
    optimizer = torch.optim.AdamW(
        effect.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    best_mse = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        effect.train()
        order = torch.as_tensor(
            np.random.default_rng(args.seed + epoch).permutation(
                training_indices
            ),
            device=device,
        )
        total_loss = 0.0
        batches = 0
        for start_index in range(0, len(order), args.batch_size):
            indices = order[start_index : start_index + args.batch_size]
            effects, predictions = effect(observation_tensor[indices])
            normalized_predictions = (
                predictions - effect.value_mean
            ) / effect.value_std
            normalized_targets = (
                value_tensor[indices] - effect.value_mean
            ) / effect.value_std
            value_loss = F.mse_loss(
                normalized_predictions,
                normalized_targets,
            )
            covariance_loss = effect_covariance_loss(effects)
            loss = value_loss + args.covariance_weight * covariance_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(effect.parameters(), 10.0)
            optimizer.step()
            total_loss += float(value_loss.detach())
            batches += 1

        effect.eval()
        validation_index_tensor = torch.as_tensor(
            validation_indices,
            device=device,
        )
        with torch.no_grad():
            _, validation_predictions = effect(
                observation_tensor[validation_index_tensor]
            )
            validation_targets = value_tensor[validation_index_tensor]
            validation_mse = F.mse_loss(
                validation_predictions,
                validation_targets,
            )
            validation_rmse = float(torch.sqrt(validation_mse))
            validation_correlation = correlation(
                validation_predictions,
                validation_targets,
            )
        if float(validation_mse) < best_mse:
            best_mse = float(validation_mse)
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in effect.state_dict().items()
            }
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(
                f"REFINE epoch={epoch:03d}/{args.epochs} "
                f"train_mse={total_loss / batches:.6f} "
                f"val_rmse={validation_rmse:.4f} "
                f"val_corr={validation_correlation:.5f}",
                flush=True,
            )

    if best_state is None:
        raise RuntimeError("refinement produced no checkpoint")
    effect.load_state_dict(best_state)
    checkpoint = {
        "format": "task_effect_value_v1",
        "config": source_checkpoint["config"],
        "state_dict": effect.state_dict(),
        "source_model": str(Path(args.sac_model).resolve()),
        "seed": args.seed,
        "refinement": {
            "input_model": str(Path(args.input_model).resolve()),
            "query_count": len(query_observations),
            "replay_count": len(replay_observations),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    metrics = {
        "query_count": len(query_observations),
        "replay_count": len(replay_observations),
        "training_count": len(training_indices),
        "validation_count": validation_count,
        "validation_rmse": float(np.sqrt(best_mse)),
        "collection_episodes": collection_records,
        "seconds": perf_counter() - start,
        "checkpoint": str(output.resolve()),
    }
    output.with_suffix(".json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        f"DONE query_count={metrics['query_count']} "
        f"val_rmse={metrics['validation_rmse']:.4f} "
        f"seconds={metrics['seconds']:.1f} "
        f"checkpoint={output.resolve()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
