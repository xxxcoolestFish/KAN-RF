"""Evaluate zero, random, SAC and real-simulator Oracle-CEM controllers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev

import gymnasium as gym
import numpy as np

from kanrf.pusher_oracle import PusherOracleCEM, object_goal_distance


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(mean(values)),
        "std": float(pstdev(values)) if len(values) > 1 else 0.0,
        "min": float(min(values)),
        "max": float(max(values)),
    }


def load_sac(path: str | None):
    if path is None:
        raise ValueError("--sac-model is required for the sac controller")
    from stable_baselines3 import SAC

    return SAC.load(path, device="auto")


def run_controller(args: argparse.Namespace, controller: str) -> dict:
    model = load_sac(args.sac_model) if controller == "sac" else None
    planner = (
        PusherOracleCEM(
            horizon=args.horizon,
            action_repeat=args.action_repeat,
            population=args.population,
            elite_fraction=args.elite_fraction,
            iterations=args.iterations,
            discount=args.discount,
            initial_std_scale=args.initial_std_scale,
            temporal_correlation=args.temporal_correlation,
            seed=args.seed + 9000,
        )
        if controller == "oracle"
        else None
    )
    rng = np.random.default_rng(args.seed + 7000)
    episodes = []

    print(f"\n=== Controller: {controller} ===", flush=True)
    for episode in range(args.episodes):
        seed = args.seed + episode * 101
        env = gym.make("Pusher-v5", max_episode_steps=args.max_steps)
        observation, _ = env.reset(seed=seed)
        initial_distance = object_goal_distance(env)
        total_reward = 0.0
        reward_dist = 0.0
        reward_near = 0.0
        reward_ctrl = 0.0
        action_norms: list[float] = []
        planning_times: list[float] = []
        distances = [initial_distance]

        for step in range(args.max_steps):
            if controller == "zero":
                action = np.zeros(env.action_space.shape, dtype=np.float32)
                plan_result = None
            elif controller == "random":
                action = rng.uniform(
                    env.action_space.low,
                    env.action_space.high,
                ).astype(np.float32)
                plan_result = None
            elif controller == "sac":
                action, _ = model.predict(observation, deterministic=True)
                plan_result = None
            elif controller == "oracle":
                show_cem = args.debug_every > 0 and step % args.debug_every == 0

                def debug(trace) -> None:
                    if show_cem:
                        print(
                            "  "
                            f"cem_iter={trace.iteration} "
                            f"best={trace.best_return:.3f} "
                            f"elite={trace.elite_mean:.3f} "
                            f"pop={trace.population_mean:.3f}"
                            f"+/-{trace.population_std:.3f} "
                            f"action_std={trace.action_std_mean:.3f}",
                            flush=True,
                        )

                plan_result = planner.plan(env, debug_callback=debug)
                action = plan_result.action
                planning_times.append(plan_result.planning_ms)
            else:
                raise ValueError(f"unknown controller {controller}")

            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            reward_dist += float(info.get("reward_dist", 0.0))
            reward_near += float(info.get("reward_near", 0.0))
            reward_ctrl += float(info.get("reward_ctrl", 0.0))
            action_norms.append(float(np.linalg.norm(action)))
            current_distance = object_goal_distance(env)
            distances.append(current_distance)

            if args.debug_every > 0 and (
                step % args.debug_every == 0 or terminated or truncated
            ):
                plan_text = (
                    f" plan_ms={plan_result.planning_ms:.1f}"
                    f" pred_return={plan_result.predicted_return:.3f}"
                    if plan_result is not None
                    else ""
                )
                print(
                    f"ep={episode} step={step:03d} "
                    f"reward={reward:+.4f} total={total_reward:+.3f} "
                    f"distance={current_distance:.4f} "
                    f"action_norm={action_norms[-1]:.3f}{plan_text}",
                    flush=True,
                )
            if terminated or truncated:
                break

        final_distance = distances[-1]
        record = {
            "episode": episode,
            "seed": seed,
            "return": total_reward,
            "steps": len(action_norms),
            "initial_distance": initial_distance,
            "final_distance": final_distance,
            "minimum_distance": float(min(distances)),
            "distance_progress": initial_distance - final_distance,
            "success_final_005": bool(final_distance < 0.05),
            "success_any_005": bool(min(distances) < 0.05),
            "mean_action_norm": float(mean(action_norms)),
            "mean_planning_ms": (
                float(mean(planning_times)) if planning_times else 0.0
            ),
            "reward_dist": reward_dist,
            "reward_near": reward_near,
            "reward_ctrl": reward_ctrl,
        }
        episodes.append(record)
        env.close()
        print(
            f"EPISODE_DONE controller={controller} ep={episode} "
            f"return={total_reward:.3f} progress={record['distance_progress']:.4f} "
            f"final_distance={final_distance:.4f} "
            f"min_distance={record['minimum_distance']:.4f}",
            flush=True,
        )

    if planner is not None:
        planner.close()

    result = {
        "controller": controller,
        "config": {
            "episodes": args.episodes,
            "max_steps": args.max_steps,
            "horizon": args.horizon,
            "action_repeat": args.action_repeat,
            "population": args.population,
            "elite_fraction": args.elite_fraction,
            "iterations": args.iterations,
            "discount": args.discount,
            "initial_std_scale": args.initial_std_scale,
            "temporal_correlation": args.temporal_correlation,
            "seed": args.seed,
        },
        "return": summarize([record["return"] for record in episodes]),
        "final_distance": summarize(
            [record["final_distance"] for record in episodes]
        ),
        "distance_progress": summarize(
            [record["distance_progress"] for record in episodes]
        ),
        "success_final_005": float(
            mean(float(record["success_final_005"]) for record in episodes)
        ),
        "success_any_005": float(
            mean(float(record["success_any_005"]) for record in episodes)
        ),
        "episodes": episodes,
    }
    print(
        f"SUMMARY controller={controller} "
        f"return={result['return']['mean']:.3f}"
        f"+/-{result['return']['std']:.3f} "
        f"progress={result['distance_progress']['mean']:.4f} "
        f"final_distance={result['final_distance']['mean']:.4f}",
        flush=True,
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--controllers",
        nargs="+",
        choices=("zero", "random", "sac", "oracle"),
        default=("zero", "random", "oracle"),
    )
    parser.add_argument("--sac-model")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1811)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--action-repeat", type=int, default=2)
    parser.add_argument("--population", type=int, default=128)
    parser.add_argument("--elite-fraction", type=float, default=0.1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--initial-std-scale", type=float, default=0.35)
    parser.add_argument("--temporal-correlation", type=float, default=0.7)
    parser.add_argument("--debug-every", type=int, default=10)
    parser.add_argument(
        "--json-out",
        default="results/pusher_gate_a_baselines.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = {
        controller: run_controller(args, controller)
        for controller in args.controllers
    }
    output = Path(args.json_out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nWROTE {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
