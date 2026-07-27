"""Paired exact-vs-effect CEM diagnosis under identical initial candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev

import gymnasium as gym
import numpy as np
import torch

from kanrf.effect_interface import TaskEffectValue
from kanrf.pusher_oracle import PusherOracleCEM, object_goal_distance


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(mean(values)),
        "std": float(pstdev(values)) if len(values) > 1 else 0.0,
        "min": float(min(values)),
        "max": float(max(values)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sac-model",
        default="results/pusher-v5-SAC-expert.zip",
    )
    parser.add_argument(
        "--effect-model",
        default="results/pusher_task_effect_value_d4.pt",
    )
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--action-repeat", type=int, default=3)
    parser.add_argument("--population", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--initial-std-scale", type=float, default=0.15)
    parser.add_argument("--temporal-correlation", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=1811)
    parser.add_argument("--debug-every", type=int, default=5)
    parser.add_argument(
        "--json-out",
        default="results/pusher_paired_planning_d4.json",
    )
    args = parser.parse_args()

    from stable_baselines3 import SAC

    teacher = SAC.load(args.sac_model, device="auto")
    checkpoint = torch.load(
        args.effect_model,
        map_location="cpu",
        weights_only=False,
    )
    effect = TaskEffectValue(**checkpoint["config"])
    effect.load_state_dict(checkpoint["state_dict"])
    effect.eval()

    def teacher_value(states: np.ndarray) -> np.ndarray:
        tensor, _ = teacher.policy.obs_to_tensor(states)
        with torch.no_grad():
            actions = teacher.actor(tensor, deterministic=True)
            q_values = teacher.critic(tensor, actions)
            values = torch.minimum(q_values[0], q_values[1]).squeeze(-1)
        return values.cpu().numpy()

    def effect_value(states: np.ndarray) -> np.ndarray:
        tensor = torch.as_tensor(states, dtype=torch.float32)
        with torch.no_grad():
            _, values = effect(tensor)
        return values.numpy()

    def make_planner(seed: int) -> PusherOracleCEM:
        return PusherOracleCEM(
            horizon=args.horizon,
            action_repeat=args.action_repeat,
            population=args.population,
            iterations=args.iterations,
            discount=args.discount,
            initial_std_scale=args.initial_std_scale,
            temporal_correlation=args.temporal_correlation,
            seed=seed,
        )

    episodes = []
    all_regrets: list[float] = []
    all_action_differences: list[float] = []
    for episode in range(args.episodes):
        env = gym.make("Pusher-v5", max_episode_steps=args.max_steps)
        observation, _ = env.reset(seed=args.seed + 101 * episode)
        initial_distance = object_goal_distance(env)
        total_reward = 0.0
        records = []
        for step in range(args.max_steps):
            paired_seed = args.seed + 100003 * episode + 997 * step
            exact_planner = make_planner(paired_seed)
            effect_planner = make_planner(paired_seed)
            judge = make_planner(paired_seed + 1)
            exact_plan = exact_planner.plan(
                env,
                terminal_value_fn=teacher_value,
            )
            effect_plan = effect_planner.plan(
                env,
                terminal_value_fn=effect_value,
            )
            base = env.unwrapped
            qpos = np.asarray(base.data.qpos, dtype=np.float64).copy()
            qvel = np.asarray(base.data.qvel, dtype=np.float64).copy()
            paired_sequences = np.stack(
                (exact_plan.sequence, effect_plan.sequence)
            )
            exact_scores = judge.evaluate_sequences(
                qpos,
                qvel,
                paired_sequences,
                terminal_value_fn=teacher_value,
            )
            raw_regret = float(exact_scores[0] - exact_scores[1])
            regret = max(raw_regret, 0.0)
            action_difference = float(
                np.linalg.norm(exact_plan.action - effect_plan.action)
            )
            selected_action = effect_plan.action
            observation, reward, terminated, truncated, _ = env.step(
                selected_action
            )
            total_reward += float(reward)
            distance = object_goal_distance(env)
            record = {
                "step": step,
                "exact_score": float(exact_scores[0]),
                "effect_sequence_exact_score": float(exact_scores[1]),
                "raw_regret": raw_regret,
                "regret": regret,
                "action_difference": action_difference,
                "exact_action_norm": float(np.linalg.norm(exact_plan.action)),
                "effect_action_norm": float(np.linalg.norm(effect_plan.action)),
                "distance": distance,
            }
            records.append(record)
            all_regrets.append(regret)
            all_action_differences.append(action_difference)
            if args.debug_every > 0 and (
                step % args.debug_every == 0 or terminated or truncated
            ):
                print(
                    f"PAIR ep={episode} step={step:03d} "
                    f"regret={regret:.4f} raw={raw_regret:+.4f} "
                    f"action_diff={action_difference:.3f} "
                    f"exact_norm={record['exact_action_norm']:.3f} "
                    f"effect_norm={record['effect_action_norm']:.3f} "
                    f"distance={distance:.4f}",
                    flush=True,
                )
            exact_planner.close()
            effect_planner.close()
            judge.close()
            if terminated or truncated:
                break
        episodes.append(
            {
                "episode": episode,
                "return": total_reward,
                "initial_distance": initial_distance,
                "final_distance": records[-1]["distance"],
                "progress": initial_distance - records[-1]["distance"],
                "mean_regret": float(mean(r["regret"] for r in records)),
                "mean_action_difference": float(
                    mean(r["action_difference"] for r in records)
                ),
                "records": records,
            }
        )
        env.close()

    result = {
        "config": vars(args),
        "regret": summarize(all_regrets),
        "action_difference": summarize(all_action_differences),
        "episodes": episodes,
    }
    output = Path(args.json_out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        f"SUMMARY regret={result['regret']['mean']:.4f}"
        f"+/-{result['regret']['std']:.4f} "
        f"action_diff={result['action_difference']['mean']:.4f} "
        f"final_distance={mean(e['final_distance'] for e in episodes):.4f}",
        flush=True,
    )
    print(f"WROTE {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
