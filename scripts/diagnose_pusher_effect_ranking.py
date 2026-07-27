"""Measure whether a distilled effect preserves local action-sequence rankings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev

import gymnasium as gym
import numpy as np
import torch

from kanrf.effect_interface import TaskEffectValue
from kanrf.pusher_oracle import PusherOracleCEM


def ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    result = np.empty_like(order, dtype=np.float64)
    result[order] = np.arange(len(values), dtype=np.float64)
    return result


def correlation(first: np.ndarray, second: np.ndarray) -> float:
    first = first - first.mean()
    second = second - second.mean()
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    return float(first @ second / max(float(denominator), 1e-12))


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(mean(values)),
        "std": float(pstdev(values)) if len(values) > 1 else 0.0,
        "min": float(min(values)),
        "max": float(max(values)),
    }


def load_models(args: argparse.Namespace):
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
    return teacher, effect


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
    parser.add_argument("--anchors", type=int, default=30)
    parser.add_argument("--anchor-stride", type=int, default=3)
    parser.add_argument("--candidates", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--action-repeat", type=int, default=3)
    parser.add_argument("--noise-scale", type=float, default=0.2)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1811)
    parser.add_argument(
        "--anchor-policy",
        choices=("expert", "noisy_expert", "random", "effect"),
        default="expert",
    )
    parser.add_argument(
        "--candidate-center",
        choices=("expert", "zero"),
        default="expert",
    )
    parser.add_argument("--debug-every", type=int, default=5)
    parser.add_argument(
        "--json-out",
        default="results/pusher_effect_ranking_diagnostic.json",
    )
    args = parser.parse_args()
    teacher, effect = load_models(args)
    env = gym.make("Pusher-v5", max_episode_steps=100)
    observation, _ = env.reset(seed=args.seed)
    planner = PusherOracleCEM(
        horizon=args.horizon,
        action_repeat=args.action_repeat,
        population=args.candidates,
        iterations=1,
        discount=args.discount,
        seed=args.seed + 8000,
    )
    rng = np.random.default_rng(args.seed + 9000)
    records = []

    def policy_fn(states):
        actions, _ = teacher.predict(states, deterministic=True)
        return actions

    def teacher_value(states):
        tensor, _ = teacher.policy.obs_to_tensor(states)
        with torch.no_grad():
            actions = teacher.actor(tensor, deterministic=True)
            q_values = teacher.critic(tensor, actions)
            values = torch.minimum(q_values[0], q_values[1]).squeeze(-1)
        return values.cpu().numpy()

    def effect_value(states):
        tensor = torch.as_tensor(states, dtype=torch.float32)
        with torch.no_grad():
            _, values = effect(tensor)
        return values.numpy()

    step = 0
    while len(records) < args.anchors:
        if step % args.anchor_stride == 0:
            base = env.unwrapped
            qpos = np.asarray(base.data.qpos, dtype=np.float64).copy()
            qvel = np.asarray(base.data.qvel, dtype=np.float64).copy()
            proposal = planner.policy_proposal(env, policy_fn)
            center = (
                proposal
                if args.candidate_center == "expert"
                else np.zeros_like(proposal)
            )
            noise = rng.normal(
                0.0,
                args.noise_scale * (planner.high - planner.low),
                size=(
                    args.candidates,
                    args.horizon,
                    planner.action_dim,
                ),
            )
            candidates = np.clip(
                center[None, ...] + noise,
                planner.low,
                planner.high,
            )
            candidates[0] = center
            candidates[1] = 0.0
            rewards, terminals, discounts = planner.rollout_sequences(
                qpos,
                qvel,
                candidates,
            )
            exact_values = teacher_value(terminals)
            predicted_values = effect_value(terminals)
            exact_scores = rewards + discounts * exact_values
            predicted_scores = rewards + discounts * predicted_values
            exact_best = int(np.argmax(exact_scores))
            predicted_best = int(np.argmax(predicted_scores))
            exact_top = set(np.argsort(exact_scores)[-args.top_k :].tolist())
            predicted_top = set(
                np.argsort(predicted_scores)[-args.top_k :].tolist()
            )
            record = {
                "anchor": len(records),
                "step": step,
                "value_pearson": correlation(exact_values, predicted_values),
                "score_spearman": correlation(
                    ranks(exact_scores),
                    ranks(predicted_scores),
                ),
                "top_k_overlap": len(exact_top & predicted_top) / args.top_k,
                "exact_regret": float(
                    exact_scores[exact_best] - exact_scores[predicted_best]
                ),
                "predicted_best_exact_rank": float(
                    ranks(exact_scores)[predicted_best]
                    / max(args.candidates - 1, 1)
                ),
                "exact_margin": float(
                    np.max(exact_scores)
                    - np.partition(exact_scores, -2)[-2]
                ),
                "value_rmse": float(
                    np.sqrt(np.mean((exact_values - predicted_values) ** 2))
                ),
            }
            records.append(record)
            if (
                args.debug_every > 0
                and len(records) % args.debug_every == 0
            ):
                print(
                    f"RANK anchor={len(records):03d}/{args.anchors} "
                    f"pearson={record['value_pearson']:.3f} "
                    f"spearman={record['score_spearman']:.3f} "
                    f"topk={record['top_k_overlap']:.3f} "
                    f"regret={record['exact_regret']:.3f} "
                    f"best_rank={record['predicted_best_exact_rank']:.3f}",
                    flush=True,
                )

        if args.anchor_policy == "expert":
            action, _ = teacher.predict(observation, deterministic=True)
        elif args.anchor_policy == "noisy_expert":
            action, _ = teacher.predict(observation, deterministic=True)
            action = np.clip(
                action
                + rng.normal(
                    0.0,
                    0.5,
                    size=env.action_space.shape,
                ),
                env.action_space.low,
                env.action_space.high,
            )
        elif args.anchor_policy == "random":
            action = rng.uniform(
                env.action_space.low,
                env.action_space.high,
            )
        elif args.anchor_policy == "effect":
            plan = planner.plan(
                env,
                terminal_value_fn=effect_value,
            )
            action = plan.action
        else:
            raise ValueError(args.anchor_policy)
        observation, _, terminated, truncated, _ = env.step(action)
        step += 1
        if terminated or truncated:
            observation, _ = env.reset(seed=args.seed + 101 * step)

    metrics = {
        "config": vars(args),
        "value_pearson": summarize([r["value_pearson"] for r in records]),
        "score_spearman": summarize([r["score_spearman"] for r in records]),
        "top_k_overlap": summarize([r["top_k_overlap"] for r in records]),
        "exact_regret": summarize([r["exact_regret"] for r in records]),
        "predicted_best_exact_rank": summarize(
            [r["predicted_best_exact_rank"] for r in records]
        ),
        "exact_margin": summarize([r["exact_margin"] for r in records]),
        "value_rmse": summarize([r["value_rmse"] for r in records]),
        "records": records,
    }
    output = Path(args.json_out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        "SUMMARY "
        f"value_pearson={metrics['value_pearson']['mean']:.4f} "
        f"score_spearman={metrics['score_spearman']['mean']:.4f} "
        f"top_k={metrics['top_k_overlap']['mean']:.4f} "
        f"regret={metrics['exact_regret']['mean']:.4f} "
        f"best_rank={metrics['predicted_best_exact_rank']['mean']:.4f}",
        flush=True,
    )
    print(f"WROTE {output.resolve()}", flush=True)
    planner.close()
    env.close()


if __name__ == "__main__":
    main()
