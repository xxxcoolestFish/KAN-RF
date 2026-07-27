"""Evaluate and diagnose exact-dynamics local reachability graph routing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev

import gymnasium as gym
import numpy as np

from kanrf.pusher_graph_router import (
    PusherOracleGraphRouter,
    fingertip_object_distance,
)
from kanrf.pusher_oracle import object_goal_distance


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(mean(values)),
        "std": float(pstdev(values)) if len(values) > 1 else 0.0,
        "min": float(min(values)),
        "max": float(max(values)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--branching", type=int, default=24)
    parser.add_argument("--beam-width", type=int, default=48)
    parser.add_argument("--action-scale", type=float, default=0.5)
    parser.add_argument("--noise-scale", type=float, default=0.25)
    parser.add_argument("--merge-radius", type=float, default=0.35)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--heuristic-steps", type=int, default=100)
    parser.add_argument(
        "--action-strategy",
        choices=("primitive", "sensitivity"),
        default="primitive",
    )
    parser.add_argument("--sensitivity-probe", type=float, default=0.5)
    parser.add_argument("--sensitivity-steps", type=int, default=3)
    parser.add_argument("--sac-model")
    parser.add_argument("--seed", type=int, default=1811)
    parser.add_argument("--debug-every", type=int, default=10)
    parser.add_argument(
        "--json-out",
        default="results/pusher_graph_router_exact.json",
    )
    args = parser.parse_args()
    model = None
    if args.sac_model is not None:
        from stable_baselines3 import SAC

        model = SAC.load(args.sac_model, device="auto")

        def policy_fn(states: np.ndarray) -> np.ndarray:
            actions, _ = model.predict(states, deterministic=True)
            return np.asarray(actions)

    else:
        policy_fn = None

    episodes = []
    for episode in range(args.episodes):
        episode_seed = args.seed + 101 * episode
        env = gym.make("Pusher-v5", max_episode_steps=args.max_steps)
        observation, _ = env.reset(seed=episode_seed)
        router = PusherOracleGraphRouter(
            depth=args.depth,
            branching=args.branching,
            beam_width=args.beam_width,
            action_scale=args.action_scale,
            noise_scale=args.noise_scale,
            merge_radius=args.merge_radius,
            discount=args.discount,
            heuristic_steps=args.heuristic_steps,
            action_strategy=args.action_strategy,
            sensitivity_probe=args.sensitivity_probe,
            sensitivity_steps=args.sensitivity_steps,
            seed=args.seed + 9000 + episode,
        )
        initial_distance = object_goal_distance(env)
        distances = [initial_distance]
        contact_distances = [fingertip_object_distance(env)]
        total_reward = 0.0
        planning_times = []
        observation_errors = []
        reward_errors = []
        distance_errors = []
        merge_rates = []
        best_layer_distances = []
        proposal_distances = []
        proposal_selections = []

        for step in range(args.max_steps):
            show_debug = (
                args.debug_every > 0 and step % args.debug_every == 0
            )

            def debug(trace) -> None:
                if show_debug:
                    print(
                        "  "
                        f"layer={trace.depth} parents={trace.parents} "
                        f"expanded={trace.expanded} unique={trace.unique} "
                        f"merged={trace.merged} kept={trace.kept} "
                        f"best_score={trace.best_score:+.3f} "
                        f"route_dist={trace.route_distance:.4f} "
                        f"min_dist={trace.minimum_distance:.4f} "
                        f"route_contact={trace.route_contact_distance:.4f} "
                        f"min_contact={trace.minimum_contact_distance:.4f} "
                        f"state_reward={trace.route_state_reward:+.4f}",
                        flush=True,
                    )

            plan = router.plan(
                env,
                debug_callback=debug,
                policy_fn=policy_fn,
            )
            if show_debug and plan.sensitivity is not None:
                singular_values = np.asarray(
                    plan.sensitivity.singular_values
                )
                print(
                    "  "
                    f"sensitivity_rank={plan.sensitivity.rank} "
                    f"singular_values="
                    f"{np.array2string(singular_values, precision=3)} "
                    f"task_grad_norm="
                    f"{plan.sensitivity.task_gradient_norm:.3e}",
                    flush=True,
                )
            observation, reward, terminated, truncated, _ = env.step(
                plan.action
            )
            distance = object_goal_distance(env)
            contact_distance = fingertip_object_distance(env)
            observation_error = float(
                np.linalg.norm(
                    observation - plan.predicted_first_observation
                )
            )
            reward_error = abs(
                float(reward) - plan.predicted_first_reward
            )
            distance_error = abs(
                distance - plan.predicted_first_distance
            )
            expanded = sum(layer.expanded for layer in plan.layers)
            merged = sum(layer.merged for layer in plan.layers)
            merge_rate = merged / max(expanded, 1)
            total_reward += float(reward)
            distances.append(distance)
            contact_distances.append(contact_distance)
            planning_times.append(plan.planning_ms)
            observation_errors.append(observation_error)
            reward_errors.append(reward_error)
            distance_errors.append(distance_error)
            merge_rates.append(merge_rate)
            best_layer_distances.append(
                plan.layers[-1].route_distance
            )
            if plan.proposal_action_distance is not None:
                proposal_distances.append(plan.proposal_action_distance)
                proposal_selections.append(
                    float(plan.proposal_action_distance < 1e-6)
                )

            if show_debug or terminated or truncated:
                print(
                    f"STEP ep={episode} step={step:03d} "
                    f"reward={float(reward):+.4f} "
                    f"total={total_reward:+.3f} distance={distance:.4f} "
                    f"contact={contact_distance:.4f} "
                    f"route_end_dist={plan.layers[-1].route_distance:.4f} "
                    f"route_contact="
                    f"{plan.layers[-1].route_contact_distance:.4f} "
                    f"action_norm={np.linalg.norm(plan.action):.3f} "
                    f"merge_rate={merge_rate:.3f} "
                    f"obs_error={observation_error:.2e} "
                    f"reward_error={reward_error:.2e} "
                    f"distance_error={distance_error:.2e} "
                    f"proposal_dist="
                    f"{plan.proposal_action_distance if plan.proposal_action_distance is not None else float('nan'):.3f} "
                    f"plan_ms={plan.planning_ms:.1f}",
                    flush=True,
                )
                print(
                    "  first_action="
                    + np.array2string(
                        plan.action,
                        precision=3,
                        suppress_small=True,
                    ),
                    flush=True,
                )
            if terminated or truncated:
                break

        final_distance = distances[-1]
        record = {
            "episode": episode,
            "seed": episode_seed,
            "return": total_reward,
            "steps": len(planning_times),
            "initial_distance": initial_distance,
            "final_distance": final_distance,
            "minimum_distance": float(min(distances)),
            "final_contact_distance": contact_distances[-1],
            "minimum_contact_distance": float(min(contact_distances)),
            "distance_progress": initial_distance - final_distance,
            "success_final_005": bool(final_distance < 0.05),
            "success_any_005": bool(min(distances) < 0.05),
            "mean_planning_ms": float(mean(planning_times)),
            "mean_observation_error": float(mean(observation_errors)),
            "mean_reward_error": float(mean(reward_errors)),
            "mean_distance_error": float(mean(distance_errors)),
            "mean_merge_rate": float(mean(merge_rates)),
            "mean_route_end_distance": float(mean(best_layer_distances)),
            "mean_proposal_action_distance": (
                float(mean(proposal_distances))
                if proposal_distances
                else None
            ),
            "proposal_selection_rate": (
                float(mean(proposal_selections))
                if proposal_selections
                else None
            ),
        }
        episodes.append(record)
        router.close()
        env.close()
        print(
            f"EPISODE_DONE ep={episode} return={total_reward:.3f} "
            f"progress={record['distance_progress']:.4f} "
            f"final_distance={final_distance:.4f} "
            f"min_distance={record['minimum_distance']:.4f} "
            f"mean_plan_ms={record['mean_planning_ms']:.1f}",
            flush=True,
        )

    result = {
        "config": vars(args),
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
        "mean_planning_ms": summarize(
            [record["mean_planning_ms"] for record in episodes]
        ),
        "mean_merge_rate": summarize(
            [record["mean_merge_rate"] for record in episodes]
        ),
        "episodes": episodes,
    }
    output = Path(args.json_out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        f"SUMMARY return={result['return']['mean']:.3f}"
        f"+/-{result['return']['std']:.3f} "
        f"progress={result['distance_progress']['mean']:.4f} "
        f"final_distance={result['final_distance']['mean']:.4f} "
        f"success={result['success_final_005']:.3f} "
        f"plan_ms={result['mean_planning_ms']['mean']:.1f}",
        flush=True,
    )
    print(f"WROTE {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
