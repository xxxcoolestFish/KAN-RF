"""Iterative planner-in-loop distillation of a control-equivalent task effect."""

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

from kanrf.effect_interface import (
    TaskEffectValue,
    controllable_gradient_loss,
    control_equivalence_loss,
    effect_covariance_loss,
)
from kanrf.pusher_oracle import PusherOracleCEM, object_goal_distance
from scripts.train_pusher_effect_value import (
    collect_observations,
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


def make_planner(args: argparse.Namespace, seed: int) -> PusherOracleCEM:
    return PusherOracleCEM(
        horizon=args.horizon,
        action_repeat=args.action_repeat,
        population=args.population,
        elite_fraction=args.elite_fraction,
        iterations=args.iterations,
        discount=args.discount,
        initial_std_scale=args.initial_std_scale,
        temporal_correlation=args.temporal_correlation,
        seed=seed,
    )


def collect_groups(
    effect: TaskEffectValue,
    teacher,
    round_index: int,
    args: argparse.Namespace,
) -> tuple[dict[str, np.ndarray], list[dict]]:
    rng = np.random.default_rng(args.seed + 10007 * round_index)
    terminal_groups = []
    reward_groups = []
    discount_groups = []
    score_groups = []
    records = []

    def teacher_value(states: np.ndarray) -> np.ndarray:
        return critic_values(teacher, states, batch_size=4096)

    def effect_value(states: np.ndarray) -> np.ndarray:
        tensor = torch.as_tensor(states, dtype=torch.float32)
        with torch.no_grad():
            _, values = effect(tensor)
        return values.numpy()

    for episode in range(args.episodes_per_round):
        env = gym.make(
            "Pusher-v5",
            max_episode_steps=args.steps_per_episode,
        )
        episode_seed = (
            args.seed + 1009 * round_index + 101 * episode
        )
        observation, _ = env.reset(seed=episode_seed)
        student_planner = make_planner(
            args,
            args.seed + 20011 * round_index + episode,
        )
        exact_planner = make_planner(
            args,
            args.seed + 30011 * round_index + episode,
        )
        judge = make_planner(
            args,
            args.seed + 40009 * round_index + episode,
        )
        initial_distance = object_goal_distance(env)
        total_reward = 0.0
        step_regrets = []
        for step in range(args.steps_per_episode):
            student_plan = student_planner.plan(
                env,
                terminal_value_fn=effect_value,
            )
            exact_plan = exact_planner.plan(
                env,
                terminal_value_fn=teacher_value,
            )
            candidates = np.empty(
                (
                    args.candidates_per_group,
                    args.horizon,
                    student_planner.action_dim,
                ),
                dtype=np.float64,
            )
            split = args.candidates_per_group // 2
            student_noise = rng.normal(
                0.0,
                args.query_noise_scale
                * (student_planner.high - student_planner.low),
                size=(split, args.horizon, student_planner.action_dim),
            )
            exact_noise = rng.normal(
                0.0,
                args.query_noise_scale
                * (student_planner.high - student_planner.low),
                size=(
                    args.candidates_per_group - split,
                    args.horizon,
                    student_planner.action_dim,
                ),
            )
            candidates[:split] = student_plan.sequence + student_noise
            candidates[split:] = exact_plan.sequence + exact_noise
            candidates = np.clip(
                candidates,
                student_planner.low,
                student_planner.high,
            )
            candidates[0] = student_plan.sequence
            candidates[1] = exact_plan.sequence
            if args.candidates_per_group > 2:
                candidates[2] = 0.0

            base = env.unwrapped
            qpos = np.asarray(base.data.qpos, dtype=np.float64).copy()
            qvel = np.asarray(base.data.qvel, dtype=np.float64).copy()
            rewards, terminals, discounts = judge.rollout_sequences(
                qpos,
                qvel,
                candidates,
            )
            terminal_values = teacher_value(terminals)
            exact_scores = rewards + discounts * terminal_values
            student_index = 0
            exact_index = 1
            regret = max(
                float(exact_scores[exact_index] - exact_scores[student_index]),
                0.0,
            )
            step_regrets.append(regret)
            terminal_groups.append(terminals)
            reward_groups.append(rewards.astype(np.float32))
            discount_groups.append(discounts.astype(np.float32))
            score_groups.append(exact_scores.astype(np.float32))

            observation, reward, terminated, truncated, _ = env.step(
                student_plan.action
            )
            total_reward += float(reward)
            if args.debug_every > 0 and (
                step % args.debug_every == 0 or terminated or truncated
            ):
                print(
                    f"COLLECT round={round_index} ep={episode} "
                    f"step={step:03d} groups={len(terminal_groups)} "
                    f"regret={regret:.4f} "
                    f"distance={object_goal_distance(env):.4f}",
                    flush=True,
                )
            if terminated or truncated:
                break
        final_distance = object_goal_distance(env)
        records.append(
            {
                "round": round_index,
                "episode": episode,
                "seed": episode_seed,
                "return": total_reward,
                "initial_distance": initial_distance,
                "final_distance": final_distance,
                "progress": initial_distance - final_distance,
                "mean_exact_vs_student_regret": float(
                    np.mean(step_regrets)
                ),
            }
        )
        student_planner.close()
        exact_planner.close()
        judge.close()
        env.close()
    return (
        {
            "terminals": np.stack(terminal_groups).astype(np.float32),
            "rewards": np.stack(reward_groups).astype(np.float32),
            "discounts": np.stack(discount_groups).astype(np.float32),
            "exact_scores": np.stack(score_groups).astype(np.float32),
        },
        records,
    )


def train_round(
    effect: TaskEffectValue,
    teacher,
    cumulative_groups: dict[str, np.ndarray],
    replay_observations: np.ndarray,
    replay_values: np.ndarray,
    round_index: int,
    args: argparse.Namespace,
) -> tuple[TaskEffectValue, dict]:
    device = torch.device(args.device)
    effect = effect.to(device)
    terminals = torch.as_tensor(
        cumulative_groups["terminals"],
        device=device,
    )
    rewards = torch.as_tensor(
        cumulative_groups["rewards"],
        device=device,
    )
    discounts = torch.as_tensor(
        cumulative_groups["discounts"],
        device=device,
    )
    exact_scores = torch.as_tensor(
        cumulative_groups["exact_scores"],
        device=device,
    )
    exact_terminal_values = (exact_scores - rewards) / discounts.clamp_min(
        1e-6
    )
    replay_states = torch.as_tensor(replay_observations, device=device)
    replay_targets = torch.as_tensor(replay_values, device=device)
    optimizer = torch.optim.AdamW(
        effect.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    best_loss = float("inf")
    best_state = None
    group_count = terminals.shape[0]
    validation_count = max(1, int(group_count * args.validation_fraction))
    split_rng = np.random.default_rng(args.seed + 70001 + round_index)
    group_order = split_rng.permutation(group_count)
    validation_indices = torch.as_tensor(
        group_order[:validation_count],
        device=device,
    )
    training_indices = torch.as_tensor(
        group_order[validation_count:],
        device=device,
    )
    last_stats = {}

    for epoch in range(1, args.epochs_per_round + 1):
        effect.train()
        permutation = training_indices[
            torch.randperm(len(training_indices), device=device)
        ]
        epoch_value = 0.0
        epoch_advantage = 0.0
        epoch_margin = 0.0
        epoch_gradient = 0.0
        epoch_replay = 0.0
        epoch_agreement = 0.0
        batches = 0
        for start in range(0, len(permutation), args.group_batch_size):
            indices = permutation[start : start + args.group_batch_size]
            group_terminals = terminals[indices]
            flat_terminals = group_terminals.flatten(0, 1)
            effects, predicted_values = effect(flat_terminals)
            predicted_values = predicted_values.reshape(
                group_terminals.shape[:2]
            )
            predicted_scores = (
                rewards[indices] + discounts[indices] * predicted_values
            )
            normalized_predicted_values = (
                predicted_values - effect.value_mean
            ) / effect.value_std
            normalized_exact_values = (
                exact_terminal_values[indices] - effect.value_mean
            ) / effect.value_std
            terminal_value_loss = F.mse_loss(
                normalized_predicted_values,
                normalized_exact_values,
            )
            equivalence = control_equivalence_loss(
                predicted_scores,
                exact_scores[indices],
                minimum_margin=args.minimum_margin,
                maximum_margin=args.maximum_margin,
            )
            if args.gradient_weight > 0.0:
                gradient_count = min(
                    args.gradient_candidates,
                    group_terminals.shape[1],
                )
                gradient_states = (
                    group_terminals[:, :gradient_count]
                    .detach()
                    .clone()
                    .requires_grad_(True)
                )
                _, student_gradient_values = effect(
                    gradient_states.flatten(0, 1)
                )
                student_gradients = torch.autograd.grad(
                    student_gradient_values.sum(),
                    gradient_states,
                    create_graph=True,
                )[0]
                teacher_states = (
                    gradient_states.detach()
                    .clone()
                    .requires_grad_(True)
                )
                teacher_flat = teacher_states.flatten(0, 1)
                teacher_actions = teacher.actor(
                    teacher_flat,
                    deterministic=True,
                )
                teacher_q_values = teacher.critic(
                    teacher_flat,
                    teacher_actions,
                )
                teacher_values = torch.minimum(
                    teacher_q_values[0],
                    teacher_q_values[1],
                )
                teacher_gradients = torch.autograd.grad(
                    teacher_values.sum(),
                    teacher_states,
                    create_graph=False,
                )[0].detach()
                reachable_displacements = (
                    gradient_states.detach()
                    - group_terminals.mean(dim=1, keepdim=True)
                )
                gradient_loss = controllable_gradient_loss(
                    student_gradients,
                    teacher_gradients,
                    reachable_displacements,
                )
            else:
                gradient_loss = terminal_value_loss.new_zeros(())
            replay_indices = torch.randint(
                0,
                len(replay_states),
                (min(args.replay_batch_size, len(replay_states)),),
                device=device,
            )
            replay_effects, replay_predictions = effect(
                replay_states[replay_indices]
            )
            replay_loss = F.mse_loss(
                (replay_predictions - effect.value_mean) / effect.value_std,
                (replay_targets[replay_indices] - effect.value_mean)
                / effect.value_std,
            )
            covariance_loss = effect_covariance_loss(
                torch.cat((effects, replay_effects), dim=0)
            )
            loss = (
                args.terminal_value_weight * terminal_value_loss
                + args.advantage_weight * equivalence.advantage_loss
                + args.margin_weight * equivalence.margin_loss
                + args.gradient_weight * gradient_loss
                + args.replay_weight * replay_loss
                + args.covariance_weight * covariance_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(effect.parameters(), 10.0)
            optimizer.step()
            epoch_value += float(terminal_value_loss.detach())
            epoch_advantage += float(equivalence.advantage_loss.detach())
            epoch_margin += float(equivalence.margin_loss.detach())
            epoch_gradient += float(gradient_loss.detach())
            epoch_replay += float(replay_loss.detach())
            epoch_agreement += float(equivalence.top1_agreement.detach())
            batches += 1

        effect.eval()
        with torch.no_grad():
            validation_terminals = terminals[validation_indices]
            _, validation_values = effect(
                validation_terminals.flatten(0, 1)
            )
            validation_values = validation_values.reshape(
                validation_terminals.shape[:2]
            )
            validation_scores = (
                rewards[validation_indices]
                + discounts[validation_indices] * validation_values
            )
            validation_equivalence = control_equivalence_loss(
                validation_scores,
                exact_scores[validation_indices],
                minimum_margin=args.minimum_margin,
                maximum_margin=args.maximum_margin,
            )
            validation_value_loss = F.mse_loss(
                (
                    validation_values - effect.value_mean
                )
                / effect.value_std,
                (
                    exact_terminal_values[validation_indices]
                    - effect.value_mean
                )
                / effect.value_std,
            )
            validation_objective = (
                validation_value_loss
                + validation_equivalence.advantage_loss
                + validation_equivalence.margin_loss
            )
        if float(validation_objective) < best_loss:
            best_loss = float(validation_objective)
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in effect.state_dict().items()
            }
        if epoch == 1 or epoch % 3 == 0 or epoch == args.epochs_per_round:
            print(
                f"TRAIN round={round_index} epoch={epoch:02d}/"
                f"{args.epochs_per_round} "
                f"value={epoch_value / batches:.4f} "
                f"adv={epoch_advantage / batches:.4f} "
                f"margin={epoch_margin / batches:.4f} "
                f"gradient={epoch_gradient / batches:.4f} "
                f"replay={epoch_replay / batches:.4f} "
                f"agree={epoch_agreement / batches:.3f} "
                f"val_obj={float(validation_objective):.4f} "
                f"val_agree={float(validation_equivalence.top1_agreement):.3f}",
                flush=True,
            )
        last_stats = {
            "train_value_loss": epoch_value / batches,
            "train_advantage_loss": epoch_advantage / batches,
            "train_margin_loss": epoch_margin / batches,
            "train_gradient_loss": epoch_gradient / batches,
            "train_replay_loss": epoch_replay / batches,
            "train_top1_agreement": epoch_agreement / batches,
            "validation_objective": float(validation_objective),
            "validation_top1_agreement": float(
                validation_equivalence.top1_agreement
            ),
        }
    if best_state is None:
        raise RuntimeError("no control-equivalence checkpoint was produced")
    effect.load_state_dict(best_state)
    return effect.cpu().eval(), {
        **last_stats,
        "best_validation_objective": best_loss,
        "group_count": group_count,
    }


def concatenate_groups(
    existing: dict[str, np.ndarray] | None,
    new: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    if existing is None:
        return new
    return {
        key: np.concatenate((existing[key], new[key]), axis=0)
        for key in new
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sac-model",
        default="results/pusher-v5-SAC-expert.zip",
    )
    parser.add_argument(
        "--input-model",
        default="results/pusher_task_effect_value_d4_onpolicy.pt",
    )
    parser.add_argument(
        "--output-prefix",
        default="results/pusher_control_equivalent_d4",
    )
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--episodes-per-round", type=int, default=1)
    parser.add_argument("--steps-per-episode", type=int, default=60)
    parser.add_argument("--candidates-per-group", type=int, default=24)
    parser.add_argument("--query-noise-scale", type=float, default=0.15)
    parser.add_argument("--replay-states", type=int, default=10000)
    parser.add_argument("--epochs-per-round", type=int, default=12)
    parser.add_argument("--group-batch-size", type=int, default=8)
    parser.add_argument("--replay-batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--terminal-value-weight", type=float, default=1.0)
    parser.add_argument("--advantage-weight", type=float, default=1.0)
    parser.add_argument("--margin-weight", type=float, default=0.5)
    parser.add_argument("--gradient-weight", type=float, default=0.0)
    parser.add_argument("--gradient-candidates", type=int, default=8)
    parser.add_argument("--replay-weight", type=float, default=1.0)
    parser.add_argument("--covariance-weight", type=float, default=1e-3)
    parser.add_argument("--minimum-margin", type=float, default=0.05)
    parser.add_argument("--maximum-margin", type=float, default=1.0)
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
    if args.gradient_weight > 0.0 and args.gradient_candidates < 1:
        parser.error("--gradient-candidates must be positive when enabled")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    start = perf_counter()
    teacher = load_sac(args.sac_model, args.device)
    effect, source_checkpoint = load_effect(args.input_model)
    replay_observations, _ = collect_observations(
        teacher,
        count=args.replay_states,
        num_envs=8,
        seed=args.seed + 79,
        debug_every=max(args.replay_states // 2, 1),
    )
    replay_values = critic_values(
        teacher,
        replay_observations,
        batch_size=4096,
    )
    cumulative_groups = None
    all_collection_records = []
    round_metrics = []
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    for round_index in range(1, args.rounds + 1):
        new_groups, records = collect_groups(
            effect,
            teacher,
            round_index,
            args,
        )
        cumulative_groups = concatenate_groups(
            cumulative_groups,
            new_groups,
        )
        all_collection_records.extend(records)
        effect, metrics = train_round(
            effect,
            teacher,
            cumulative_groups,
            replay_observations,
            replay_values,
            round_index,
            args,
        )
        checkpoint = {
            "format": "task_effect_value_v1",
            "config": source_checkpoint["config"],
            "state_dict": effect.state_dict(),
            "source_model": str(Path(args.sac_model).resolve()),
            "seed": args.seed,
            "control_equivalence": {
                "round": round_index,
                "input_model": str(Path(args.input_model).resolve()),
                "group_count": metrics["group_count"],
                "gradient_weight": args.gradient_weight,
                "gradient_candidates": args.gradient_candidates,
            },
        }
        checkpoint_path = Path(
            f"{output_prefix}_round{round_index}.pt"
        )
        torch.save(checkpoint, checkpoint_path)
        round_record = {
            "round": round_index,
            "checkpoint": str(checkpoint_path.resolve()),
            "collection": records,
            **metrics,
        }
        round_metrics.append(round_record)
        print(
            f"ROUND_DONE round={round_index} "
            f"groups={metrics['group_count']} "
            f"val_obj={metrics['best_validation_objective']:.4f} "
            f"checkpoint={checkpoint_path.resolve()}",
            flush=True,
        )

    result = {
        "config": vars(args),
        "rounds": round_metrics,
        "collection_records": all_collection_records,
        "seconds": perf_counter() - start,
    }
    metrics_path = Path(f"{output_prefix}_training.json")
    metrics_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        f"DONE rounds={args.rounds} seconds={result['seconds']:.1f} "
        f"metrics={metrics_path.resolve()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
