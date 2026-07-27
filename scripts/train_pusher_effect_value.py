"""Distill a source SAC critic into an automatic low-dimensional task effect."""

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
from kanrf.pusher_oracle import PusherOracleCEM


def load_sac(path: str, device: str):
    from stable_baselines3 import SAC

    return SAC.load(path, device=device)


def critic_values(model, observations: np.ndarray, batch_size: int) -> np.ndarray:
    values = []
    for start in range(0, len(observations), batch_size):
        batch = observations[start : start + batch_size]
        obs_tensor, _ = model.policy.obs_to_tensor(batch)
        with torch.no_grad():
            actions = model.actor(obs_tensor, deterministic=True)
            q_values = model.critic(obs_tensor, actions)
            value = torch.minimum(q_values[0], q_values[1]).squeeze(-1)
        values.append(value.cpu().numpy())
    return np.concatenate(values).astype(np.float32)


def collect_observations(
    model,
    count: int,
    num_envs: int,
    seed: int,
    debug_every: int,
) -> tuple[np.ndarray, dict[str, int]]:
    """Collect source-task coverage without selecting state coordinates."""
    envs = [
        gym.make("Pusher-v5", max_episode_steps=100)
        for _ in range(num_envs)
    ]
    observations: list[np.ndarray] = []
    current = []
    episodes = np.zeros(num_envs, dtype=np.int64)
    rng = np.random.default_rng(seed + 991)
    modes = ("expert", "mild_noise", "strong_noise", "random")
    mode_counts = {mode: 0 for mode in modes}

    for env_index, env in enumerate(envs):
        observation, _ = env.reset(seed=seed + 1009 * env_index)
        current.append(observation)
    current_batch = np.asarray(current, dtype=np.float32)
    start_time = perf_counter()

    while len(observations) < count:
        expert_actions, _ = model.predict(current_batch, deterministic=True)
        actions = np.asarray(expert_actions, dtype=np.float32)
        for env_index, env in enumerate(envs):
            mode = modes[(env_index + int(episodes[env_index])) % len(modes)]
            if mode == "mild_noise":
                actions[env_index] += rng.normal(0.0, 0.35, actions.shape[1])
            elif mode == "strong_noise":
                actions[env_index] += rng.normal(0.0, 0.9, actions.shape[1])
            elif mode == "random":
                actions[env_index] = rng.uniform(
                    env.action_space.low,
                    env.action_space.high,
                )
            actions[env_index] = np.clip(
                actions[env_index],
                env.action_space.low,
                env.action_space.high,
            )
            mode_counts[mode] += 1

        for env_index, env in enumerate(envs):
            if len(observations) >= count:
                break
            observations.append(current_batch[env_index].copy())
            next_observation, _, terminated, truncated, _ = env.step(
                actions[env_index]
            )
            if terminated or truncated:
                episodes[env_index] += 1
                next_observation, _ = env.reset(
                    seed=seed
                    + 1009 * env_index
                    + 7919 * int(episodes[env_index])
                )
            current_batch[env_index] = next_observation

        if debug_every > 0 and len(observations) % debug_every < num_envs:
            elapsed = perf_counter() - start_time
            print(
                f"COLLECT states={len(observations)}/{count} "
                f"episodes={int(episodes.sum())} "
                f"rate={len(observations) / max(elapsed, 1e-9):.1f}/s",
                flush=True,
            )

    for env in envs:
        env.close()
    return np.stack(observations), mode_counts


def collect_counterfactual_groups(
    model,
    groups: int,
    candidates: int,
    horizon: int,
    action_repeat: int,
    noise_scale: float,
    discount: float,
    seed: int,
    debug_every: int,
    value_batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Collect state-local action alternatives for listwise value training."""
    if groups <= 0:
        return (
            np.empty((0, candidates, 23), dtype=np.float32),
            np.empty((0, candidates), dtype=np.float32),
            np.empty((0, candidates), dtype=np.float32),
            np.empty((0, candidates), dtype=np.float32),
        )
    env = gym.make("Pusher-v5", max_episode_steps=100)
    planner = PusherOracleCEM(
        horizon=horizon,
        action_repeat=action_repeat,
        population=candidates,
        iterations=1,
        discount=discount,
        seed=seed + 5001,
    )
    rng = np.random.default_rng(seed + 6001)
    modes = ("expert", "mild_noise", "strong_noise", "random")
    block_size = max(1, int(np.ceil(groups / len(modes))))
    observation, _ = env.reset(seed=seed + 7001)
    active_mode = None
    terminal_groups = []
    reward_groups = []
    discount_groups = []
    exact_score_groups = []
    start_time = perf_counter()

    def policy_fn(states):
        actions, _ = model.predict(states, deterministic=True)
        return actions

    for group_index in range(groups):
        mode = modes[min(group_index // block_size, len(modes) - 1)]
        if mode != active_mode:
            active_mode = mode
            observation, _ = env.reset(
                seed=seed + 7001 + 997 * group_index
            )

        proposal = planner.policy_proposal(env, policy_fn)
        centers = np.zeros(
            (candidates, horizon, planner.action_dim),
            dtype=np.float64,
        )
        centers[: candidates // 2] = proposal
        noise = rng.normal(
            0.0,
            noise_scale * (planner.high - planner.low),
            size=centers.shape,
        )
        sequences = np.clip(
            centers + noise,
            planner.low,
            planner.high,
        )
        sequences[0] = proposal
        if candidates > 1:
            sequences[1] = 0.0
        base = env.unwrapped
        qpos = np.asarray(base.data.qpos, dtype=np.float64).copy()
        qvel = np.asarray(base.data.qvel, dtype=np.float64).copy()
        rewards, terminals, terminal_discounts = planner.rollout_sequences(
            qpos,
            qvel,
            sequences,
        )
        terminal_values = critic_values(
            model,
            terminals,
            value_batch_size,
        )
        exact_scores = rewards + terminal_discounts * terminal_values
        terminal_groups.append(terminals)
        reward_groups.append(rewards.astype(np.float32))
        discount_groups.append(terminal_discounts.astype(np.float32))
        exact_score_groups.append(exact_scores.astype(np.float32))

        action, _ = model.predict(observation, deterministic=True)
        if mode == "mild_noise":
            action = action + rng.normal(0.0, 0.35, action.shape)
        elif mode == "strong_noise":
            action = action + rng.normal(0.0, 0.9, action.shape)
        elif mode == "random":
            action = rng.uniform(
                env.action_space.low,
                env.action_space.high,
            )
        action = np.clip(
            action,
            env.action_space.low,
            env.action_space.high,
        )
        observation, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            observation, _ = env.reset(
                seed=seed + 7001 + 997 * group_index + 1
            )

        if debug_every > 0 and (
            (group_index + 1) % debug_every == 0
            or group_index + 1 == groups
        ):
            elapsed = perf_counter() - start_time
            print(
                f"RANK_COLLECT groups={group_index + 1}/{groups} "
                f"mode={mode} "
                f"rate={(group_index + 1) / max(elapsed, 1e-9):.1f}/s",
                flush=True,
            )

    planner.close()
    env.close()
    return (
        np.stack(terminal_groups),
        np.stack(reward_groups),
        np.stack(discount_groups),
        np.stack(exact_score_groups),
    )


def correlation(first: torch.Tensor, second: torch.Tensor) -> float:
    first = first - first.mean()
    second = second - second.mean()
    denominator = torch.linalg.vector_norm(first) * torch.linalg.vector_norm(second)
    return float((first @ second / denominator.clamp_min(1e-12)).item())


def train(args: argparse.Namespace) -> dict:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    print(
        f"DEVICE requested={args.device} resolved={device} "
        f"cuda_available={torch.cuda.is_available()}",
        flush=True,
    )

    teacher = load_sac(args.sac_model, str(device))
    observations, mode_counts = collect_observations(
        teacher,
        count=args.transitions,
        num_envs=args.num_envs,
        seed=args.seed,
        debug_every=args.collect_debug_every,
    )
    values = critic_values(teacher, observations, args.value_batch_size)
    print(
        f"TARGET value_mean={values.mean():.4f} value_std={values.std():.4f} "
        f"value_min={values.min():.4f} value_max={values.max():.4f}",
        flush=True,
    )
    (
        ranking_terminals,
        ranking_rewards,
        ranking_discounts,
        ranking_exact_scores,
    ) = collect_counterfactual_groups(
        teacher,
        groups=args.ranking_groups,
        candidates=args.ranking_candidates,
        horizon=args.ranking_horizon,
        action_repeat=args.ranking_action_repeat,
        noise_scale=args.ranking_noise_scale,
        discount=args.ranking_discount,
        seed=args.seed,
        debug_every=args.ranking_collect_debug_every,
        value_batch_size=args.value_batch_size,
    )

    permutation = np.random.default_rng(args.seed + 17).permutation(len(values))
    validation_count = max(1, int(len(values) * args.validation_fraction))
    validation_indices = permutation[:validation_count]
    train_indices = permutation[validation_count:]
    train_observations = torch.as_tensor(
        observations[train_indices], device=device
    )
    train_values = torch.as_tensor(values[train_indices], device=device)
    validation_observations = torch.as_tensor(
        observations[validation_indices], device=device
    )
    validation_values = torch.as_tensor(
        values[validation_indices], device=device
    )
    ranking_terminal_tensor = torch.as_tensor(
        ranking_terminals,
        device=device,
    )
    ranking_reward_tensor = torch.as_tensor(
        ranking_rewards,
        device=device,
    )
    ranking_discount_tensor = torch.as_tensor(
        ranking_discounts,
        device=device,
    )
    ranking_exact_score_tensor = torch.as_tensor(
        ranking_exact_scores,
        device=device,
    )

    model = TaskEffectValue(
        obs_dim=observations.shape[1],
        effect_dim=args.effect_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)
    model.set_normalization(train_observations, train_values)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    best_validation_mse = float("inf")
    best_state = None
    training_start = perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(train_indices), device=device)
        epoch_value_loss = 0.0
        epoch_covariance_loss = 0.0
        epoch_ranking_loss = 0.0
        epoch_relative_loss = 0.0
        batches = 0
        for start in range(0, len(order), args.batch_size):
            indices = order[start : start + args.batch_size]
            batch_observations = train_observations[indices]
            batch_values = train_values[indices]
            effects, predicted_values = model(batch_observations)
            normalized_targets = (
                batch_values - model.value_mean
            ) / model.value_std
            normalized_predictions = (
                predicted_values - model.value_mean
            ) / model.value_std
            value_loss = F.mse_loss(
                normalized_predictions,
                normalized_targets,
            )
            covariance_loss = effect_covariance_loss(effects)
            loss = value_loss + args.covariance_weight * covariance_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.max_grad_norm
            )
            optimizer.step()
            epoch_value_loss += float(value_loss.detach())
            epoch_covariance_loss += float(covariance_loss.detach())
            batches += 1

        ranking_batches = 0
        if args.ranking_groups > 0:
            ranking_order = torch.randperm(args.ranking_groups, device=device)
            for start in range(
                0,
                args.ranking_groups,
                args.ranking_group_batch_size,
            ):
                group_indices = ranking_order[
                    start : start + args.ranking_group_batch_size
                ]
                terminals = ranking_terminal_tensor[group_indices]
                flat_terminals = terminals.flatten(0, 1)
                effects, predicted_values = model(flat_terminals)
                predicted_values = predicted_values.reshape(
                    terminals.shape[:2]
                )
                predicted_scores = (
                    ranking_reward_tensor[group_indices]
                    + ranking_discount_tensor[group_indices]
                    * predicted_values
                )
                exact_scores = ranking_exact_score_tensor[group_indices]
                teacher_probabilities = torch.softmax(
                    exact_scores / args.ranking_temperature,
                    dim=-1,
                )
                predicted_log_probabilities = torch.log_softmax(
                    predicted_scores / args.ranking_temperature,
                    dim=-1,
                )
                ranking_loss = F.kl_div(
                    predicted_log_probabilities,
                    teacher_probabilities,
                    reduction="batchmean",
                )
                exact_centered = exact_scores - exact_scores.mean(
                    dim=-1,
                    keepdim=True,
                )
                predicted_centered = predicted_scores - predicted_scores.mean(
                    dim=-1,
                    keepdim=True,
                )
                local_scale = exact_centered.std(
                    dim=-1,
                    keepdim=True,
                    unbiased=False,
                ).clamp_min(0.25)
                relative_loss = F.smooth_l1_loss(
                    predicted_centered / local_scale,
                    exact_centered / local_scale,
                )
                covariance_loss = effect_covariance_loss(effects)
                loss = (
                    args.ranking_weight * ranking_loss
                    + args.relative_weight * relative_loss
                    + args.covariance_weight * covariance_loss
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.max_grad_norm
                )
                optimizer.step()
                epoch_ranking_loss += float(ranking_loss.detach())
                epoch_relative_loss += float(relative_loss.detach())
                ranking_batches += 1

        model.eval()
        with torch.no_grad():
            validation_effects, validation_predictions = model(
                validation_observations
            )
            validation_mse = F.mse_loss(
                validation_predictions,
                validation_values,
            )
            validation_rmse = float(torch.sqrt(validation_mse).item())
            validation_corr = correlation(
                validation_predictions,
                validation_values,
            )
            effect_std = validation_effects.std(dim=0, unbiased=False)
        if float(validation_mse) < best_validation_mse:
            best_validation_mse = float(validation_mse)
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
        if epoch == 1 or epoch % args.debug_every == 0 or epoch == args.epochs:
            print(
                f"EPOCH {epoch:03d}/{args.epochs} "
                f"train_value={epoch_value_loss / batches:.6f} "
                f"train_cov={epoch_covariance_loss / batches:.6f} "
                f"rank_kl={epoch_ranking_loss / max(ranking_batches, 1):.6f} "
                f"rank_rel={epoch_relative_loss / max(ranking_batches, 1):.6f} "
                f"val_rmse={validation_rmse:.4f} "
                f"val_corr={validation_corr:.5f} "
                f"effect_std_min={float(effect_std.min()):.3f} "
                f"effect_std_max={float(effect_std.max()):.3f}",
                flush=True,
            )

    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        validation_effects, validation_predictions = model(
            validation_observations
        )
        final_mse = F.mse_loss(validation_predictions, validation_values)
        final_rmse = float(torch.sqrt(final_mse).item())
        final_corr = correlation(validation_predictions, validation_values)

    checkpoint = {
        "format": "task_effect_value_v1",
        "config": {
            "obs_dim": observations.shape[1],
            "effect_dim": args.effect_dim,
            "hidden_dim": args.hidden_dim,
        },
        "state_dict": model.state_dict(),
        "source_model": str(Path(args.sac_model).resolve()),
        "seed": args.seed,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    metrics = {
        "transitions": args.transitions,
        "train_count": len(train_indices),
        "validation_count": validation_count,
        "effect_dim": args.effect_dim,
        "ranking_groups": args.ranking_groups,
        "ranking_candidates": args.ranking_candidates,
        "mode_counts": mode_counts,
        "validation_rmse": final_rmse,
        "validation_correlation": final_corr,
        "best_validation_mse": best_validation_mse,
        "training_seconds": perf_counter() - training_start,
        "checkpoint": str(output.resolve()),
    }
    metrics_path = output.with_suffix(".json")
    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        f"DONE val_rmse={final_rmse:.4f} val_corr={final_corr:.5f} "
        f"seconds={metrics['training_seconds']:.1f} "
        f"checkpoint={output.resolve()}",
        flush=True,
    )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sac-model",
        default="results/pusher-v5-SAC-expert.zip",
    )
    parser.add_argument(
        "--output",
        default="results/pusher_task_effect_value.pt",
    )
    parser.add_argument("--transitions", type=int, default=30000)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--effect-dim", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--value-batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--covariance-weight", type=float, default=1e-3)
    parser.add_argument("--max-grad-norm", type=float, default=10.0)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--ranking-groups", type=int, default=0)
    parser.add_argument("--ranking-candidates", type=int, default=64)
    parser.add_argument("--ranking-horizon", type=int, default=3)
    parser.add_argument("--ranking-action-repeat", type=int, default=3)
    parser.add_argument("--ranking-noise-scale", type=float, default=0.2)
    parser.add_argument("--ranking-discount", type=float, default=0.99)
    parser.add_argument("--ranking-temperature", type=float, default=1.0)
    parser.add_argument("--ranking-weight", type=float, default=1.0)
    parser.add_argument("--relative-weight", type=float, default=0.25)
    parser.add_argument("--ranking-group-batch-size", type=int, default=8)
    parser.add_argument(
        "--ranking-collect-debug-every",
        type=int,
        default=25,
    )
    parser.add_argument("--seed", type=int, default=1811)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--collect-debug-every", type=int, default=5000)
    parser.add_argument("--debug-every", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
