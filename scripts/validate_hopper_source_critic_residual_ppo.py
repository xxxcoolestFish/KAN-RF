"""Hopper residual PPO with inherited source Critic and cognitive decoding."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from scripts.diagnose_hopper_pullback_effect import (
    fit_distilled_source_counterfactual_context,
    fit_orthogonal_control_transform,
    fit_paired_source_counterfactual_context,
)
from scripts.prescreen_hopper_physics_shifts import SHIFTS, make_shifted_env
from scripts.validate_hopper_joint_online_adaptation import (
    FrozenSourcePolicy,
    cognitive_action_and_features,
    decode_cached_effect_residual,
    load_cognition,
)


class RankOneMeanActor(nn.Module):
    """A low-rank mean policy with full-rank exploration noise."""

    def __init__(self, state_dim: int, hidden_dim: int, initial_log_std: float):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.amplitude = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.amplitude.weight)
        nn.init.zeros_(self.amplitude.bias)
        direction = torch.randn(3)
        self.direction = nn.Parameter(direction / direction.norm())
        self.log_std = nn.Parameter(torch.full((3,), initial_log_std))

    def mean(self, state):
        amplitude = torch.tanh(self.amplitude(self.trunk(state)))
        direction = self.direction / self.direction.norm().clamp_min(1e-6)
        return amplitude * direction

    def distribution(self, state):
        mean = self.mean(state)
        std = self.log_std.clamp(-5.0, 0.0).exp().expand_as(mean)
        return torch.distributions.Normal(mean, std)


class FullMeanActor(nn.Module):
    """Zero-initialized full mean; rank is measured rather than hard-coded."""

    def __init__(self, state_dim: int, hidden_dim: int, initial_log_std: float):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.mean_head = nn.Linear(hidden_dim, 3)
        nn.init.zeros_(self.mean_head.weight)
        nn.init.zeros_(self.mean_head.bias)
        self.log_std = nn.Parameter(torch.full((3,), initial_log_std))

    def mean(self, state):
        return torch.tanh(self.mean_head(self.trunk(state)))

    def distribution(self, state):
        mean = self.mean(state)
        std = self.log_std.clamp(-5.0, 0.0).exp().expand_as(mean)
        return torch.distributions.Normal(mean, std)


class ResidualValue(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, state):
        return self.network(state).squeeze(-1)


def normalized_state(source_policy, observation):
    observation = torch.as_tensor(
        observation,
        dtype=torch.float32,
        device=source_policy.device,
    )
    return (
        (observation - source_policy.mean)
        / (source_policy.variance + 1e-8).sqrt()
    ).clamp(-10.0, 10.0)


@torch.no_grad()
def source_value(source_policy, normalized):
    return source_policy.model.policy.predict_values(
        normalized,
    ).squeeze(-1)


@torch.no_grad()
def physical_action(
    observation,
    residual,
    source_policy,
    basis,
    source_context,
    target_context,
    delta_scale,
    args,
):
    source_action = source_policy.action(observation).cpu().numpy()
    if args.residual_space == "action":
        return np.clip(
            source_action
            + args.residual_scale
            * np.asarray(residual, dtype=np.float32),
            -1.0,
            1.0,
        )
    _, _, details = cognitive_action_and_features(
        observation,
        source_policy,
        basis,
        source_context,
        target_context,
        delta_scale,
        args.pullback_damping,
        "identity",
        0.05,
        return_details=True,
    )
    return decode_cached_effect_residual(
        residual,
        details,
        basis,
        target_context,
        args,
    )


@torch.no_grad()
def evaluate(
    actor,
    source_policy,
    basis,
    source_context,
    target_context,
    delta_scale,
    args,
):
    returns = []
    lengths = []
    residual_norms = []
    for episode in range(args.evaluation_episodes):
        environment = make_shifted_env(
            SHIFTS[args.target],
            args.seed + 10000 + episode,
        )()
        observation, _ = environment.reset(
            seed=args.seed + 10000 + episode,
        )
        total = 0.0
        length = 0
        while True:
            state = normalized_state(
                source_policy, observation,
            ).unsqueeze(0)
            residual = actor.mean(state)[0].cpu().numpy()
            action = physical_action(
                observation,
                residual,
                source_policy,
                basis,
                source_context,
                target_context,
                delta_scale,
                args,
            )
            observation, reward, terminated, truncated, _ = (
                environment.step(action)
            )
            total += reward
            length += 1
            residual_norms.append(float(np.linalg.norm(residual)))
            if terminated or truncated:
                break
        environment.close()
        returns.append(total)
        lengths.append(length)
    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_episode_length": float(np.mean(lengths)),
        "residual_l2_mean": float(np.mean(residual_norms)),
    }


def collect_rollout(
    environments,
    observations,
    actor,
    residual_value,
    source_policy,
    basis,
    source_context,
    target_context,
    delta_scale,
    reward_scale,
    args,
    *,
    actor_enabled,
):
    states = []
    residuals = []
    log_probabilities = []
    rewards = []
    dones = []
    values = []
    episode_returns = []
    running_returns = getattr(
        collect_rollout, "running_returns", np.zeros(len(environments)),
    )
    for _ in range(args.rollout_steps):
        state = torch.stack([
            normalized_state(source_policy, observation)
            for observation in observations
        ])
        with torch.no_grad():
            distribution = actor.distribution(state)
            residual = (
                distribution.sample()
                if actor_enabled
                else torch.zeros(
                    len(environments), 3, device=state.device,
                )
            )
            log_probability = distribution.log_prob(residual).sum(dim=-1)
            value = (
                source_value(source_policy, state)
                + residual_value(state)
            )
        next_observations = []
        step_rewards = []
        step_dones = []
        for index, environment in enumerate(environments):
            action = physical_action(
                observations[index],
                residual[index].cpu().numpy(),
                source_policy,
                basis,
                source_context,
                target_context,
                delta_scale,
                args,
            )
            following, reward, terminated, truncated, _ = environment.step(
                action,
            )
            done = terminated or truncated
            running_returns[index] += reward
            if done:
                episode_returns.append(float(running_returns[index]))
                running_returns[index] = 0.0
                following, _ = environment.reset()
            next_observations.append(following)
            step_rewards.append(reward / reward_scale)
            step_dones.append(done)
        states.append(state)
        residuals.append(residual)
        log_probabilities.append(log_probability)
        rewards.append(torch.as_tensor(
            step_rewards, dtype=torch.float32, device=state.device,
        ))
        dones.append(torch.as_tensor(
            step_dones, dtype=torch.float32, device=state.device,
        ))
        values.append(value)
        observations = next_observations
    collect_rollout.running_returns = running_returns
    with torch.no_grad():
        final_state = torch.stack([
            normalized_state(source_policy, observation)
            for observation in observations
        ])
        final_value = (
            source_value(source_policy, final_state)
            + residual_value(final_state)
        )
    return {
        "state": torch.stack(states),
        "residual": torch.stack(residuals),
        "old_log_probability": torch.stack(log_probabilities),
        "reward": torch.stack(rewards),
        "done": torch.stack(dones),
        "value": torch.stack(values),
        "final_value": final_value,
        "observations": observations,
        "episode_returns": episode_returns,
    }


def advantages_and_returns(rollout, args):
    reward = rollout["reward"]
    done = rollout["done"]
    value = rollout["value"]
    advantage = torch.zeros_like(reward)
    accumulator = torch.zeros(
        reward.shape[1], dtype=reward.dtype, device=reward.device,
    )
    next_value = rollout["final_value"]
    for time in reversed(range(reward.shape[0])):
        continuation = 1.0 - done[time]
        delta = (
            reward[time]
            + args.gamma * next_value * continuation
            - value[time]
        )
        accumulator = (
            delta
            + args.gamma
            * args.gae_lambda
            * continuation
            * accumulator
        )
        advantage[time] = accumulator
        next_value = value[time]
    return advantage, advantage + value


def update_networks(
    rollout,
    advantage,
    target_return,
    actor,
    residual_value,
    source_policy,
    actor_optimizer,
    critic_optimizer,
    args,
    *,
    actor_enabled,
):
    state = rollout["state"].flatten(0, 1)
    residual = rollout["residual"].flatten(0, 1)
    old_log_probability = rollout["old_log_probability"].flatten()
    advantage = advantage.flatten()
    target_return = target_return.flatten()
    if actor_enabled:
        advantage = (
            advantage - advantage.mean()
        ) / advantage.std().clamp_min(1e-6)
    indices = torch.arange(state.shape[0], device=state.device)
    actor_losses = []
    critic_losses = []
    for _ in range(args.update_epochs):
        permutation = indices[torch.randperm(indices.shape[0])]
        for start in range(0, len(permutation), args.minibatch_size):
            batch = permutation[start:start + args.minibatch_size]
            if actor_enabled:
                distribution = actor.distribution(state[batch])
                log_probability = distribution.log_prob(
                    residual[batch],
                ).sum(dim=-1)
                ratio = (
                    log_probability - old_log_probability[batch]
                ).exp()
                clipped = ratio.clamp(
                    1.0 - args.clip_ratio,
                    1.0 + args.clip_ratio,
                )
                policy_loss = -torch.minimum(
                    ratio * advantage[batch],
                    clipped * advantage[batch],
                ).mean()
                mean_penalty = actor.mean(state[batch]).square().mean()
                entropy = distribution.entropy().sum(dim=-1).mean()
                actor_loss = (
                    policy_loss
                    + args.source_trust_weight * mean_penalty
                    - args.entropy_coefficient * entropy
                )
                actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(
                    actor.parameters(), args.max_grad_norm,
                )
                actor_optimizer.step()
                actor_losses.append(float(actor_loss.detach()))
            predicted_value = (
                source_value(source_policy, state[batch])
                + residual_value(state[batch])
            )
            critic_loss = 0.5 * (
                predicted_value - target_return[batch]
            ).square().mean()
            critic_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(
                residual_value.parameters(), args.max_grad_norm,
            )
            critic_optimizer.step()
            critic_losses.append(float(critic_loss.detach()))
    return {
        "actor_loss": (
            float(np.mean(actor_losses)) if actor_losses else 0.0
        ),
        "critic_loss": float(np.mean(critic_losses)),
        "advantage_abs_mean": float(advantage.abs().mean()),
    }


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    source_policy = FrozenSourcePolicy(
        args.source_model, args.source_norm, device, args.seed,
    )
    basis, source_context, _, delta_scale = load_cognition(args, device)
    if args.cognition_mode == "distilled_source_twin":
        target_context, _ = fit_distilled_source_counterfactual_context(
            source_policy,
            basis,
            source_context,
            args,
            device,
        )
    elif args.cognition_mode == "paired_source_oracle":
        target_context, _ = fit_paired_source_counterfactual_context(
            source_policy,
            basis,
            source_context,
            args,
            device,
        )
    else:
        args.cognition_mode = "orthogonal_transform"
        target_context, _ = fit_orthogonal_control_transform(
            source_policy,
            basis,
            source_context,
            args,
            device,
        )
    actor_class = (
        RankOneMeanActor if args.actor_rank == 1 else FullMeanActor
    )
    actor = actor_class(
        basis.state_dim,
        args.hidden_dim,
        args.initial_log_std,
    ).to(device)
    residual_value = ResidualValue(
        basis.state_dim, args.hidden_dim,
    ).to(device)
    actor_optimizer = torch.optim.Adam(
        actor.parameters(), lr=args.actor_learning_rate,
    )
    critic_optimizer = torch.optim.Adam(
        residual_value.parameters(), lr=args.critic_learning_rate,
    )
    reward_scale = float(
        source_policy.model.get_env().ret_rms.var ** 0.5
    ) if source_policy.model.get_env() is not None else args.reward_scale
    # PPO.load does not retain VecNormalize; use its stored source scale.
    reward_scale = args.reward_scale

    environments = [
        make_shifted_env(
            SHIFTS[args.target], args.seed + 500 + index,
        )()
        for index in range(args.parallel_envs)
    ]
    observations = [
        environment.reset(seed=args.seed + 500 + index)[0]
        for index, environment in enumerate(environments)
    ]
    history = [{
        "phase": 0,
        "decision_transitions": 0,
        "actor_enabled": False,
        **evaluate(
            actor,
            source_policy,
            basis,
            source_context,
            target_context,
            delta_scale,
            args,
        ),
    }]
    print(history[-1], flush=True)
    completed = 0
    phase = 0
    transitions_per_phase = args.parallel_envs * args.rollout_steps
    while completed < args.decision_transitions:
        phase += 1
        actor_enabled = phase > args.critic_warmup_phases
        rollout = collect_rollout(
            environments,
            observations,
            actor,
            residual_value,
            source_policy,
            basis,
            source_context,
            target_context,
            delta_scale,
            reward_scale,
            args,
            actor_enabled=actor_enabled,
        )
        observations = rollout["observations"]
        advantage, target_return = advantages_and_returns(rollout, args)
        training = update_networks(
            rollout,
            advantage,
            target_return,
            actor,
            residual_value,
            source_policy,
            actor_optimizer,
            critic_optimizer,
            args,
            actor_enabled=actor_enabled,
        )
        completed += transitions_per_phase
        record = {
            "phase": phase,
            "decision_transitions": completed,
            "actor_enabled": actor_enabled,
            "rollout_completed_episode_return_mean": (
                float(np.mean(rollout["episode_returns"]))
                if rollout["episode_returns"]
                else None
            ),
            **training,
        }
        if (
            phase % args.evaluate_every_phases == 0
            or completed >= args.decision_transitions
        ):
            record.update(evaluate(
                actor,
                source_policy,
                basis,
                source_context,
                target_context,
                delta_scale,
                args,
            ))
        history.append(record)
        print(record, flush=True)
    for environment in environments:
        environment.close()
    output = {
        "experiment": "HopperInheritedCriticResidualPPO",
        "target": args.target,
        "residual_space": args.residual_space,
        "source_actor_frozen": True,
        "source_critic_inherited": True,
        "residual_value_zero_initialized": True,
        "actor_delayed_until_critic_warmup": True,
        "actor_mean_rank_constraint": args.actor_rank,
        "cognition_mode": args.cognition_mode,
        "physical_parameters_visible_to_learner": False,
        "config": vars(args),
        "history": history,
    }
    Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json_out).write_text(
        json.dumps(output, indent=2) + "\n",
        encoding="utf-8",
    )
    torch.save({
        "actor": actor.state_dict(),
        "residual_value": residual_value.state_dict(),
        "config": vars(args),
    }, args.model_out)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1811)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--target", default="combo_mild")
    parser.add_argument("--parallel-envs", type=int, default=4)
    parser.add_argument("--cognition-warmup", type=int, default=2048)
    parser.add_argument("--cognition-batch", type=int, default=64)
    parser.add_argument("--warmup-noise", type=float, default=0.3)
    parser.add_argument(
        "--warmup-exploration",
        choices=("gaussian_clipped", "symmetric"),
        default="symmetric",
    )
    parser.add_argument("--transform-ridge", type=float, default=10.0)
    parser.add_argument("--drift-ridge", type=float, default=100.0)
    parser.add_argument("--stein-ridge", type=float, default=1.0)
    parser.add_argument("--transform-iterations", type=int, default=5)
    parser.add_argument(
        "--cognition-mode",
        choices=(
            "orthogonal_transform",
            "paired_source_oracle",
            "distilled_source_twin",
        ),
        default="orthogonal_transform",
    )
    parser.add_argument("--decision-transitions", type=int, default=4096)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--critic-warmup-phases", type=int, default=2)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument(
        "--actor-rank",
        type=int,
        choices=(0, 1),
        default=0,
        help="0 uses a full mean; 1 uses the rank-one diagnostic actor.",
    )
    parser.add_argument("--actor-learning-rate", type=float, default=1e-4)
    parser.add_argument("--critic-learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--source-trust-weight", type=float, default=0.1)
    parser.add_argument("--entropy-coefficient", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--initial-log-std", type=float, default=-2.0)
    parser.add_argument("--residual-scale", type=float, default=0.25)
    parser.add_argument(
        "--residual-space",
        choices=("action", "effect"),
        default="effect",
    )
    parser.add_argument("--pullback-damping", type=float, default=0.05)
    parser.add_argument("--reward-scale", type=float, default=125.347642)
    parser.add_argument("--evaluation-episodes", type=int, default=3)
    parser.add_argument("--evaluate-every-phases", type=int, default=1)
    parser.add_argument(
        "--source-model",
        default="results/hopper_source_sb3_ppo_continued_seed1811.zip",
    )
    parser.add_argument(
        "--source-norm",
        default="results/hopper_source_sb3_vecnorm_continued_seed1811.pkl",
    )
    parser.add_argument(
        "--cognition-checkpoint",
        default="results/hopper_source_control_sobolev_calibrated_seed1811.pt",
    )
    parser.add_argument(
        "--source-twin-checkpoint",
        default="results/hopper_source_affine_twin_seed1811.pt",
    )
    parser.add_argument(
        "--model-out",
        default="results/hopper_source_critic_effect_seed1811.pt",
    )
    parser.add_argument(
        "--json-out",
        default="results/hopper_source_critic_effect_seed1811.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
