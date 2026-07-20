"""Matched-protocol direct PPO with source and held-out evaluations."""

from __future__ import annotations

import argparse
import json

import torch

from physics_transfer.variants import step
from scripts.stage15_real_outcome_replay import PRETRAIN_FACTOR
from scripts.stage23_multistep_terminal_value import GOAL
from scripts.stage41_ppo_cognitive_actor import (
    DirectGaussianActor,
    Rollout,
    ValueCritic,
    ppo_update,
    tip_height,
)

HELDOUT_FACTOR = (9.8, 0.04, 1.1, 0.9)


def reset_down_states(count: int, noise: float = 0.04) -> torch.Tensor:
    angles = torch.randn(count, 2) * noise
    return torch.stack([
        torch.cos(angles[:, 0]), torch.sin(angles[:, 0]),
        torch.cos(angles[:, 1]), torch.sin(angles[:, 1]),
        torch.randn(count) * noise, torch.randn(count) * noise,
    ], dim=-1)


def reward_fn(state, next_state, action):
    height = tip_height(next_state)
    success = height >= 1.0
    reward = 0.25 * (height + 2.0) + 5.0 * success.float()
    return reward - 0.005 * action.square().sum(dim=-1), success


def collect_rollout(actor, critic, factor, goal, num_envs, horizon,
                    gamma, gae_lambda, seed):
    torch.manual_seed(seed)
    state = reset_down_states(num_envs)
    factor_tensor = torch.tensor(factor, dtype=state.dtype).view(1, 4).expand(num_envs, -1)
    goal_batch = goal.expand(num_envs, -1)
    states, actions, log_probs, values, rewards, dones = [], [], [], [], [], []
    successes = 0
    for _ in range(horizon):
        with torch.no_grad():
            action, log_prob, _ = actor.sample(state, goal_batch)
            value = critic(state, goal_batch)
            next_state = step(state, action, factor_tensor[:, 0], factor_tensor[:, 1],
                               factor_tensor[:, 2], factor_tensor[:, 3])
            reward, done = reward_fn(state, next_state, action)
        successes += int(done.sum())
        states.append(state); actions.append(action); log_probs.append(log_prob)
        values.append(value); rewards.append(reward); dones.append(done)
        state = torch.where(done.unsqueeze(-1), reset_down_states(num_envs), next_state)
    with torch.no_grad():
        last_value = critic(state, goal_batch)
    states = torch.stack(states); actions = torch.stack(actions)
    log_probs = torch.stack(log_probs); values = torch.stack(values)
    rewards = torch.stack(rewards); dones = torch.stack(dones)
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros(num_envs); next_value = last_value
    for index in reversed(range(horizon)):
        nonterminal = 1.0 - dones[index].float()
        delta = rewards[index] + gamma * next_value * nonterminal - values[index]
        gae = delta + gamma * gae_lambda * nonterminal * gae
        advantages[index] = gae
        next_value = values[index]
    returns = advantages + values
    goals = goal_batch.unsqueeze(0).expand(horizon, -1, -1)
    return Rollout(
        states.reshape(-1, 6), goals.reshape(-1, 6), actions.reshape(-1, 1),
        log_probs.reshape(-1), values.reshape(-1), rewards.reshape(-1),
        dones.reshape(-1), advantages.reshape(-1), returns.reshape(-1),
    ), successes


@torch.no_grad()
def evaluate(actor, factor, goal, count, steps, seed):
    torch.manual_seed(seed)
    state = reset_down_states(count)
    factor_tensor = torch.tensor(factor, dtype=state.dtype).view(1, 4).expand(count, -1)
    maximum = torch.full((count,), -float("inf"))
    success = torch.zeros(count, dtype=torch.bool)
    for _ in range(steps):
        action, _, _ = actor.sample(state, goal.expand(count, -1), deterministic=True)
        state = step(state, action, factor_tensor[:, 0], factor_tensor[:, 1],
                     factor_tensor[:, 2], factor_tensor[:, 3])
        height = tip_height(state)
        maximum = torch.maximum(maximum, height)
        success |= height >= 1.0
    return {
        "success_count": int(success.sum()),
        "success_rate": float(success.float().mean()),
        "mean_max_height": float(maximum.mean()),
        "max_height": float(maximum.max()),
    }


def train(args):
    torch.manual_seed(args.seed)
    goal = GOAL.view(1, -1)
    actor = DirectGaussianActor(args.hidden_dim)
    actor.log_std.data.fill_(args.log_std_init)
    critic = ValueCritic(args.hidden_dim)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)
    history = []
    for iteration in range(args.iterations):
        rollout, collected_successes = collect_rollout(
            actor, critic, PRETRAIN_FACTOR[0], goal, args.num_envs,
            args.rollout_horizon, args.gamma, args.gae_lambda, args.seed + iteration,
        )
        update = ppo_update(
            actor, critic, rollout, actor_optimizer, critic_optimizer,
            args.clip_ratio, args.value_coef, args.entropy_coef,
            args.ppo_epochs, args.minibatch, args.seed + 10000 + iteration,
        )
        if iteration == 0 or (iteration + 1) % args.eval_every == 0:
            evaluation = evaluate(actor, PRETRAIN_FACTOR[0], goal, args.test_count,
                                  args.eval_steps, args.test_seed + iteration)
            history.append({"iteration": iteration + 1,
                            "collected_successes": collected_successes,
                            **update, **evaluation})
    return {
        "environment": PRETRAIN_FACTOR[0],
        "heldout_factor": HELDOUT_FACTOR,
        "start_state": "hanging_down_with_small_noise",
        "evaluation_steps": args.eval_steps,
        "history": history,
        "source_evaluation": evaluate(actor, PRETRAIN_FACTOR[0], goal, args.test_count,
                                       args.eval_steps, args.test_seed + 1000),
        "heldout_evaluation": evaluate(actor, HELDOUT_FACTOR, goal, args.test_count,
                                        args.eval_steps, args.test_seed + 2000),
        "actor_parameter_count": sum(p.numel() for p in actor.parameters()),
        "critic_parameter_count": sum(p.numel() for p in critic.parameters()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--rollout-horizon", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch", type=int, default=1024)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--log-std-init", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--test-count", type=int, default=64)
    parser.add_argument("--test-seed", type=int, default=20260719)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-out", type=str, default="")
    args = parser.parse_args()
    result = {"architecture": "StandardPPO_DirectMLP_NoCognition",
              "config": vars(args), "result": train(args)}
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")


if __name__ == "__main__":
    main()
