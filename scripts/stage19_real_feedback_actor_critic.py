"""Model-free online actor-critic learning from real Acrobot feedback.

The cognitive model is used only for pretraining and one-time operator
initialization.  During deployment no MPC/action teacher is queried.  The
decision actor and twin critics learn only from real transitions
``(s, q, a, r, s_next, done)``.
"""

from __future__ import annotations

import argparse
import copy
import json
import random

import torch
import torch.nn.functional as F
from torch import nn

from physics_transfer.multifactor_data import _random_states
from physics_transfer.variants import step
from scripts.stage13_online_task_loss_adaptation import (
    RuntimeTaskDecision,
    initialize_decision,
    initial_operator,
    tip_height,
)
from scripts.stage15_real_outcome_replay import PRETRAIN_FACTOR, pretrain


class TransitionReplay:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.items = []

    def add(self, state, operator, action, reward, next_state, next_operator, done):
        self.items.append((
            state.detach().clone(), operator.detach().clone(), action.detach().clone(),
            float(reward), next_state.detach().clone(),
            next_operator.detach().clone(), float(done),
        ))
        if len(self.items) > self.capacity:
            self.items = self.items[-self.capacity:]

    def sample(self, batch_size: int):
        if len(self.items) < batch_size:
            return None
        indices = random.sample(range(len(self.items)), batch_size)
        selected = [self.items[index] for index in indices]
        states = torch.cat([item[0] for item in selected], dim=0)
        operators = torch.cat([item[1] for item in selected], dim=0)
        actions = torch.cat([item[2] for item in selected], dim=0)
        rewards = torch.tensor([item[3] for item in selected], dtype=states.dtype).view(-1, 1)
        next_states = torch.cat([item[4] for item in selected], dim=0)
        next_operators = torch.cat([item[5] for item in selected], dim=0)
        dones = torch.tensor([item[6] for item in selected], dtype=states.dtype).view(-1, 1)
        return states, operators, actions, rewards, next_states, next_operators, dones

    def __len__(self):
        return len(self.items)


class Critic(nn.Module):
    def __init__(self, state_dim: int = 6, operator_dim: int = 54, action_dim: int = 1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim + operator_dim + action_dim, 128),
            nn.LayerNorm(128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, state, operator, action):
        return self.network(torch.cat([state, operator, action], dim=-1))


def make_actor(initialized, scope: str):
    actor = RuntimeTaskDecision(initialized)
    if scope == "full":
        initial_basis = actor.base_basis.detach().clone()
        del actor._buffers["base_basis"]
        actor.base_basis = nn.Parameter(initial_basis)
        for parameter in actor.parameters():
            parameter.requires_grad = True
    return actor


def soft_update(target, source, tau):
    with torch.no_grad():
        for target_parameter, source_parameter in zip(target.parameters(), source.parameters()):
            target_parameter.mul_(1.0 - tau).add_(tau * source_parameter)


def transition_reward(state, next_state, action):
    height = tip_height(next_state)
    previous_height = tip_height(state)
    velocity = 0.05 * (next_state[:, 4].square() + next_state[:, 5].square())
    effort = 0.01 * action[:, 0].square()
    done = height >= 1.0
    # Dense progress shaping plus a terminal success reward.  The terminal
    # event is the same event used by evaluation, rather than an MPC label.
    reward = 0.5 * (height - previous_height) - 0.25 * F.relu(1.0 - height).square()
    reward = reward - velocity - effort + 10.0 * done.float()
    return reward.item(), bool(done.item())


def make_fixed_states(count, seed):
    generator = torch.Generator().manual_seed(seed)
    return _random_states(count, generator=generator)


def evaluate(actor, q_const, initial_states, factor, steps):
    state = initial_states.detach().clone()
    factor_tensor = torch.tensor(factor, dtype=state.dtype).view(1, 4).expand(state.shape[0], -1)
    heights = []
    with torch.no_grad():
        for _ in range(steps):
            operator = q_const.expand(state.shape[0], -1)
            action = actor(state, operator)["action"]
            state = step(
                state, action,
                factor_tensor[:, 0], factor_tensor[:, 1],
                factor_tensor[:, 2], factor_tensor[:, 3],
            )
            heights.append(tip_height(state))
    maximum = torch.stack(heights, dim=1).max(dim=1).values
    success = maximum >= 1.0
    return {
        "success_count": int(success.sum().item()),
        "success_rate": float(success.float().mean().item()),
        "mean_max_height": float(maximum.mean().item()),
        "max_height": maximum.tolist(),
    }


def update_actor_critic(actor, target_actor, critic1, critic2, target_critic1,
                        target_critic2, actor_optimizer, critic_optimizer,
                        replay, args, actor_references):
    sampled = replay.sample(args.batch_size)
    if sampled is None:
        return None
    states, operators, actions, rewards, next_states, next_operators, dones = sampled
    with torch.no_grad():
        target_actions = target_actor(next_states, next_operators)["action"]
        noise = (torch.randn_like(target_actions) * args.target_policy_noise).clamp(
            -args.target_noise_clip, args.target_noise_clip
        )
        target_actions = (target_actions + noise).clamp(-1.0, 1.0)
        target_value = torch.minimum(
            target_critic1(next_states, next_operators, target_actions),
            target_critic2(next_states, next_operators, target_actions),
        )
        target = rewards + args.gamma * (1.0 - dones) * target_value

    current1 = critic1(states, operators, actions)
    current2 = critic2(states, operators, actions)
    critic_loss = F.smooth_l1_loss(current1, target) + F.smooth_l1_loss(current2, target)
    critic_optimizer.zero_grad(); critic_loss.backward()
    torch.nn.utils.clip_grad_norm_(list(critic1.parameters()) + list(critic2.parameters()), 10.0)
    critic_optimizer.step()

    actor_loss_value = 0.0
    if args.update_index % args.policy_delay == 0:
        policy_action = actor(states, operators)["action"]
        actor_loss = -critic1(states, operators, policy_action).mean()
        if actor_references:
            distance = sum((parameter - reference).square().mean()
                           for parameter, reference in zip(actor.parameters(), actor_references))
            actor_loss = actor_loss + args.actor_trust_weight * distance
        actor_optimizer.zero_grad(); actor_loss.backward()
        trainable = [parameter for parameter in actor.parameters() if parameter.requires_grad]
        torch.nn.utils.clip_grad_norm_(trainable, 10.0)
        actor_optimizer.step()
        soft_update(target_actor, actor, args.tau)
        actor_loss_value = actor_loss.item()

    soft_update(target_critic1, critic1, args.tau)
    soft_update(target_critic2, critic2, args.tau)
    args.update_index += 1
    return {
        "critic_loss": critic_loss.item(),
        "actor_loss": actor_loss_value,
        "target_mean": target.mean().item(),
    }


def train(args, actor_scope):
    torch.manual_seed(args.seed); random.seed(args.seed)
    cognitive = pretrain(args.cognitive_steps, args.sequence_steps, args.seed)
    initialized = initialize_decision(
        cognitive, args.sequence_steps, args.base_steps, args.mapper_steps, args.seed
    )
    q_const = initial_operator(cognitive, args.sequence_steps, args.seed).detach()
    actor = make_actor(initialized, actor_scope)
    target_actor = copy.deepcopy(actor)
    critic1, critic2 = Critic(), Critic()
    target_critic1, target_critic2 = copy.deepcopy(critic1), copy.deepcopy(critic2)
    actor_optimizer = torch.optim.Adam(
        [parameter for parameter in actor.parameters() if parameter.requires_grad],
        lr=args.actor_lr,
    )
    critic_optimizer = torch.optim.Adam(
        list(critic1.parameters()) + list(critic2.parameters()), lr=args.critic_lr
    )
    replay = TransitionReplay(args.replay_capacity)
    references = [parameter.detach().clone() for parameter in actor.parameters()
                  if parameter.requires_grad]
    # References are only used when the actor scope is residual; for full scope
    # the trust coefficient defaults to zero in the launcher.
    if actor_scope == "residual":
        references = [parameter.detach().clone() for parameter in actor.runtime_residual.parameters()]
    else:
        references = []
    args.update_index = 1
    episodes = []
    factor = PRETRAIN_FACTOR[0]
    for episode in range(args.episodes):
        torch.manual_seed(args.seed + 1000 + episode)
        state = _random_states(1)
        episode_return = 0.0
        maximum_height = -float("inf")
        updates = []
        for _ in range(args.rollout_steps):
            operator = q_const
            with torch.no_grad():
                policy_action = actor(state, operator)["action"]
            noise_scale = max(args.min_exploration_noise,
                              args.exploration_noise * (1.0 - episode / max(args.episodes - 1, 1)))
            action = (policy_action + noise_scale * torch.randn_like(policy_action)).clamp(-1.0, 1.0)
            factor_tensor = torch.tensor(factor, dtype=state.dtype).view(1, 4)
            next_state = step(
                state, action,
                factor_tensor[:, 0], factor_tensor[:, 1],
                factor_tensor[:, 2], factor_tensor[:, 3],
            )
            current_reward, done = transition_reward(state, next_state, action)
            next_operator = q_const
            replay.add(state, operator, action, current_reward,
                       next_state, next_operator, done)
            episode_return += current_reward
            maximum_height = max(maximum_height, tip_height(next_state).item())
            state = next_state.detach()
            if len(replay) >= args.warmup_transitions:
                result = update_actor_critic(
                    actor, target_actor, critic1, critic2, target_critic1,
                    target_critic2, actor_optimizer, critic_optimizer,
                    replay, args, references,
                )
                if result is not None:
                    updates.append(result)
            if done:
                break
        episodes.append({
            "success": maximum_height >= 1.0,
            "max_height": maximum_height,
            "return": episode_return,
            "updates": len(updates),
            "last_critic_loss": updates[-1]["critic_loss"] if updates else 0.0,
            "last_actor_loss": updates[-1]["actor_loss"] if updates else 0.0,
        })
    fixed_states = make_fixed_states(args.test_count, args.test_seed)
    return {
        "actor_scope": actor_scope,
        "initial_fixed_evaluation": evaluate(
            make_actor(initialized, actor_scope), q_const, fixed_states, factor, args.rollout_steps
        ),
        "online_training": {
            "success_count": sum(item["success"] for item in episodes),
            "success_rate": sum(item["success"] for item in episodes) / len(episodes),
            "last_20_success_rate": sum(item["success"] for item in episodes[-20:]) / min(20, len(episodes)),
            "mean_return": sum(item["return"] for item in episodes) / len(episodes),
            "episodes": episodes,
        },
        "final_fixed_evaluation": evaluate(actor, q_const, fixed_states, factor, args.rollout_steps),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--actor-scope", choices=("residual", "full"), default="full")
    parser.add_argument("--cognitive-steps", type=int, default=100)
    parser.add_argument("--base-steps", type=int, default=100)
    parser.add_argument("--mapper-steps", type=int, default=100)
    parser.add_argument("--sequence-steps", type=int, default=16)
    parser.add_argument("--replay-capacity", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--warmup-transitions", type=int, default=512)
    parser.add_argument("--episodes", type=int, default=250)
    parser.add_argument("--rollout-steps", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--policy-delay", type=int, default=2)
    parser.add_argument("--target-policy-noise", type=float, default=0.10)
    parser.add_argument("--target-noise-clip", type=float, default=0.20)
    parser.add_argument("--exploration-noise", type=float, default=0.20)
    parser.add_argument("--min-exploration-noise", type=float, default=0.03)
    parser.add_argument("--actor-trust-weight", type=float, default=0.0)
    parser.add_argument("--test-count", type=int, default=20)
    parser.add_argument("--test-seed", type=int, default=20260718)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(train(args, args.actor_scope), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

