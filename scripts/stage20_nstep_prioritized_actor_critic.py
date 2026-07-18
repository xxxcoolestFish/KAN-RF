"""Test a corrected real-feedback loss without an action teacher.

Changes over Stage 19:
- n-step TD targets for long-horizon credit assignment;
- potential-based height shaping, which preserves the sparse task objective;
- prioritized replay with extra mass on successful episodes;
- a small parameter trust-region penalty for the actor.
"""

from __future__ import annotations

import argparse
import copy
import json
import random

import torch
import torch.nn.functional as F

from physics_transfer.multifactor_data import _random_states
from physics_transfer.variants import step
from scripts.stage13_online_task_loss_adaptation import initialize_decision, initial_operator, tip_height
from scripts.stage15_real_outcome_replay import PRETRAIN_FACTOR, pretrain
from scripts.stage19_real_feedback_actor_critic import Critic, evaluate, make_actor, soft_update


class NStepPrioritizedReplay:
    def __init__(self, capacity, n_step, gamma, alpha, beta, success_priority):
        self.capacity = capacity
        self.n_step = n_step
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.success_priority = success_priority
        self.items = []
        self.priorities = []

    def add_episode(self, trajectory):
        for start in range(len(trajectory)):
            discounted = 0.0
            end = start
            done_n = False
            for offset in range(self.n_step):
                end = start + offset
                item = trajectory[end]
                discounted += (self.gamma ** offset) * item["reward"]
                if item["done"] or end == len(trajectory) - 1:
                    done_n = bool(item["done"])
                    break
            final = trajectory[end]
            success_episode = any(item["done"] for item in trajectory)
            record = (
                trajectory[start]["state"].detach().clone(),
                trajectory[start]["operator"].detach().clone(),
                trajectory[start]["action"].detach().clone(),
                float(discounted),
                final["next_state"].detach().clone(),
                final["operator"].detach().clone(),
                float(done_n),
            )
            priority = 1.0 + self.success_priority * float(success_episode)
            if done_n:
                priority += self.success_priority
            self.items.append(record)
            self.priorities.append(priority)
        if len(self.items) > self.capacity:
            excess = len(self.items) - self.capacity
            self.items = self.items[excess:]
            self.priorities = self.priorities[excess:]

    def __len__(self):
        return len(self.items)

    def sample(self, batch_size):
        if len(self.items) < batch_size:
            return None
        priorities = torch.tensor(self.priorities, dtype=torch.float32)
        probabilities = priorities.pow(self.alpha)
        probabilities = probabilities / probabilities.sum()
        indices = torch.multinomial(probabilities, batch_size, replacement=True)
        selected = [self.items[int(index)] for index in indices]
        states = torch.cat([item[0] for item in selected], dim=0)
        operators = torch.cat([item[1] for item in selected], dim=0)
        actions = torch.cat([item[2] for item in selected], dim=0)
        rewards = torch.tensor([item[3] for item in selected], dtype=states.dtype).view(-1, 1)
        next_states = torch.cat([item[4] for item in selected], dim=0)
        next_operators = torch.cat([item[5] for item in selected], dim=0)
        dones = torch.tensor([item[6] for item in selected], dtype=states.dtype).view(-1, 1)
        weights = (len(self.items) * probabilities[indices]).pow(-self.beta)
        weights = (weights / weights.max()).to(states.dtype).view(-1, 1)
        return indices, states, operators, actions, rewards, next_states, next_operators, dones, weights

    def update_priorities(self, indices, errors):
        for index, error in zip(indices.tolist(), errors.detach().cpu().tolist()):
            self.priorities[index] = max(float(abs(error)) + 1e-3, 1e-3)


def shaped_reward(state, next_state, action, gamma):
    height = tip_height(next_state)
    previous_height = tip_height(state)
    velocity = 0.05 * (next_state[:, 4].square() + next_state[:, 5].square())
    effort = 0.01 * action[:, 0].square()
    done = height >= 1.0
    # Potential shaping: gamma*Phi(s') - Phi(s), Phi=tip height.
    reward = gamma * height - previous_height - velocity - effort
    reward = reward + 10.0 * done.float()
    return reward.item(), bool(done.item())


def update(actor, target_actor, critic1, critic2, target_critic1, target_critic2,
           actor_optimizer, critic_optimizer, replay, args, references):
    sampled = replay.sample(args.batch_size)
    if sampled is None:
        return None
    indices, states, operators, actions, rewards, next_states, next_operators, dones, weights = sampled
    with torch.no_grad():
        next_actions = target_actor(next_states, next_operators)["action"]
        noise = (torch.randn_like(next_actions) * args.target_policy_noise).clamp(
            -args.target_noise_clip, args.target_noise_clip
        )
        next_actions = (next_actions + noise).clamp(-1.0, 1.0)
        next_value = torch.minimum(
            target_critic1(next_states, next_operators, next_actions),
            target_critic2(next_states, next_operators, next_actions),
        )
        target = rewards + (args.gamma ** args.n_step) * (1.0 - dones) * next_value
        target = target.clamp(-100.0, 100.0)
    q1 = critic1(states, operators, actions)
    q2 = critic2(states, operators, actions)
    error1 = q1 - target
    error2 = q2 - target
    critic_loss = (weights * F.smooth_l1_loss(q1, target, reduction="none")).mean()
    critic_loss = critic_loss + (weights * F.smooth_l1_loss(q2, target, reduction="none")).mean()
    critic_optimizer.zero_grad(); critic_loss.backward()
    torch.nn.utils.clip_grad_norm_(list(critic1.parameters()) + list(critic2.parameters()), 5.0)
    critic_optimizer.step()
    replay.update_priorities(indices, torch.minimum(error1.abs(), error2.abs()).squeeze(-1))

    actor_loss_value = 0.0
    if args.update_index % args.policy_delay == 0:
        policy_action = actor(states, operators)["action"]
        actor_loss = -critic1(states, operators, policy_action).mean()
        if references:
            trust = sum((parameter - reference).square().mean()
                        for parameter, reference in zip(actor.runtime_residual.parameters(), references))
            actor_loss = actor_loss + args.actor_trust_weight * trust
        actor_optimizer.zero_grad(); actor_loss.backward()
        trainable = [parameter for parameter in actor.parameters() if parameter.requires_grad]
        torch.nn.utils.clip_grad_norm_(trainable, 5.0)
        actor_optimizer.step()
        soft_update(target_actor, actor, args.tau)
        actor_loss_value = actor_loss.item()
    soft_update(target_critic1, critic1, args.tau)
    soft_update(target_critic2, critic2, args.tau)
    args.update_index += 1
    return {"critic_loss": critic_loss.item(), "actor_loss": actor_loss_value}


def run(args):
    torch.manual_seed(args.seed); random.seed(args.seed)
    cognitive = pretrain(args.cognitive_steps, args.sequence_steps, args.seed)
    initialized = initialize_decision(
        cognitive, args.sequence_steps, args.base_steps, args.mapper_steps, args.seed
    )
    q_const = initial_operator(cognitive, args.sequence_steps, args.seed).detach()
    actor = make_actor(initialized, args.actor_scope)
    target_actor = copy.deepcopy(actor)
    critic1, critic2 = Critic(), Critic()
    target_critic1, target_critic2 = copy.deepcopy(critic1), copy.deepcopy(critic2)
    actor_optimizer = torch.optim.Adam(
        [parameter for parameter in actor.parameters() if parameter.requires_grad], lr=args.actor_lr
    )
    critic_optimizer = torch.optim.Adam(
        list(critic1.parameters()) + list(critic2.parameters()), lr=args.critic_lr
    )
    replay = NStepPrioritizedReplay(
        args.replay_capacity, args.n_step, args.gamma, args.priority_alpha,
        args.priority_beta, args.success_priority,
    )
    references = [parameter.detach().clone() for parameter in actor.runtime_residual.parameters()]
    fixed_generator = torch.Generator().manual_seed(args.test_seed)
    fixed_states = _random_states(args.test_count, generator=fixed_generator)
    initial = evaluate(actor, q_const, fixed_states, PRETRAIN_FACTOR[0], args.rollout_steps)
    args.update_index = 1
    training = []
    factor = PRETRAIN_FACTOR[0]
    for episode in range(args.episodes):
        torch.manual_seed(args.seed + 1000 + episode)
        state = _random_states(1)
        trajectory = []
        maximum_height = -float("inf")
        episode_return = 0.0
        for _ in range(args.rollout_steps):
            operator = q_const
            with torch.no_grad():
                policy_action = actor(state, operator)["action"]
            noise_scale = max(
                args.min_exploration_noise,
                args.exploration_noise * (1.0 - episode / max(args.episodes - 1, 1)),
            )
            action = (policy_action + noise_scale * torch.randn_like(policy_action)).clamp(-1.0, 1.0)
            factor_tensor = torch.tensor(factor, dtype=state.dtype).view(1, 4)
            next_state = step(
                state, action,
                factor_tensor[:, 0], factor_tensor[:, 1],
                factor_tensor[:, 2], factor_tensor[:, 3],
            )
            reward, done = shaped_reward(state, next_state, action, args.gamma)
            trajectory.append({
                "state": state.detach(), "operator": operator.detach(),
                "action": action.detach(), "reward": reward,
                "next_state": next_state.detach(), "done": done,
                "max_height": tip_height(next_state).item(),
            })
            episode_return += reward
            maximum_height = max(maximum_height, tip_height(next_state).item())
            state = next_state.detach()
            if done:
                break
        replay.add_episode(trajectory)
        metrics = []
        for _ in range(args.updates_per_episode):
            if len(replay) < args.warmup_transitions:
                break
            result = update(
                actor, target_actor, critic1, critic2, target_critic1,
                target_critic2, actor_optimizer, critic_optimizer,
                replay, args, references if args.actor_scope == "residual" else [],
            )
            if result is not None:
                metrics.append(result)
        training.append({
            "success": maximum_height >= 1.0,
            "max_height": maximum_height,
            "return": episode_return,
            "critic_loss": metrics[-1]["critic_loss"] if metrics else 0.0,
            "actor_loss": metrics[-1]["actor_loss"] if metrics else 0.0,
        })
    final = evaluate(actor, q_const, fixed_states, factor, args.rollout_steps)
    return {
        "actor_scope": args.actor_scope,
        "cognitive_usage": "pretraining_and_one_time_initialization_only",
        "teacher_usage": "none_during_online_learning",
        "loss": {"n_step": args.n_step, "priority_alpha": args.priority_alpha,
                 "success_priority": args.success_priority},
        "initial_fixed_evaluation": initial,
        "online_training": {
            "success_rate": sum(item["success"] for item in training) / len(training),
            "last_20_success_rate": sum(item["success"] for item in training[-20:]) / min(20, len(training)),
            "mean_return": sum(item["return"] for item in training) / len(training),
            "episodes": training,
        },
        "final_fixed_evaluation": final,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--actor-scope", choices=("residual", "full"), default="full")
    parser.add_argument("--cognitive-steps", type=int, default=100)
    parser.add_argument("--base-steps", type=int, default=100)
    parser.add_argument("--mapper-steps", type=int, default=100)
    parser.add_argument("--sequence-steps", type=int, default=16)
    parser.add_argument("--n-step", type=int, default=8)
    parser.add_argument("--replay-capacity", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--warmup-transitions", type=int, default=512)
    parser.add_argument("--updates-per-episode", type=int, default=32)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--rollout-steps", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--actor-lr", type=float, default=5e-5)
    parser.add_argument("--critic-lr", type=float, default=5e-4)
    parser.add_argument("--tau", type=float, default=0.003)
    parser.add_argument("--policy-delay", type=int, default=2)
    parser.add_argument("--target-policy-noise", type=float, default=0.08)
    parser.add_argument("--target-noise-clip", type=float, default=0.15)
    parser.add_argument("--exploration-noise", type=float, default=0.20)
    parser.add_argument("--min-exploration-noise", type=float, default=0.04)
    parser.add_argument("--actor-trust-weight", type=float, default=1e-4)
    parser.add_argument("--priority-alpha", type=float, default=0.6)
    parser.add_argument("--priority-beta", type=float, default=0.4)
    parser.add_argument("--success-priority", type=float, default=4.0)
    parser.add_argument("--test-count", type=int, default=20)
    parser.add_argument("--test-seed", type=int, default=20260718)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(run(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

