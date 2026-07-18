"""Corrected model-free actor-critic experiment using only real transitions.

This launcher deliberately contains no MPC/action teacher.  The cognitive
model is used only before deployment to produce a fixed operator code.
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
from scripts.stage19_real_feedback_actor_critic import (
    Critic,
    TransitionReplay,
    evaluate,
    make_actor,
    transition_reward,
    soft_update,
)


def update(actor, target_actor, critic1, critic2, target_critic1, target_critic2,
           actor_optimizer, critic_optimizer, replay, args):
    sampled = replay.sample(args.batch_size)
    if sampled is None:
        return None
    states, operators, actions, rewards, next_states, next_operators, dones = sampled
    with torch.no_grad():
        next_actions = target_actor(next_states, next_operators)["action"]
        noise = (torch.randn_like(next_actions) * args.target_policy_noise).clamp(
            -args.target_noise_clip, args.target_noise_clip
        )
        next_actions = (next_actions + noise).clamp(-1.0, 1.0)
        target_q = torch.minimum(
            target_critic1(next_states, next_operators, next_actions),
            target_critic2(next_states, next_operators, next_actions),
        )
        target = rewards + args.gamma * (1.0 - dones) * target_q
    q1 = critic1(states, operators, actions)
    q2 = critic2(states, operators, actions)
    critic_loss = F.smooth_l1_loss(q1, target) + F.smooth_l1_loss(q2, target)
    critic_optimizer.zero_grad(); critic_loss.backward()
    torch.nn.utils.clip_grad_norm_(list(critic1.parameters()) + list(critic2.parameters()), 10.0)
    critic_optimizer.step()
    actor_loss = 0.0
    if args.update_index % args.policy_delay == 0:
        policy_action = actor(states, operators)["action"]
        loss = -critic1(states, operators, policy_action).mean()
        actor_optimizer.zero_grad(); loss.backward()
        trainable = [parameter for parameter in actor.parameters() if parameter.requires_grad]
        torch.nn.utils.clip_grad_norm_(trainable, 10.0)
        actor_optimizer.step()
        soft_update(target_actor, actor, args.tau)
        actor_loss = loss.item()
    soft_update(target_critic1, critic1, args.tau)
    soft_update(target_critic2, critic2, args.tau)
    args.update_index += 1
    return {"critic_loss": critic_loss.item(), "actor_loss": actor_loss}


def make_test_states(count, seed):
    generator = torch.Generator().manual_seed(seed)
    return _random_states(count, generator=generator)


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
        [p for p in actor.parameters() if p.requires_grad], lr=args.actor_lr
    )
    critic_optimizer = torch.optim.Adam(
        list(critic1.parameters()) + list(critic2.parameters()), lr=args.critic_lr
    )
    replay = TransitionReplay(args.replay_capacity)
    fixed_states = make_test_states(args.test_count, args.test_seed)
    initial = evaluate(actor, q_const, fixed_states, PRETRAIN_FACTOR[0], args.rollout_steps)
    args.update_index = 1
    training = []
    factor = PRETRAIN_FACTOR[0]
    for episode in range(args.episodes):
        torch.manual_seed(args.seed + 1000 + episode)
        state = _random_states(1)
        episode_return = 0.0
        max_height = -float("inf")
        update_metrics = []
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
            current_reward, done = transition_reward(state, next_state, action)
            replay.add(state, operator, action, current_reward, next_state, operator, done)
            episode_return += current_reward
            max_height = max(max_height, tip_height(next_state).item())
            state = next_state.detach()
            if len(replay) >= args.warmup_transitions:
                metrics = update(
                    actor, target_actor, critic1, critic2, target_critic1,
                    target_critic2, actor_optimizer, critic_optimizer,
                    replay, args,
                )
                if metrics is not None:
                    update_metrics.append(metrics)
            if done:
                break
        training.append({
            "success": max_height >= 1.0,
            "max_height": max_height,
            "return": episode_return,
            "critic_loss": update_metrics[-1]["critic_loss"] if update_metrics else 0.0,
            "actor_loss": update_metrics[-1]["actor_loss"] if update_metrics else 0.0,
        })
    final = evaluate(actor, q_const, fixed_states, factor, args.rollout_steps)
    return {
        "actor_scope": args.actor_scope,
        "cognitive_usage": "pretraining_and_one_time_initialization_only",
        "teacher_usage": "none_during_online_learning",
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
    parser.add_argument("--test-count", type=int, default=20)
    parser.add_argument("--test-seed", type=int, default=20260718)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(run(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

