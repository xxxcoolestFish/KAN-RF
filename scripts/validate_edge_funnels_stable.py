"""Stable macro-episode wrapper for feedback reachability validation."""

from __future__ import annotations

import json

import torch

from scripts import validate_edge_funnels as base


def collect_macro_rollout(actor, critic, dynamics, funnels, args, seed):
    """Collect bounded macro episodes with clipped normalized progress."""
    generator = torch.Generator().manual_seed(seed)
    state, center, scale = base.reset_batch(funnels, args.num_envs, generator)
    age = torch.zeros(args.num_envs, dtype=torch.long)
    states, centers, scales, actions = [], [], [], []
    log_probs, values, rewards, terminals = [], [], [], []
    successes = 0
    for _ in range(args.rollout_horizon):
        with torch.no_grad():
            action, log_prob = actor.sample(state, center, scale)
            value = critic(state, center, scale)
            before = funnels.normalized_distance(state, center, scale)
            next_state = dynamics(state, action)
            after = funnels.normalized_distance(next_state, center, scale)
            success = after <= 1.0
            progress = (before - after).clamp(-0.25, 0.25)
            reward = args.progress_reward * progress
            reward = reward + args.success_reward * success.float()
            reward = reward - args.action_penalty * action.square().squeeze(-1)
            terminal = success | ((age + 1) >= funnels.planner.macro_steps)
        successes += int(success.sum())
        states.append(state); centers.append(center); scales.append(scale)
        actions.append(action); log_probs.append(log_prob); values.append(value)
        rewards.append(reward); terminals.append(terminal)
        reset_state, reset_center, reset_scale = base.reset_batch(
            funnels, args.num_envs, generator,
        )
        state = torch.where(terminal.unsqueeze(-1), reset_state, next_state)
        center = torch.where(terminal.unsqueeze(-1), reset_center, center)
        scale = torch.where(terminal.unsqueeze(-1), reset_scale, scale)
        age = torch.where(terminal, torch.zeros_like(age), age + 1)

    with torch.no_grad():
        last_value = critic(state, center, scale)
    states = torch.stack(states); centers = torch.stack(centers)
    scales = torch.stack(scales); actions = torch.stack(actions)
    log_probs = torch.stack(log_probs); values = torch.stack(values)
    rewards = torch.stack(rewards); terminals = torch.stack(terminals)
    advantage = torch.zeros_like(rewards)
    gae = torch.zeros(args.num_envs)
    next_value = last_value
    for index in reversed(range(args.rollout_horizon)):
        nonterminal = 1.0 - terminals[index].float()
        delta = rewards[index] + args.gamma * next_value * nonterminal - values[index]
        gae = delta + args.gamma * args.gae_lambda * nonterminal * gae
        advantage[index] = gae
        next_value = values[index]
    returns = advantage + values
    return base.Rollout(
        states.reshape(-1, 6), centers.reshape(-1, 6), scales.reshape(-1, 6),
        actions.reshape(-1, 1), log_probs.reshape(-1),
        advantage.reshape(-1), returns.reshape(-1),
    ), {
        "collected_edge_completions": successes,
        "mean_reward": float(rewards.mean()),
    }


def main():
    args = base.parse_args()
    base.collect_rollout = collect_macro_rollout
    output = {
        "experiment": "OracleFeedbackReachabilityFunnelValidationStable",
        "config": vars(args),
        "result": base.run(args),
    }
    text = json.dumps(output, ensure_ascii=False, indent=2)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
