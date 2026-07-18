"""Decision-only continual learning after one-time cognitive initialization.

The cognitive network is used only to pretrain and transfer an initialization.
It is not queried during deployment.  A fixed initialization operator is fed
to the decision network, and only the runtime physics residual learns from
real episode outcomes in the original training environment.
"""

from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F

from physics_transfer.multifactor_data import _random_states
from physics_transfer.variants import step
from scripts.stage13_online_task_loss_adaptation import (
    RuntimeTaskDecision,
    initialize_decision,
    tip_height,
)
from scripts.stage15_real_outcome_replay import reward
from scripts.stage15_real_outcome_replay import project_runtime
from scripts.stage15_real_outcome_replay import OutcomeReplay
from scripts.stage15_real_outcome_replay import pretrain, PRETRAIN_FACTOR


def update_from_outcomes(decision, optimizer, replay, references, args):
    losses, weights_seen = [], []
    for _ in range(args.updates_per_episode):
        sampled = replay.sample(args.replay_batch)
        if sampled is None:
            continue
        states, operators, _, actions, returns = sampled
        centered = returns - returns.median()
        weights = torch.exp((centered / args.return_temperature).clamp(-3.0, 3.0)).detach()
        prediction = decision(states, operators)["action"]
        loss = (weights * (prediction - actions).square()).mean()
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(decision.runtime_residual.parameters(), 5.0)
        optimizer.step()
        norm = project_runtime(decision, references, args.trust_radius)
        losses.append(loss.item()); weights_seen.append(weights.mean().item())
    return sum(losses) / max(len(losses), 1), sum(weights_seen) / max(len(weights_seen), 1), norm if losses else 0.0


def run_episode(decision, q_const, replay, optimizer, references,
                factor, args, seed, training):
    torch.manual_seed(seed)
    state = _random_states(1)
    factor_tensor = torch.tensor(factor).repeat(1, 1)
    trajectory, heights, returns = [], [], []
    for _ in range(args.rollout_steps):
        operator = q_const.expand(state.shape[0], -1)
        policy_action = decision(state, operator)["action"]
        if training:
            action = (policy_action + args.exploration_noise * torch.randn_like(policy_action)).clamp(-1.0, 1.0)
        else:
            action = policy_action
        next_state = step(
            state, action.detach(), factor_tensor[:, 0], factor_tensor[:, 1],
            factor_tensor[:, 2], factor_tensor[:, 3],
        )
        current_reward = reward(next_state, action.detach()).item()
        trajectory.append({
            "state": state.detach(),
            "operator": operator.detach(),
            "latent": torch.zeros(1, 1),
            "action": action.detach(),
            "reward": current_reward,
        })
        heights.append(tip_height(next_state).item())
        returns.append(current_reward)
        state = next_state.detach()

    if training:
        replay.add_episode(trajectory, args.gamma)
        loss, weight, norm = update_from_outcomes(decision, optimizer, replay, references, args)
    else:
        loss, weight, norm = 0.0, 0.0, 0.0
    return {
        "success": max(heights) >= 1.0,
        "max_height": max(heights),
        "mean_return": sum(returns) / len(returns),
        "replay_loss": loss,
        "mean_weight": weight,
        "residual_norm": norm,
    }


def summarize(episodes):
    return {
        "success_count": sum(item["success"] for item in episodes),
        "success_rate": sum(item["success"] for item in episodes) / len(episodes),
        "first_success_rate": sum(item["success"] for item in episodes[:3]) / min(3, len(episodes)),
        "last_success_rate": sum(item["success"] for item in episodes[-3:]) / min(3, len(episodes)),
        "mean_max_height": sum(item["max_height"] for item in episodes) / len(episodes),
        "mean_return": sum(item["mean_return"] for item in episodes) / len(episodes),
        "mean_replay_loss": sum(item["replay_loss"] for item in episodes) / len(episodes),
        "mean_residual_norm": sum(item["residual_norm"] for item in episodes) / len(episodes),
        "episodes": episodes,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cognitive-steps", type=int, default=100)
    parser.add_argument("--base-steps", type=int, default=100)
    parser.add_argument("--mapper-steps", type=int, default=100)
    parser.add_argument("--sequence-steps", type=int, default=16)
    parser.add_argument("--replay-capacity", type=int, default=4096)
    parser.add_argument("--replay-batch", type=int, default=128)
    parser.add_argument("--updates-per-episode", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--return-temperature", type=float, default=1.0)
    parser.add_argument("--trust-radius", type=float, default=0.05)
    parser.add_argument("--exploration-noise", type=float, default=0.10)
    parser.add_argument("--rollout-steps", type=int, default=64)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cognitive = pretrain(args.cognitive_steps, args.sequence_steps, args.seed)
    initialized = initialize_decision(
        cognitive, args.sequence_steps, args.base_steps, args.mapper_steps, args.seed
    )
    # Extract a fixed operator once, then discard the cognitive network.
    from scripts.stage13_online_task_loss_adaptation import initial_operator
    q_const = initial_operator(cognitive, args.sequence_steps, args.seed).detach()
    decision = RuntimeTaskDecision(initialized)
    references = [parameter.detach().clone() for parameter in decision.runtime_residual.parameters()]
    optimizer = torch.optim.Adam(decision.runtime_residual.parameters(), lr=5e-4)
    replay = OutcomeReplay(args.replay_capacity)
    training = []
    for episode in range(args.episodes):
        training.append(run_episode(
            decision, q_const, replay, optimizer, references,
            PRETRAIN_FACTOR[0], args, args.seed + 1000 + episode, True,
        ))
    evaluation = []
    for episode in range(args.episodes):
        evaluation.append(run_episode(
            decision, q_const, replay, optimizer, references,
            PRETRAIN_FACTOR[0], args, args.seed + 5000 + episode, False,
        ))
    print(json.dumps({
        "environment": PRETRAIN_FACTOR[0],
        "cognitive_usage": "pretraining_and_one_time_initialization_only",
        "training": summarize(training),
        "final_no_exploration_evaluation": summarize(evaluation),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
