"""Use real episode outcomes to train the runtime physics branch.

The cognitive model remains a predictor.  After each real episode, actions
are replayed with return-to-go weights (advantage-weighted regression), while
the cognitive model rollout loss is only an auxiliary regularizer.
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
    pretrain_task_loss,
    rollout_task_loss,
    set_cognitive_grad,
    tip_height,
)
from scripts.stage2_lowrank_gap_calibration import HELDOUT
from scripts.stage6_single_env_adaptation import PRETRAIN_FACTOR, pretrain
from scripts.stage7_single_env_decision_adaptation import operator_query


def reward(state, action):
    height = tip_height(state)
    velocity = 0.05 * (state[:, 4].square() + state[:, 5].square())
    effort = 0.01 * action[:, 0].square()
    success_bonus = 5.0 * (height >= 1.0).float()
    return height - velocity - effort + success_bonus


class OutcomeReplay:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.items = []

    def add_episode(self, trajectory, gamma: float):
        returns = []
        running = 0.0
        for item in reversed(trajectory):
            running = float(item["reward"]) + gamma * running
            returns.append(running)
        returns.reverse()
        for item, value in zip(trajectory, returns):
            self.items.append((
                item["state"].detach().clone(),
                item["operator"].detach().clone(),
                item["latent"].detach().clone(),
                item["action"].detach().clone(),
                float(value),
            ))
        if len(self.items) > self.capacity:
            self.items = self.items[-self.capacity:]

    def sample(self, batch_size: int):
        if not self.items:
            return None
        count = min(batch_size, len(self.items))
        index = torch.randperm(len(self.items))[:count]
        selected = [self.items[int(i)] for i in index]
        states = torch.cat([item[0] for item in selected], dim=0)
        operators = torch.cat([item[1] for item in selected], dim=0)
        latents = torch.cat([item[2] for item in selected], dim=0)
        actions = torch.cat([item[3] for item in selected], dim=0)
        returns = torch.tensor([item[4] for item in selected], dtype=states.dtype).view(-1, 1)
        return states, operators, latents, actions, returns


def project_runtime(decision, references, radius: float):
    parameters = list(decision.runtime_residual.parameters())
    with torch.no_grad():
        difference = torch.cat([
            (parameter - reference).flatten()
            for parameter, reference in zip(parameters, references)
        ])
        norm = torch.linalg.vector_norm(difference)
        if norm > radius:
            scale = radius / (norm + 1e-8)
            for parameter, reference in zip(parameters, references):
                parameter.copy_(reference + scale * (parameter - reference))
        return min(norm.item(), radius)


def update_from_outcomes(cognitive, decision, optimizer, replay, references, args):
    losses, weights_mean = [], []
    for _ in range(args.updates_per_episode):
        sampled = replay.sample(args.replay_batch)
        if sampled is None:
            continue
        states, operators, latents, actions, returns = sampled
        centered = returns - returns.median()
        weights = torch.exp((centered / args.return_temperature).clamp(-3.0, 3.0)).detach()
        prediction = decision(states, operators)["action"]
        real_loss = (weights * (prediction - actions).square()).mean()

        set_cognitive_grad(cognitive, False)
        model_loss, _ = rollout_task_loss(
            cognitive, decision, states, latents, args.online_horizon
        )
        total_loss = real_loss + args.model_loss_weight * model_loss
        optimizer.zero_grad(); total_loss.backward()
        torch.nn.utils.clip_grad_norm_(decision.runtime_residual.parameters(), 5.0)
        optimizer.step()
        residual_norm = project_runtime(decision, references, args.trust_radius)
        set_cognitive_grad(cognitive, True)
        losses.append({
            "real_loss": real_loss.item(),
            "model_loss": model_loss.item(),
            "total_loss": total_loss.item(),
            "residual_norm": residual_norm,
        })
        weights_mean.append(weights.mean().item())
    return losses, weights_mean


def run_episode(cognitive, decision, cognitive_optimizer, decision_optimizer,
                replay, references, factor, args, seed):
    torch.manual_seed(seed)
    state = _random_states(1)
    latent = cognitive.initial_latent(1)
    factor_tensor = torch.tensor(factor).repeat(1, 1)
    trajectory, heights, prediction_errors = [], [], []
    for _ in range(args.rollout_steps):
        with torch.no_grad():
            operator = operator_query(cognitive, state, latent)
        output = decision(state, operator)
        action = output["action"]
        next_state = step(
            state, action.detach(), factor_tensor[:, 0], factor_tensor[:, 1],
            factor_tensor[:, 2], factor_tensor[:, 3],
        )
        current_reward = reward(next_state, action.detach()).item()
        prediction = cognitive.predict_next(state, action.detach(), latent)
        cognitive_loss = F.smooth_l1_loss(prediction, next_state)
        cognitive_optimizer.zero_grad(); cognitive_loss.backward()
        torch.nn.utils.clip_grad_norm_(cognitive.parameters(), 5.0)
        cognitive_optimizer.step()
        with torch.no_grad():
            next_latent = cognitive.observe_transition(
                state, action.detach(), next_state, latent
            ).detach()
            trajectory.append({
                "state": state,
                "operator": operator,
                "latent": latent,
                "action": action.detach(),
                "reward": current_reward,
            })
            heights.append(tip_height(next_state).item())
            prediction_errors.append(F.mse_loss(prediction, next_state).item())
            state, latent = next_state.detach(), next_latent

    replay.add_episode(trajectory, args.gamma)
    losses, weights = update_from_outcomes(
        cognitive, decision, decision_optimizer, replay, references, args
    )
    return {
        "success": max(heights) >= 1.0,
        "max_height": max(heights),
        "mean_return": sum(item["reward"] for item in trajectory) / len(trajectory),
        "mean_prediction_mse": sum(prediction_errors) / len(prediction_errors),
        "real_loss": sum(item["real_loss"] for item in losses) / max(len(losses), 1),
        "model_loss": sum(item["model_loss"] for item in losses) / max(len(losses), 1),
        "residual_norm": losses[-1]["residual_norm"] if losses else 0.0,
        "mean_weight": sum(weights) / max(len(weights), 1),
    }


def run_phase(cognitive, decision, factor, args, seed_offset,
              cognitive_optimizer, decision_optimizer, replay, references):
    episodes = []
    for episode in range(args.episodes):
        episodes.append(run_episode(
            cognitive, decision, cognitive_optimizer, decision_optimizer,
            replay, references, factor, args, args.seed + seed_offset + episode,
        ))
    return episodes


def summarize(episodes):
    return {
        "success_count": sum(item["success"] for item in episodes),
        "success_rate": sum(item["success"] for item in episodes) / len(episodes),
        "first_success_rate": sum(item["success"] for item in episodes[:3]) / min(3, len(episodes)),
        "last_success_rate": sum(item["success"] for item in episodes[-3:]) / min(3, len(episodes)),
        "mean_max_height": sum(item["max_height"] for item in episodes) / len(episodes),
        "mean_return": sum(item["mean_return"] for item in episodes) / len(episodes),
        "mean_prediction_mse": sum(item["mean_prediction_mse"] for item in episodes) / len(episodes),
        "mean_real_loss": sum(item["real_loss"] for item in episodes) / len(episodes),
        "mean_model_loss": sum(item["model_loss"] for item in episodes) / len(episodes),
        "mean_residual_norm": sum(item["residual_norm"] for item in episodes) / len(episodes),
        "episodes": episodes,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cognitive-steps", type=int, default=100)
    parser.add_argument("--base-steps", type=int, default=100)
    parser.add_argument("--mapper-steps", type=int, default=100)
    parser.add_argument("--decision-steps", type=int, default=100)
    parser.add_argument("--sequence-steps", type=int, default=16)
    parser.add_argument("--offline-horizon", type=int, default=8)
    parser.add_argument("--online-horizon", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--replay-capacity", type=int, default=4096)
    parser.add_argument("--replay-batch", type=int, default=128)
    parser.add_argument("--updates-per-episode", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--return-temperature", type=float, default=1.0)
    parser.add_argument("--model-loss-weight", type=float, default=0.25)
    parser.add_argument("--trust-radius", type=float, default=0.05)
    parser.add_argument("--rollout-steps", type=int, default=64)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--cognitive-online-lr", type=float, default=2e-4)
    parser.add_argument("--decision-online-lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cognitive = pretrain(args.cognitive_steps, args.sequence_steps, args.seed)
    initialized = initialize_decision(
        cognitive, args.sequence_steps, args.base_steps, args.mapper_steps, args.seed
    )
    task_pretrain = pretrain_task_loss(
        cognitive, initialized, args.decision_steps, args.sequence_steps,
        args.offline_horizon, args.batch_size, args.seed,
    )
    decision = RuntimeTaskDecision(initialized)
    references = [parameter.detach().clone() for parameter in decision.runtime_residual.parameters()]
    cognitive_optimizer = torch.optim.Adam(cognitive.parameters(), lr=args.cognitive_online_lr)
    decision_optimizer = torch.optim.Adam(
        decision.runtime_residual.parameters(), lr=args.decision_online_lr
    )
    replay = OutcomeReplay(args.replay_capacity)
    same = run_phase(
        cognitive, decision, PRETRAIN_FACTOR[0], args, 1000,
        cognitive_optimizer, decision_optimizer, replay, references,
    )
    changed = run_phase(
        cognitive, decision, HELDOUT[0], args, 3000,
        cognitive_optimizer, decision_optimizer, replay, references,
    )
    print(json.dumps({
        "pretrain_factor": PRETRAIN_FACTOR[0],
        "changed_factor": HELDOUT[0],
        "task_pretrain": task_pretrain,
        "real_outcome_replay": {
            "gamma": args.gamma,
            "return_temperature": args.return_temperature,
            "model_loss_weight": args.model_loss_weight,
            "trust_radius": args.trust_radius,
        },
        "same_environment": summarize(same),
        "changed_environment": summarize(changed),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
