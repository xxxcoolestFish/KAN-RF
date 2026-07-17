"""Stabilize Stage 13 online task-loss adaptation.

This version keeps the same cognitive/decision architecture and adds only
update safeguards: a short transition-context replay, prediction-confidence
gating, and projection of the runtime physics residual into a trust region.
"""

from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F

from scripts.stage2_lowrank_gap_calibration import HELDOUT
from scripts.stage6_single_env_adaptation import PRETRAIN_FACTOR, pretrain
from scripts.stage13_online_task_loss_adaptation import (
    RuntimeTaskDecision,
    _random_states,
    initialize_decision,
    pretrain_task_loss,
    rollout_task_loss,
    set_cognitive_grad,
    step,
    tip_height,
)


class ContextReplay:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.items = []

    def add(self, state, latent):
        self.items.append((state.detach().clone(), latent.detach().clone()))
        if len(self.items) > self.capacity:
            self.items.pop(0)

    def sample(self, batch_size: int):
        if not self.items:
            return None
        count = min(batch_size, len(self.items))
        index = torch.randperm(len(self.items))[:count]
        states = torch.cat([self.items[int(i)][0] for i in index], dim=0)
        latents = torch.cat([self.items[int(i)][1] for i in index], dim=0)
        return states, latents


def project_runtime(decision, radius: float):
    parameters = list(decision.runtime_residual.parameters())
    with torch.no_grad():
        norm = torch.linalg.vector_norm(torch.cat([p.flatten() for p in parameters]))
        if norm > radius:
            scale = radius / (norm + 1e-8)
            for parameter in parameters:
                parameter.mul_(scale)
        return min(norm.item(), radius)


def stable_episode(cognitive, decision, cognitive_optimizer, decision_optimizer,
                   replay, factor, args, seed):
    torch.manual_seed(seed)
    state = _random_states(1)
    latent = cognitive.initial_latent(1)
    factor_tensor = torch.tensor(factor).repeat(1, 1)
    heights, prediction_errors, losses, gates, residuals = [], [], [], [], []

    for _ in range(args.rollout_steps):
        with torch.no_grad():
            from scripts.stage7_single_env_decision_adaptation import operator_query
            operator = operator_query(cognitive, state, latent)
        output = decision(state, operator)
        action = output["action"]
        next_state = step(
            state, action.detach(), factor_tensor[:, 0], factor_tensor[:, 1],
            factor_tensor[:, 2], factor_tensor[:, 3],
        )

        prediction = cognitive.predict_next(state, action.detach(), latent)
        cognitive_loss = F.smooth_l1_loss(prediction, next_state)
        cognitive_optimizer.zero_grad(); cognitive_loss.backward()
        torch.nn.utils.clip_grad_norm_(cognitive.parameters(), 5.0)
        cognitive_optimizer.step()
        with torch.no_grad():
            latent = cognitive.observe_transition(
                state, action.detach(), next_state, latent
            ).detach()
            replay.add(next_state, latent)

        set_cognitive_grad(cognitive, False)
        current_loss, current_metrics = rollout_task_loss(
            cognitive, decision, next_state, latent, args.online_horizon
        )
        sampled = replay.sample(args.replay_batch)
        if sampled is not None:
            replay_states, replay_latents = sampled
            replay_loss, _ = rollout_task_loss(
                cognitive, decision, replay_states, replay_latents, args.online_horizon
            )
            total_loss = (1.0 - args.replay_weight) * current_loss + args.replay_weight * replay_loss
        else:
            total_loss = current_loss

        error = cognitive_loss.item()
        gate = min(1.0, max(0.0, torch.exp(
            torch.tensor(-error / max(args.confidence_scale, 1e-8))
        ).item()))
        accepted = error <= args.confidence_threshold
        if accepted:
            decision_optimizer.zero_grad()
            (gate * total_loss).backward()
            torch.nn.utils.clip_grad_norm_(decision.runtime_residual.parameters(), 5.0)
            decision_optimizer.step()
            residual_norm = project_runtime(decision, args.trust_radius)
        else:
            residual_norm = project_runtime(decision, args.trust_radius)
        set_cognitive_grad(cognitive, True)

        with torch.no_grad():
            heights.append(tip_height(next_state).item())
            prediction_errors.append(F.mse_loss(prediction, next_state).item())
            losses.append(current_metrics["task_loss"])
            gates.append(gate if accepted else 0.0)
            residuals.append(residual_norm)
            state = next_state.detach()

    return {
        "success": max(heights) >= 1.0,
        "max_height": max(heights),
        "final_height": heights[-1],
        "mean_prediction_mse": sum(prediction_errors) / len(prediction_errors),
        "first_update_loss": sum(losses[:8]) / min(8, len(losses)),
        "last_update_loss": sum(losses[-8:]) / min(8, len(losses)),
        "mean_gate": sum(gates) / len(gates),
        "mean_residual_norm": sum(residuals) / len(residuals),
    }


def run_phase(cognitive, decision, factor, args, seed_offset,
              cognitive_optimizer, decision_optimizer, replay):
    episodes = []
    for episode in range(args.episodes):
        episodes.append(stable_episode(
            cognitive, decision, cognitive_optimizer, decision_optimizer,
            replay, factor, args, args.seed + seed_offset + episode,
        ))
    return episodes


def summarize(episodes):
    return {
        "success_count": sum(item["success"] for item in episodes),
        "success_rate": sum(item["success"] for item in episodes) / len(episodes),
        "first_success_rate": sum(item["success"] for item in episodes[:3]) / min(3, len(episodes)),
        "last_success_rate": sum(item["success"] for item in episodes[-3:]) / min(3, len(episodes)),
        "mean_max_height": sum(item["max_height"] for item in episodes) / len(episodes),
        "mean_prediction_mse": sum(item["mean_prediction_mse"] for item in episodes) / len(episodes),
        "mean_first_update_loss": sum(item["first_update_loss"] for item in episodes) / len(episodes),
        "mean_last_update_loss": sum(item["last_update_loss"] for item in episodes) / len(episodes),
        "mean_gate": sum(item["mean_gate"] for item in episodes) / len(episodes),
        "mean_residual_norm": sum(item["mean_residual_norm"] for item in episodes) / len(episodes),
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
    parser.add_argument("--replay-batch", type=int, default=8)
    parser.add_argument("--replay-capacity", type=int, default=32)
    parser.add_argument("--replay-weight", type=float, default=0.5)
    parser.add_argument("--trust-radius", type=float, default=0.05)
    parser.add_argument("--confidence-threshold", type=float, default=0.05)
    parser.add_argument("--confidence-scale", type=float, default=0.01)
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
    cognitive_optimizer = torch.optim.Adam(cognitive.parameters(), lr=args.cognitive_online_lr)
    decision_optimizer = torch.optim.Adam(
        decision.runtime_residual.parameters(), lr=args.decision_online_lr
    )
    replay = ContextReplay(args.replay_capacity)
    same = run_phase(
        cognitive, decision, PRETRAIN_FACTOR[0], args, 1000,
        cognitive_optimizer, decision_optimizer, replay,
    )
    changed = run_phase(
        cognitive, decision, HELDOUT[0], args, 3000,
        cognitive_optimizer, decision_optimizer, replay,
    )
    print(json.dumps({
        "pretrain_factor": PRETRAIN_FACTOR[0],
        "changed_factor": HELDOUT[0],
        "stabilization": {
            "trust_radius": args.trust_radius,
            "confidence_threshold": args.confidence_threshold,
            "replay_weight": args.replay_weight,
        },
        "task_pretrain": task_pretrain,
        "same_environment": summarize(same),
        "changed_environment": summarize(changed),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
