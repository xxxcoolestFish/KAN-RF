"""Validate a task-level decision loss through the cognitive model rollout.

This experiment deliberately does not distill an MPC action.  The cognitive
network is frozen as a differentiable dynamics predictor; only the decision
network is trained by the task cost of its imagined trajectory.
"""

from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F

from physics_transfer.multifactor_data import _random_states
from physics_transfer.separated_decision import SeparatedPhysicsDecision
from physics_transfer.streaming_cognitive import StreamingCognitiveWorldModel
from physics_transfer.transition_data import sample_transition_sequence_batch
from physics_transfer.variants import step
from scripts.stage2_lowrank_gap_calibration import HELDOUT
from scripts.stage6_single_env_adaptation import PRETRAIN_FACTOR, pretrain
from scripts.stage7_single_env_decision_adaptation import operator_query


OPERATOR_DIM = 54
HIDDEN_DIM = 24


def tip_height(state: torch.Tensor) -> torch.Tensor:
    theta1 = torch.atan2(state[:, 1], state[:, 0])
    theta12 = theta1 + torch.atan2(state[:, 3], state[:, 2])
    return -torch.cos(theta1) - torch.cos(theta12)


def task_cost(state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """Task-level Acrobot cost: reach height 1 with low angular velocity."""
    height_error = F.relu(1.0 - tip_height(state))
    velocity_cost = 0.05 * (state[:, 4].square() + state[:, 5].square())
    effort_cost = 0.01 * action[:, 0].square()
    return height_error.square() + velocity_cost + effort_cost


def cognitive_pretrain(steps: int, sequence_steps: int, seed: int):
    return pretrain(steps, sequence_steps, seed)


def rollout_loss(cognitive, decision, batch, horizon: int):
    """Differentiate task loss through imagined cognitive dynamics."""
    with torch.no_grad():
        output = cognitive.forward_sequence(
            batch["state"], batch["action"], batch["next_state"]
        )
        start = min(batch["state"].shape[1] - 1, batch["state"].shape[1] // 2)
        current = batch["state"][:, start].detach()
        latent = output["pre_latents"][:, start].detach()

    costs, actions = [], []
    for _ in range(horizon):
        # q is the physical context supplied by the frozen cognitive model.
        operator = operator_query(cognitive, current, latent).detach()
        decision_output = decision(current, operator)
        action = decision_output["action"]
        # Keep gradients through the action and predicted state, but never
        # update cognitive parameters during decision training.
        predicted_next = cognitive.predict_next(current, action, latent)
        costs.append(task_cost(predicted_next, action))
        actions.append(action)
        current = predicted_next

    action_stack = torch.stack(actions, dim=1)
    cost_stack = torch.stack(costs, dim=1)
    smooth = (action_stack[:, 1:] - action_stack[:, :-1]).square().mean()
    task = cost_stack[:, -1].mean() + 0.25 * cost_stack.mean()
    regularization = 1e-3 * decision.physics_basis.square().mean()
    loss = task + 0.02 * smooth + regularization
    return loss, {
        "loss": loss.item(),
        "task": task.item(),
        "smooth": smooth.item(),
        "mean_rollout_cost": cost_stack.mean().item(),
    }


def train_decision(cognitive, steps: int, sequence_steps: int,
                   horizon: int, batch_size: int, seed: int):
    torch.manual_seed(seed + 12000)
    decision = SeparatedPhysicsDecision(
        6, 1, OPERATOR_DIM, hidden_dim=HIDDEN_DIM, n_prototypes=8
    )
    cognitive.eval()
    for parameter in cognitive.parameters():
        parameter.requires_grad = False
    optimizer = torch.optim.Adam(decision.parameters(), lr=2e-3)
    first, last = None, None
    for index in range(steps):
        batch = sample_transition_sequence_batch(
            batch_size, sequence_steps, PRETRAIN_FACTOR
        )
        loss, metrics = rollout_loss(cognitive, decision, batch, horizon)
        if first is None:
            first = metrics
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(decision.parameters(), 5.0)
        optimizer.step()
        last = metrics
    return decision, {"first": first, "last": last}


def evaluate(cognitive, decision, factor, episodes: int, rollout_steps: int,
             seed: int):
    cognitive.eval(); decision.eval()
    successes, min_heights, final_heights, prediction_errors = 0, [], [], []
    for episode in range(episodes):
        torch.manual_seed(seed + episode)
        state = _random_states(1)
        latent = cognitive.initial_latent(1)
        heights, errors = [], []
        factor_tensor = torch.tensor(factor).repeat(1, 1)
        for _ in range(rollout_steps):
            with torch.no_grad():
                operator = operator_query(cognitive, state, latent)
                output = decision(state, operator)
                action = output["action"]
                target = step(
                    state, action, factor_tensor[:, 0], factor_tensor[:, 1],
                    factor_tensor[:, 2], factor_tensor[:, 3],
                )
                predicted = cognitive.predict_next(state, action, latent)
                errors.append(F.mse_loss(predicted, target).item())
                heights.append(tip_height(target).item())
                latent = cognitive.observe_transition(
                    state, action, target, latent
                ).detach()
                state = target
        best = max(heights)
        min_heights.append(best)
        final_heights.append(heights[-1])
        prediction_errors.extend(errors)
        successes += best >= 1.0
    return {
        "factor": factor,
        "success_rate": successes / max(episodes, 1),
        "success_count": successes,
        "mean_max_height": sum(min_heights) / len(min_heights),
        "mean_final_height": sum(final_heights) / len(final_heights),
        "mean_true_prediction_mse": sum(prediction_errors) / len(prediction_errors),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cognitive-steps", type=int, default=200)
    parser.add_argument("--decision-steps", type=int, default=200)
    parser.add_argument("--sequence-steps", type=int, default=16)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--rollout-steps", type=int, default=64)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cognitive = cognitive_pretrain(args.cognitive_steps, args.sequence_steps, args.seed)
    decision, train_metrics = train_decision(
        cognitive, args.decision_steps, args.sequence_steps,
        args.horizon, args.batch_size, args.seed,
    )
    results = [
        evaluate(cognitive, decision, PRETRAIN_FACTOR[0], args.episodes,
                 args.rollout_steps, args.seed + 3000),
        evaluate(cognitive, decision, HELDOUT[0], args.episodes,
                 args.rollout_steps, args.seed + 6000),
    ]
    print(json.dumps({
        "pretrain_factor": PRETRAIN_FACTOR[0],
        "loss": train_metrics,
        "results": results,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
