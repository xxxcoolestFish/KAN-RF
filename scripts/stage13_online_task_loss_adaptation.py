"""Full pipeline with task-level online adaptation of the physics branch.

The one-time operator mapper initializes the decision network.  After
deployment, the cognitive model learns from real transitions and only a
separate runtime physics residual is updated.  The online decision loss is a
multi-step task cost through the cognitive predictor, never an action-label
loss from an MPC teacher.
"""

from __future__ import annotations

import argparse
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from physics_transfer.multifactor_data import _random_states
from physics_transfer.operator_hypernetwork import OperatorMappedDecision
from physics_transfer.separated_decision import SeparatedPhysicsDecision
from physics_transfer.transition_data import sample_transition_sequence_batch
from physics_transfer.variants import step
from scripts.stage2_lowrank_gap_calibration import HELDOUT
from scripts.stage6_single_env_adaptation import PRETRAIN_FACTOR, pretrain
from scripts.stage7_single_env_decision_adaptation import operator_query
from scripts.stage10_operator_parameter_mapping import train_mapper, train_state_base


OPERATOR_DIM = 54
HIDDEN_DIM = 24


def tip_height(state):
    theta1 = torch.atan2(state[:, 1], state[:, 0])
    theta12 = theta1 + torch.atan2(state[:, 3], state[:, 2])
    return -torch.cos(theta1) - torch.cos(theta12)


def task_cost(state, action):
    height_error = F.relu(1.0 - tip_height(state))
    velocity = 0.05 * (state[:, 4].square() + state[:, 5].square())
    effort = 0.01 * action[:, 0].square()
    return height_error.square() + velocity + effort


class RuntimeTaskDecision(nn.Module):
    """Initialized decision plus a separately trainable runtime residual."""

    def __init__(self, initialized: SeparatedPhysicsDecision):
        super().__init__()
        self.task_trunk = initialized.task_trunk
        self.task_head = initialized.task_head
        self.register_buffer("base_basis", initialized.physics_basis.detach().clone())
        self.runtime_residual = nn.Sequential(
            nn.Linear(6 + OPERATOR_DIM, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )
        nn.init.zeros_(self.runtime_residual[-1].weight)
        nn.init.zeros_(self.runtime_residual[-1].bias)
        for parameter in self.task_trunk.parameters():
            parameter.requires_grad = False
        for parameter in self.task_head.parameters():
            parameter.requires_grad = False

    def forward(self, state, operator):
        hidden = torch.tanh(self.task_trunk(state))
        task_logits = self.task_head(hidden)
        physics_logits = torch.einsum("pah,bh->ba", self.base_basis, hidden)
        residual = 0.20 * torch.tanh(
            self.runtime_residual(torch.cat([state, operator], dim=-1))
        )
        logits = task_logits + physics_logits + residual
        return {"action": torch.tanh(logits), "residual": residual, "logits": logits}


def set_cognitive_grad(cognitive, enabled):
    for parameter in cognitive.parameters():
        parameter.requires_grad = enabled


def rollout_task_loss(cognitive, decision, state, latent, horizon):
    current = state
    costs, actions = [], []
    for _ in range(horizon):
        operator = operator_query(cognitive, current, latent).detach()
        output = decision(current, operator)
        action = output["action"]
        predicted_next = cognitive.predict_next(current, action, latent)
        costs.append(task_cost(predicted_next, action))
        actions.append(action)
        current = predicted_next
    stack = torch.stack(costs, dim=1)
    action_stack = torch.stack(actions, dim=1)
    smooth = (action_stack[:, 1:] - action_stack[:, :-1]).square().mean()
    residual_norm = action_stack.new_tensor(0.0)
    if hasattr(decision, "runtime_residual"):
        residual_norm = sum(parameter.square().mean() for parameter in decision.runtime_residual.parameters())
    loss = stack[:, -1].mean() + 0.25 * stack.mean() + 0.02 * smooth + 1e-3 * residual_norm
    return loss, {
        "task_loss": loss.item(),
        "mean_rollout_cost": stack.mean().item(),
        "smooth": smooth.item(),
    }


def initial_operator(cognitive, sequence_steps, seed):
    torch.manual_seed(seed + 7000)
    codes = []
    with torch.no_grad():
        for _ in range(8):
            batch = sample_transition_sequence_batch(32, sequence_steps, PRETRAIN_FACTOR)
            output = cognitive.forward_sequence(batch["state"], batch["action"], batch["next_state"])
            index = sequence_steps // 2
            state = batch["state"][:, index]
            latent = output["pre_latents"][:, index]
            codes.append(operator_query(cognitive, state, latent))
    return torch.cat(codes).mean(dim=0, keepdim=True)


def initialize_decision(cognitive, sequence_steps, base_steps, mapper_steps, seed):
    base = train_state_base(cognitive, base_steps, sequence_steps, seed)
    mapped = train_mapper(cognitive, base, mapper_steps, sequence_steps, seed, rank=1)
    q0 = initial_operator(cognitive, sequence_steps, seed)
    initialized = SeparatedPhysicsDecision(6, 1, OPERATOR_DIM, hidden_dim=HIDDEN_DIM, n_prototypes=8)
    initialized.task_trunk.load_state_dict(mapped.task_trunk.state_dict())
    initialized.task_head.load_state_dict(mapped.task_head.state_dict())
    with torch.no_grad():
        initialized.physics_basis.copy_(mapped.base_basis + mapped.mapper(q0).mean(dim=0))
    return initialized


def pretrain_task_loss(cognitive, decision, steps, sequence_steps, horizon, batch_size, seed):
    set_cognitive_grad(cognitive, False)
    optimizer = torch.optim.Adam(decision.parameters(), lr=2e-3)
    first = last = None
    for index in range(steps):
        batch = sample_transition_sequence_batch(batch_size, sequence_steps, PRETRAIN_FACTOR)
        with torch.no_grad():
            output = cognitive.forward_sequence(batch["state"], batch["action"], batch["next_state"])
            start = sequence_steps // 2
            state = batch["state"][:, start]
            latent = output["pre_latents"][:, start]
        loss, metrics = rollout_task_loss(cognitive, decision, state, latent, horizon)
        if first is None: first = metrics
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(decision.parameters(), 5.0)
        optimizer.step(); last = metrics
    set_cognitive_grad(cognitive, True)
    return {"first": first, "last": last}


def online_episode(cognitive, decision, cognitive_optimizer, decision_optimizer,
                   factor, steps, horizon, seed):
    torch.manual_seed(seed)
    state = _random_states(1)
    latent = cognitive.initial_latent(1)
    factor_tensor = torch.tensor(factor).repeat(1, 1)
    heights, prediction_errors, update_losses, residuals = [], [], [], []
    for _ in range(steps):
        with torch.no_grad():
            operator = operator_query(cognitive, state, latent)
        decision_output = decision(state, operator)
        action = decision_output["action"]
        next_state = step(
            state, action.detach(), factor_tensor[:, 0], factor_tensor[:, 1],
            factor_tensor[:, 2], factor_tensor[:, 3],
        )

        # Update the cognitive predictor on the real transition.
        prediction = cognitive.predict_next(state, action.detach(), latent)
        cognitive_loss = F.smooth_l1_loss(prediction, next_state)
        cognitive_optimizer.zero_grad(); cognitive_loss.backward()
        torch.nn.utils.clip_grad_norm_(cognitive.parameters(), 5.0)
        cognitive_optimizer.step()
        with torch.no_grad():
            latent = cognitive.observe_transition(state, action.detach(), next_state, latent).detach()

        # Update only the runtime physics branch using the task loss.
        set_cognitive_grad(cognitive, False)
        online_loss, metrics = rollout_task_loss(cognitive, decision, next_state, latent, horizon)
        decision_optimizer.zero_grad(); online_loss.backward()
        torch.nn.utils.clip_grad_norm_(decision.runtime_residual.parameters(), 5.0)
        decision_optimizer.step()
        set_cognitive_grad(cognitive, True)

        with torch.no_grad():
            heights.append(tip_height(next_state).item())
            prediction_errors.append(F.mse_loss(prediction, next_state).item())
            update_losses.append(metrics["task_loss"])
            residuals.append(decision_output["residual"].norm().item())
            state = next_state.detach()
    return {
        "success": max(heights) >= 1.0,
        "max_height": max(heights),
        "final_height": heights[-1],
        "mean_prediction_mse": sum(prediction_errors) / len(prediction_errors),
        "first_update_loss": sum(update_losses[:8]) / min(8, len(update_losses)),
        "last_update_loss": sum(update_losses[-8:]) / min(8, len(update_losses)),
        "mean_residual": sum(residuals) / len(residuals),
    }


def run_phase(cognitive, decision, factor, args, seed_offset):
    cognitive_optimizer = torch.optim.Adam(cognitive.parameters(), lr=args.cognitive_online_lr)
    decision_optimizer = torch.optim.Adam(decision.runtime_residual.parameters(), lr=args.decision_online_lr)
    episodes = []
    for episode in range(args.episodes):
        episodes.append(online_episode(
            cognitive, decision, cognitive_optimizer, decision_optimizer,
            factor, args.rollout_steps, args.online_horizon,
            args.seed + seed_offset + episode,
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
    parser.add_argument("--rollout-steps", type=int, default=64)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--cognitive-online-lr", type=float, default=2e-4)
    parser.add_argument("--decision-online-lr", type=float, default=2e-3)
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
    same = run_phase(cognitive, decision, PRETRAIN_FACTOR[0], args, 1000)
    changed = run_phase(cognitive, decision, HELDOUT[0], args, 3000)
    print(json.dumps({
        "pretrain_factor": PRETRAIN_FACTOR[0],
        "changed_factor": HELDOUT[0],
        "task_pretrain": task_pretrain,
        "same_environment": summarize(same),
        "changed_environment": summarize(changed),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
