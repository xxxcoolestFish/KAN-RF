"""Validate multi-step cognitive training and terminal-value energy control.

This experiment keeps the embedded architecture from stage 22, but borrows two
standard model-based RL stabilizers:

1. cognitive training uses a curriculum of free-running model rollouts;
2. decision training uses a short model horizon plus a TD-trained terminal value.

The action is still obtained by differentiating through the cognitive ProtoKAN.
No action teacher or MPC action labels are used.
"""

from __future__ import annotations

import argparse
import copy
import json

import torch
import torch.nn.functional as F
from torch import nn

from kanrf._protokan import ProtoKAN
from physics_transfer.multifactor_data import _random_states
from physics_transfer.transition_data import (
    sample_transition_batch,
    sample_transition_sequence_batch,
)
from physics_transfer.variants import step
from scripts.stage15_real_outcome_replay import PRETRAIN_FACTOR


GOAL = torch.tensor([-1.0, 0.0, 1.0, 0.0, 0.0, 0.0])


def smooth_tip_height(state):
    c1, s1, c2, s2 = state[:, 0], state[:, 1], state[:, 2], state[:, 3]
    return -c1 - (c1 * c2 - s1 * s2)


def task_cost(state, action):
    gap = F.relu(1.0 - smooth_tip_height(state))
    velocity = 0.05 * (state[:, 4].square() + state[:, 5].square())
    effort = 0.01 * action[:, 0].square()
    return gap.square() + velocity + effort


class SimpleCognitiveKAN(nn.Module):
    def __init__(self, state_dim=6, action_dim=1, hidden_dim=32, n_prototypes=8):
        super().__init__()
        self.network = ProtoKAN(
            [state_dim + action_dim, hidden_dim, state_dim],
            n_prototypes=n_prototypes,
        )

    def forward(self, state, action):
        return self.network(torch.cat([state, action], dim=-1))


class EnergyKAN(nn.Module):
    def __init__(self, state_dim=6, goal_dim=6, hidden_dim=32, n_prototypes=8):
        super().__init__()
        self.network = ProtoKAN(
            [state_dim + goal_dim, hidden_dim, 1],
            n_prototypes=n_prototypes,
        )

    def forward(self, state, goal):
        return self.network(torch.cat([state, goal], dim=-1))


class TerminalValueKAN(nn.Module):
    def __init__(self, state_dim=6, goal_dim=6, hidden_dim=32, n_prototypes=8):
        super().__init__()
        self.network = ProtoKAN(
            [state_dim + goal_dim, hidden_dim, 1],
            n_prototypes=n_prototypes,
        )

    def forward(self, state, goal):
        return self.network(torch.cat([state, goal], dim=-1))


def _stats(losses):
    return {
        "first_loss": float(losses[0]),
        "last_loss": float(losses[-1]),
        "mean_last_20_loss": float(sum(losses[-20:]) / min(20, len(losses))),
    }


def _curriculum_horizon(index, total, max_horizon):
    fraction = index / max(1, total - 1)
    levels = [h for h in (1, 2, 4, 8) if h <= max_horizon]
    level = min(len(levels) - 1, int(fraction * len(levels)))
    return levels[level]


def pretrain_cognitive_multistep(model, steps, batch_size, max_horizon, seed):
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    losses, used_horizons = [], []
    for index in range(steps):
        horizon = _curriculum_horizon(index, steps, max_horizon)
        batch = sample_transition_sequence_batch(batch_size, horizon, PRETRAIN_FACTOR)
        current = batch["state"][:, 0]
        total = torch.zeros((), dtype=current.dtype)
        for t in range(horizon):
            prediction = model(current, batch["action"][:, t])
            target = batch["next_state"][:, t]
            weight = 1.0 / (t + 1) ** 0.5
            total = total + weight * F.smooth_l1_loss(prediction, target)
            current = prediction
        loss = total / horizon
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        losses.append(loss.item()); used_horizons.append(horizon)
    return {**_stats(losses), "final_horizon": used_horizons[-1]}


@torch.no_grad()
def model_rollout_error(model, horizon, batch_size=128):
    batch = sample_transition_sequence_batch(batch_size, horizon, PRETRAIN_FACTOR)
    current = batch["state"][:, 0]
    errors = []
    for t in range(horizon):
        current = model(current, batch["action"][:, t])
        errors.append(F.smooth_l1_loss(current, batch["next_state"][:, t]).item())
    return float(sum(errors) / len(errors))


def fit_one_step_energy(cognitive, energy, steps, batch_size, seed, goal):
    torch.manual_seed(seed + 1000)
    optimizer = torch.optim.Adam(energy.parameters(), lr=2e-3)
    losses = []
    for _ in range(steps):
        batch = sample_transition_batch(batch_size, PRETRAIN_FACTOR)
        with torch.no_grad():
            predicted = cognitive(batch["state"], batch["action"])
            target = task_cost(predicted, batch["action"]).unsqueeze(-1)
        prediction = energy(predicted, goal.expand(batch_size, -1))
        loss = F.smooth_l1_loss(prediction, target)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        losses.append(loss.item())
    return _stats(losses)


def controller_action(cognitive, energy, value, state, goal, inner_steps,
                      step_size, gamma, create_graph):
    action = torch.zeros(state.shape[0], 1, dtype=state.dtype)
    action.requires_grad_(True)
    for _ in range(inner_steps):
        predicted = cognitive(state, action)
        immediate = task_cost(predicted, action)
        terminal = value(predicted, goal.expand(state.shape[0], -1)).squeeze(-1)
        objective = immediate + gamma * terminal + 0.05 * energy(
            predicted, goal.expand(state.shape[0], -1)
        ).squeeze(-1)
        gradient = torch.autograd.grad(
            objective.sum(), action, create_graph=create_graph,
        )[0]
        action = (action - step_size * gradient).clamp(-1.0, 1.0)
    return action


def fit_value_td(cognitive, energy, value, steps, batch_size, seed, goal,
                 gamma, inner_steps, step_size):
    torch.manual_seed(seed + 1500)
    for parameter in cognitive.parameters():
        parameter.requires_grad = False
    for parameter in energy.parameters():
        parameter.requires_grad = False
    target_value = copy.deepcopy(value)
    optimizer = torch.optim.Adam(value.parameters(), lr=2e-3)
    losses = []
    for _ in range(steps):
        batch = sample_transition_batch(batch_size, PRETRAIN_FACTOR)
        with torch.enable_grad():
            action = controller_action(
                cognitive, energy, value, batch["state"], goal,
                inner_steps, step_size, gamma, create_graph=False,
            ).detach()
        next_state = batch["next_state"]
        with torch.no_grad():
            target = task_cost(next_state, action) + gamma * target_value(
                next_state, goal.expand(batch_size, -1)
            ).squeeze(-1)
        prediction = value(batch["state"], goal.expand(batch_size, -1)).squeeze(-1)
        loss = F.smooth_l1_loss(prediction, target)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        with torch.no_grad():
            for target_parameter, parameter in zip(target_value.parameters(), value.parameters()):
                target_parameter.mul_(0.95).add_(0.05 * parameter)
        losses.append(loss.item())
    return _stats(losses)


def train_energy_short_horizon(cognitive, energy, value, steps, batch_size,
                               horizon, seed, goal, gamma, inner_steps,
                               step_size):
    torch.manual_seed(seed + 2000)
    optimizer = torch.optim.Adam(energy.parameters(), lr=1e-3)
    losses = []
    for _ in range(steps):
        current = _random_states(batch_size)
        costs, actions = [], []
        for t in range(horizon):
            action = controller_action(
                cognitive, energy, value, current, goal,
                inner_steps, step_size, gamma, create_graph=True,
            )
            next_state = cognitive(current, action)
            costs.append(task_cost(next_state, action))
            actions.append(action)
            current = next_state
        discounted = torch.stack(
            [gamma ** t * cost for t, cost in enumerate(costs)], dim=1
        ).sum(dim=1)
        terminal = gamma ** horizon * value(
            current, goal.expand(batch_size, -1)
        ).squeeze(-1)
        action_stack = torch.stack(actions, dim=1)
        smooth = (action_stack[:, 1:] - action_stack[:, :-1]).square().mean()
        loss = (discounted + terminal).mean() + 0.02 * smooth
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(energy.parameters(), 5.0)
        optimizer.step()
        losses.append(loss.item())
    return _stats(losses)


def evaluate(cognitive, energy, value, states, factor, goal, steps, gamma,
             inner_steps, step_size):
    current = states.detach().clone()
    factor_tensor = torch.tensor(factor, dtype=current.dtype).view(1, 4).expand(current.shape[0], -1)
    maxima = torch.full((current.shape[0],), -float("inf"))
    for _ in range(steps):
        with torch.enable_grad():
            action = controller_action(
                cognitive, energy, value, current, goal,
                inner_steps, step_size, gamma, create_graph=False,
            ).detach()
        current = step(
            current, action,
            factor_tensor[:, 0], factor_tensor[:, 1],
            factor_tensor[:, 2], factor_tensor[:, 3],
        )
        maxima = torch.maximum(maxima, smooth_tip_height(current).detach())
    success = maxima >= 1.0
    return {
        "success_count": int(success.sum().item()),
        "success_rate": float(success.float().mean().item()),
        "mean_max_height": float(maxima.mean().item()),
        "max_height": maxima.tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cognitive-steps", type=int, default=300)
    parser.add_argument("--max-cognitive-horizon", type=int, default=8)
    parser.add_argument("--energy-fit-steps", type=int, default=100)
    parser.add_argument("--value-steps", type=int, default=150)
    parser.add_argument("--decision-steps", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--decision-horizon", type=int, default=4)
    parser.add_argument("--inner-steps", type=int, default=3)
    parser.add_argument("--action-step-size", type=float, default=0.25)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--rollout-steps", type=int, default=64)
    parser.add_argument("--test-count", type=int, default=20)
    parser.add_argument("--test-seed", type=int, default=20260718)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    goal = GOAL.view(1, -1)
    cognitive = SimpleCognitiveKAN()
    cognitive_fit = pretrain_cognitive_multistep(
        cognitive, args.cognitive_steps, 32,
        args.max_cognitive_horizon, args.seed,
    )
    model_errors = {
        f"horizon_{h}": model_rollout_error(cognitive, h)
        for h in (1, 2, 4, 8)
        if h <= args.max_cognitive_horizon
    }
    energy = EnergyKAN()
    value = TerminalValueKAN()
    energy_fit = fit_one_step_energy(
        cognitive, energy, args.energy_fit_steps, args.batch_size, args.seed, goal,
    )
    generator = torch.Generator().manual_seed(args.test_seed)
    test_states = _random_states(args.test_count, generator=generator)
    before_value = evaluate(
        cognitive, energy, value, test_states, PRETRAIN_FACTOR[0], goal,
        args.rollout_steps, args.gamma, args.inner_steps, args.action_step_size,
    )
    value_fit = fit_value_td(
        cognitive, energy, value, args.value_steps, args.batch_size, args.seed,
        goal, args.gamma, args.inner_steps, args.action_step_size,
    )
    before_decision = evaluate(
        cognitive, energy, value, test_states, PRETRAIN_FACTOR[0], goal,
        args.rollout_steps, args.gamma, args.inner_steps, args.action_step_size,
    )
    decision_fit = train_energy_short_horizon(
        cognitive, energy, value, args.decision_steps, args.batch_size,
        args.decision_horizon, args.seed, goal, args.gamma,
        args.inner_steps, args.action_step_size,
    )
    final = evaluate(
        cognitive, energy, value, test_states, PRETRAIN_FACTOR[0], goal,
        args.rollout_steps, args.gamma, args.inner_steps, args.action_step_size,
    )
    print(json.dumps({
        "architecture": "MultiStepProtoKANCognitive + ShortHorizonEnergy + TDBootstrapValue",
        "teacher_usage": "none",
        "cognitive_fit": cognitive_fit,
        "cognitive_model_rollout_error": model_errors,
        "energy_one_step_fit": energy_fit,
        "evaluation_after_energy_fit": before_value,
        "terminal_value_td_fit": value_fit,
        "evaluation_after_value_td": before_decision,
        "short_horizon_energy_fit": decision_fit,
        "fixed_test_evaluation": final,
        "test_seed": args.test_seed,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
