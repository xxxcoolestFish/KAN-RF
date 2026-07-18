"""Bellman cost-to-go and trust-region energy control.

Stage 23 showed that a terminal value helps, but a one-shot value fit followed
by energy optimization is unstable.  This experiment keeps the embedded
cognitive ProtoKAN, trains a positive cost-to-go with a slowly updated target,
and alternates real-transition TD updates with short-horizon energy updates.
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
from physics_transfer.transition_data import sample_transition_batch
from physics_transfer.variants import step
from scripts.stage15_real_outcome_replay import PRETRAIN_FACTOR
from scripts.stage23_multistep_terminal_value import (
    GOAL,
    EnergyKAN,
    SimpleCognitiveKAN,
    fit_one_step_energy,
    pretrain_cognitive_multistep,
    smooth_tip_height,
    task_cost,
)


class PositiveCostToGoKAN(nn.Module):
    """Non-negative goal-conditioned cost-to-go used as the terminal energy."""

    def __init__(self, state_dim=6, goal_dim=6, hidden_dim=32, n_prototypes=8):
        super().__init__()
        self.network = ProtoKAN(
            [state_dim + goal_dim, hidden_dim, 1],
            n_prototypes=n_prototypes,
        )

    def forward(self, state, goal):
        return F.softplus(self.network(torch.cat([state, goal], dim=-1)))


def _stats(losses):
    return {
        "first_loss": float(losses[0]),
        "last_loss": float(losses[-1]),
        "mean_last_20_loss": float(sum(losses[-20:]) / min(20, len(losses))),
    }


def bounded_controller_action(cognitive, energy, value, state, goal,
                              inner_steps, step_size, gamma, create_graph):
    """Differentiable action descent with a smooth trust-region step."""
    action = torch.zeros(state.shape[0], 1, dtype=state.dtype)
    action.requires_grad_(True)
    goal_batch = goal.expand(state.shape[0], -1)
    for _ in range(inner_steps):
        predicted = cognitive(state, action)
        objective = (
            task_cost(predicted, action)
            + gamma * value(predicted, goal_batch).squeeze(-1)
            + 0.05 * energy(predicted, goal_batch).squeeze(-1)
        )
        gradient = torch.autograd.grad(
            objective.sum(), action, create_graph=create_graph,
        )[0]
        # Smooth clipping avoids a large local energy gradient jumping directly
        # to an action boundary while preserving differentiability.
        gradient = torch.tanh(gradient)
        action = (action - step_size * gradient).clamp(-1.0, 1.0)
    return action


def td_update(cognitive, energy, value, target_value, optimizer, batch_size,
              seed, goal, gamma, inner_steps, step_size):
    torch.manual_seed(seed)
    batch = sample_transition_batch(batch_size, PRETRAIN_FACTOR)
    with torch.enable_grad():
        action = bounded_controller_action(
            cognitive, energy, value, batch["state"], goal,
            inner_steps, step_size, gamma, create_graph=False,
        ).detach()
    factor = batch["factors"]
    with torch.no_grad():
        real_next = step(
            batch["state"], action,
            factor[:, 0], factor[:, 1], factor[:, 2], factor[:, 3],
        )
        target = task_cost(real_next, action) + gamma * target_value(
            real_next, goal.expand(batch_size, -1)
        ).squeeze(-1)
        target = target.clamp(0.0, 20.0)
    prediction = value(batch["state"], goal.expand(batch_size, -1)).squeeze(-1)
    loss = F.smooth_l1_loss(prediction, target)
    optimizer.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(value.parameters(), 5.0)
    optimizer.step()
    with torch.no_grad():
        for target_parameter, parameter in zip(target_value.parameters(), value.parameters()):
            target_parameter.mul_(0.98).add_(0.02 * parameter)
    return loss.item()


def warmup_value(cognitive, energy, value, steps, batch_size, seed, goal,
                 gamma, inner_steps, step_size):
    for parameter in cognitive.parameters():
        parameter.requires_grad = False
    for parameter in energy.parameters():
        parameter.requires_grad = False
    target_value = copy.deepcopy(value)
    optimizer = torch.optim.Adam(value.parameters(), lr=1e-3)
    losses = []
    for index in range(steps):
        losses.append(td_update(
            cognitive, energy, value, target_value, optimizer, batch_size,
            seed + index, goal, gamma, inner_steps, step_size,
        ))
    return target_value, _stats(losses)


def alternate_td_energy(cognitive, energy, value, target_value, steps,
                        batch_size, horizon, seed, goal, gamma, inner_steps,
                        step_size):
    optimizer_value = torch.optim.Adam(value.parameters(), lr=1e-3)
    optimizer_energy = torch.optim.Adam(energy.parameters(), lr=5e-4)
    losses_value, losses_energy = [], []
    for index in range(steps):
        for parameter in value.parameters():
            parameter.requires_grad = True
        losses_value.append(td_update(
            cognitive, energy, value, target_value, optimizer_value, batch_size,
            seed + index * 17, goal, gamma, inner_steps, step_size,
        ))
        for parameter in value.parameters():
            parameter.requires_grad = False
        current = _random_states(batch_size)
        costs, actions = [], []
        for t in range(horizon):
            action = bounded_controller_action(
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
        loss_energy = (discounted + terminal).mean() + 0.02 * smooth
        optimizer_energy.zero_grad(); loss_energy.backward()
        torch.nn.utils.clip_grad_norm_(energy.parameters(), 2.0)
        optimizer_energy.step()
        losses_energy.append(loss_energy.item())
    return {"value": _stats(losses_value), "energy": _stats(losses_energy)}


def evaluate(cognitive, energy, value, states, factor, goal, steps, gamma,
             inner_steps, step_size):
    current = states.detach().clone()
    factor_tensor = torch.tensor(factor, dtype=current.dtype).view(1, 4).expand(current.shape[0], -1)
    maxima = torch.full((current.shape[0],), -float("inf"))
    for _ in range(steps):
        with torch.enable_grad():
            action = bounded_controller_action(
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
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cognitive-steps", type=int, default=300)
    parser.add_argument("--max-cognitive-horizon", type=int, default=8)
    parser.add_argument("--energy-fit-steps", type=int, default=100)
    parser.add_argument("--value-warmup-steps", type=int, default=100)
    parser.add_argument("--alternate-steps", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--inner-steps", type=int, default=3)
    parser.add_argument("--action-step-size", type=float, default=0.20)
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
    energy = EnergyKAN()
    energy_fit = fit_one_step_energy(
        cognitive, energy, args.energy_fit_steps, args.batch_size, args.seed, goal,
    )
    value = PositiveCostToGoKAN()
    generator = torch.Generator().manual_seed(args.test_seed)
    test_states = _random_states(args.test_count, generator=generator)
    initial = evaluate(
        cognitive, energy, value, test_states, PRETRAIN_FACTOR[0], goal,
        args.rollout_steps, args.gamma, args.inner_steps, args.action_step_size,
    )
    target_value, warmup_fit = warmup_value(
        cognitive, energy, value, args.value_warmup_steps, args.batch_size,
        args.seed, goal, args.gamma, args.inner_steps, args.action_step_size,
    )
    after_warmup = evaluate(
        cognitive, energy, value, test_states, PRETRAIN_FACTOR[0], goal,
        args.rollout_steps, args.gamma, args.inner_steps, args.action_step_size,
    )
    alternating_fit = alternate_td_energy(
        cognitive, energy, value, target_value, args.alternate_steps,
        args.batch_size, args.horizon, args.seed, goal, args.gamma,
        args.inner_steps, args.action_step_size,
    )
    final = evaluate(
        cognitive, energy, value, test_states, PRETRAIN_FACTOR[0], goal,
        args.rollout_steps, args.gamma, args.inner_steps, args.action_step_size,
    )
    print(json.dumps({
        "architecture": "MultiStepProtoKANCognitive + BellmanCostToGoEnergy",
        "teacher_usage": "none",
        "cognitive_fit": cognitive_fit,
        "energy_one_step_fit": energy_fit,
        "value_warmup_fit": warmup_fit,
        "alternating_fit": alternating_fit,
        "evaluation_initial": initial,
        "evaluation_after_value_warmup": after_warmup,
        "fixed_test_evaluation": final,
        "test_seed": args.test_seed,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
