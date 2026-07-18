"""First structural test of an embedded cognitive-predictive controller.

The cognitive network is intentionally simple: one ProtoKAN predicts the
next state from ``(state, action)``.  A second ProtoKAN is the new decision
parameter block and outputs a scalar energy for a predicted next state.  The
action is obtained by differentiable inner-loop descent through the cognitive
network, so the final action path cannot bypass the cognitive parameters.

No action teacher, MPC, operator code, or latent partition is used.
"""

from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F
from torch import nn

from kanrf._protokan import ProtoKAN
from physics_transfer.multifactor_data import _random_states
from physics_transfer.transition_data import sample_transition_batch
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
    """Plain one-step ProtoKAN world model without latent partitioning."""

    def __init__(self, state_dim=6, action_dim=1, hidden_dim=32, n_prototypes=8):
        super().__init__()
        self.network = ProtoKAN(
            [state_dim + action_dim, hidden_dim, state_dim],
            n_prototypes=n_prototypes,
        )

    def forward(self, state, action):
        return self.network(torch.cat([state, action], dim=-1))


class DecisionEnergyKAN(nn.Module):
    """New decision parameters: predicted state + goal -> scalar energy."""

    def __init__(self, state_dim=6, goal_dim=6, hidden_dim=32, n_prototypes=8):
        super().__init__()
        self.network = ProtoKAN(
            [state_dim + goal_dim, hidden_dim, 1],
            n_prototypes=n_prototypes,
        )

    def forward(self, predicted_state, goal):
        return self.network(torch.cat([predicted_state, goal], dim=-1))


def pretrain_cognitive(model, steps, batch_size, seed):
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    losses = []
    for _ in range(steps):
        batch = sample_transition_batch(batch_size, PRETRAIN_FACTOR)
        prediction = model(batch["state"], batch["action"])
        loss = F.smooth_l1_loss(prediction, batch["next_state"])
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        losses.append(loss.item())
    return {
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "mean_last_20_loss": sum(losses[-20:]) / min(20, len(losses)),
    }


def fit_energy_supervision(cognitive, energy, steps, batch_size, seed, goal):
    """Fit energy to the task cost of cognitive one-step predictions.

    This is not an action teacher: the target is a scalar task cost, not an
    action label.  It only gives the energy surface a sensible initialization.
    """
    torch.manual_seed(seed + 1000)
    optimizer = torch.optim.Adam(energy.parameters(), lr=2e-3)
    losses = []
    with torch.no_grad():
        for parameter in cognitive.parameters():
            parameter.requires_grad = False
    for _ in range(steps):
        batch = sample_transition_batch(batch_size, PRETRAIN_FACTOR)
        with torch.no_grad():
            predicted = cognitive(batch["state"], batch["action"])
            target = task_cost(predicted, batch["action"]).unsqueeze(-1)
        prediction = energy(predicted, goal.expand(batch_size, -1))
        loss = F.smooth_l1_loss(prediction, target)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        losses.append(loss.item())
    return {
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "mean_last_20_loss": sum(losses[-20:]) / min(20, len(losses)),
    }


def controller_action(cognitive, energy, state, goal, inner_steps, step_size, create_graph):
    """Solve the implicit energy controller by differentiable action descent."""
    action = torch.zeros(state.shape[0], 1, dtype=state.dtype, device=state.device)
    action.requires_grad_(True)
    for _ in range(inner_steps):
        predicted = cognitive(state, action)
        energy_value = energy(predicted, goal.expand(state.shape[0], -1))
        gradient = torch.autograd.grad(
            energy_value.sum(), action, create_graph=create_graph,
        )[0]
        action = (action - step_size * gradient).clamp(-1.0, 1.0)
    return action


def train_decision(cognitive, energy, steps, batch_size, horizon,
                   inner_steps, step_size, seed, goal):
    torch.manual_seed(seed + 2000)
    optimizer = torch.optim.Adam(energy.parameters(), lr=1e-3)
    losses = []
    for _ in range(steps):
        state = _random_states(batch_size)
        current = state
        costs, actions = [], []
        for _ in range(horizon):
            action = controller_action(
                cognitive, energy, current, goal,
                inner_steps, step_size, create_graph=True,
            )
            next_state = cognitive(current, action)
            costs.append(task_cost(next_state, action))
            actions.append(action)
            current = next_state
        cost_stack = torch.stack(costs, dim=1)
        action_stack = torch.stack(actions, dim=1)
        smooth = (action_stack[:, 1:] - action_stack[:, :-1]).square().mean()
        loss = cost_stack[:, -1].mean() + 0.25 * cost_stack.mean() + 0.02 * smooth
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(energy.parameters(), 5.0)
        optimizer.step()
        losses.append(loss.item())
    return {
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "mean_last_20_loss": sum(losses[-20:]) / min(20, len(losses)),
    }


def evaluate(cognitive, energy, states, factor, goal, steps, inner_steps, step_size):
    current = states.detach().clone()
    factor_tensor = torch.tensor(factor, dtype=current.dtype).view(1, 4).expand(current.shape[0], -1)
    maxima = torch.full((current.shape[0],), -float("inf"))
    for _ in range(steps):
        with torch.enable_grad():
            action = controller_action(
                cognitive, energy, current, goal,
                inner_steps, step_size, create_graph=False,
            )
        action = action.detach()
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
    parser.add_argument("--energy-fit-steps", type=int, default=100)
    parser.add_argument("--decision-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--inner-steps", type=int, default=3)
    parser.add_argument("--action-step-size", type=float, default=0.25)
    parser.add_argument("--rollout-steps", type=int, default=64)
    parser.add_argument("--test-count", type=int, default=20)
    parser.add_argument("--test-seed", type=int, default=20260718)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    goal = GOAL.view(1, -1)
    cognitive = SimpleCognitiveKAN()
    cognitive_fit = pretrain_cognitive(cognitive, args.cognitive_steps, 64, args.seed)
    energy = DecisionEnergyKAN()
    energy_fit = fit_energy_supervision(
        cognitive, energy, args.energy_fit_steps, args.batch_size, args.seed, goal
    )
    decision_fit = train_decision(
        cognitive, energy, args.decision_steps, args.batch_size,
        args.horizon, args.inner_steps, args.action_step_size, args.seed, goal,
    )
    generator = torch.Generator().manual_seed(args.test_seed)
    test_states = _random_states(args.test_count, generator=generator)
    result = evaluate(
        cognitive, energy, test_states, PRETRAIN_FACTOR[0], goal,
        args.rollout_steps, args.inner_steps, args.action_step_size,
    )
    print(json.dumps({
        "architecture": "SimpleProtoKANCognitive + ProtoKANEnergyImplicitController",
        "cognitive_parameters_retained": True,
        "teacher_usage": "none",
        "cognitive_fit": cognitive_fit,
        "energy_fit": energy_fit,
        "decision_fit": decision_fit,
        "fixed_test_evaluation": result,
        "test_seed": args.test_seed,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

