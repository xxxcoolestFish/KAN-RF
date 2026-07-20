"""Goal-directed routing on a temporal causal effect graph."""

from __future__ import annotations

import torch

from .causal_graph import rollout_model, temporal_action_effect


def goal_gradient(goal_potential, states: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
    """Differentiate a scalar goal potential with respect to state."""
    variable = states.detach().clone().requires_grad_(True)
    value = goal_potential(variable, goal).sum()
    return torch.autograd.grad(value, variable)[0]


def route_score(
    transition,
    goal_potential,
    state: torch.Tensor,
    action_sequence: torch.Tensor,
    goal: torch.Tensor,
    gamma: float = 0.95,
    epsilon: float = 1e-3,
) -> torch.Tensor:
    """Score the positive action direction using graph effects and goal gradients.

    Positive score means that increasing the selected first action coordinate is
    predicted to reduce the discounted multi-step goal potential.
    """
    predicted = rollout_model(transition, state, action_sequence)
    effects = temporal_action_effect(
        transition, state, action_sequence, epsilon=epsilon,
        action_time=0, action_index=0,
    )
    scores = torch.zeros(state.shape[0], dtype=state.dtype, device=state.device)
    for step_index in range(action_sequence.shape[1]):
        gradient = goal_gradient(
            goal_potential, predicted[:, step_index], goal,
        )
        scores = scores - (gamma ** step_index) * (
            effects[:, step_index] * gradient
        ).sum(dim=-1)
    return scores


def route_action(
    score: torch.Tensor,
    action_limit: float = 0.9,
    score_scale: float = 0.05,
) -> torch.Tensor:
    """Convert a route score into a bounded continuous action."""
    return action_limit * torch.tanh(score.unsqueeze(-1) / score_scale)


def discounted_goal_cost(
    transition,
    state: torch.Tensor,
    actions: torch.Tensor,
    goal: torch.Tensor,
    gamma: float = 0.95,
    goal_potential=None,
) -> torch.Tensor:
    """Evaluate the discounted goal potential of a fixed action sequence."""
    predicted = rollout_model(transition, state, actions)
    costs = []
    for step_index in range(actions.shape[1]):
        costs.append((gamma ** step_index) * goal_potential(
            predicted[:, step_index], goal,
        ))
    return torch.stack(costs, dim=1).sum(dim=1)
