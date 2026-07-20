"""Composable one-step influence graphs and temporal adjoint routing."""

from __future__ import annotations

from collections.abc import Callable

import torch


TensorTransition = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
StateScore = Callable[[torch.Tensor], torch.Tensor]


def repeated_transition(transition: TensorTransition, repeat: int) -> TensorTransition:
    """Turn a one-step model into a macro transition with a held action."""

    def macro(state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        for _ in range(repeat):
            state = transition(state, action)
        return state

    return macro


def local_influence_graph(transition: TensorTransition, state: torch.Tensor,
                          action: torch.Tensor):
    """Return next state and per-sample A=dF/ds, B=dF/da."""
    with torch.enable_grad():
        local_state = state.detach().requires_grad_(True)
        local_action = action.detach().requires_grad_(True)
        next_state = transition(local_state, local_action)
        state_rows, action_rows = [], []
        for output_index in range(next_state.shape[-1]):
            state_grad, action_grad = torch.autograd.grad(
                next_state[:, output_index].sum(),
                (local_state, local_action), retain_graph=True,
            )
            state_rows.append(state_grad)
            action_rows.append(action_grad)
    return (
        next_state.detach(),
        torch.stack(state_rows, dim=1).detach(),
        torch.stack(action_rows, dim=1).detach(),
    )


def unroll_influence_graph(transition: TensorTransition,
                           initial_state: torch.Tensor,
                           actions: torch.Tensor):
    """Compose local one-step influence graphs along an action sequence."""
    states = [initial_state.detach()]
    state_matrices, action_matrices = [], []
    state = initial_state
    for index in range(actions.shape[1]):
        state, state_matrix, action_matrix = local_influence_graph(
            transition, state, actions[:, index],
        )
        states.append(state)
        state_matrices.append(state_matrix)
        action_matrices.append(action_matrix)
    return (
        torch.stack(states, dim=1),
        torch.stack(state_matrices, dim=1),
        torch.stack(action_matrices, dim=1),
    )


def _score_gradient(score_fn: StateScore, state: torch.Tensor):
    with torch.enable_grad():
        local_state = state.detach().requires_grad_(True)
        score = score_fn(local_state)
        gradient = torch.autograd.grad(score.sum(), local_state)[0]
    return gradient.detach()


def temporal_reachability_route(states: torch.Tensor,
                                state_matrices: torch.Tensor,
                                action_matrices: torch.Tensor,
                                score_fn: StateScore, temperature: float):
    """Route a smooth maximum state score backward through the temporal graph.

    The returned tensor is d softmax_score / d action for every time step.
    """
    horizon = state_matrices.shape[1]
    state_scores = torch.stack(
        [score_fn(states[:, index]) for index in range(1, horizon + 1)],
        dim=1,
    )
    weights = torch.softmax(state_scores / temperature, dim=1)
    local_score_gradients = [
        _score_gradient(score_fn, states[:, index])
        for index in range(1, horizon + 1)
    ]
    costate = weights[:, -1:].mul(local_score_gradients[-1])
    action_routes = [None] * horizon
    for index in reversed(range(horizon)):
        action_routes[index] = torch.bmm(
            action_matrices[:, index].transpose(1, 2),
            costate.unsqueeze(-1),
        ).squeeze(-1)
        costate = torch.bmm(
            state_matrices[:, index].transpose(1, 2),
            costate.unsqueeze(-1),
        ).squeeze(-1)
        if index > 0:
            costate = costate + weights[:, index - 1:].mul(
                local_score_gradients[index - 1]
            )
    smooth_score = temperature * torch.logsumexp(
        state_scores / temperature, dim=1,
    )
    return torch.stack(action_routes, dim=1), smooth_score, weights


def direct_reachability_gradient(transition: TensorTransition,
                                 initial_state: torch.Tensor,
                                 actions: torch.Tensor,
                                 score_fn: StateScore, temperature: float):
    """Reference gradient through the fully unrolled transition model."""
    with torch.enable_grad():
        local_actions = actions.detach().requires_grad_(True)
        state = initial_state.detach()
        scores = []
        for index in range(local_actions.shape[1]):
            state = transition(state, local_actions[:, index])
            scores.append(score_fn(state))
        state_scores = torch.stack(scores, dim=1)
        smooth_score = temperature * torch.logsumexp(
            state_scores / temperature, dim=1,
        )
        gradient = torch.autograd.grad(smooth_score.sum(), local_actions)[0]
    return gradient.detach(), smooth_score.detach()
