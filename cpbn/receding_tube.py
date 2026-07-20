"""Short-horizon cognitive planning and time-varying local feedback."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from cpbn.time_varying_tube import _local_jacobians, tangent_error


LOCAL_METRIC = torch.tensor([1.0, 1.0, 0.35, 0.35])


def local_state_distance(
    state: torch.Tensor, target: torch.Tensor,
) -> torch.Tensor:
    error = tangent_error(state, target)
    return (error * LOCAL_METRIC.to(error)).square().sum(dim=-1).sqrt()


@dataclass
class LocalPlan:
    states: torch.Tensor
    actions: torch.Tensor
    terminal_distance: float


@torch.no_grad()
def plan_local_cem(
    dynamics: nn.Module,
    start: torch.Tensor,
    target: torch.Tensor,
    horizon: int = 24,
    action_segments: int = 4,
    population: int = 256,
    elite_count: int = 32,
    iterations: int = 5,
    seed: int = 0,
) -> LocalPlan:
    if horizon % action_segments:
        raise ValueError("horizon must be divisible by action_segments")
    generator = torch.Generator().manual_seed(seed)
    mean = torch.zeros(action_segments)
    std = torch.ones(action_segments)
    segment_steps = horizon // action_segments
    best_score = -float("inf")
    best_segment_actions = None
    for _ in range(iterations):
        segment_actions = (
            mean + std * torch.randn(
                population, action_segments, generator=generator,
            )
        ).clamp(-1.0, 1.0)
        state = start.view(1, 6).expand(population, -1).clone()
        for segment in range(action_segments):
            action = segment_actions[:, segment:segment + 1]
            for _ in range(segment_steps):
                state = dynamics(state, action)
        distance = local_state_distance(state, target.view(1, 6))
        smoothness = (
            segment_actions[:, 1:] - segment_actions[:, :-1]
        ).square().mean(dim=-1)
        effort = segment_actions.square().mean(dim=-1)
        score = -distance.square() - 0.002 * effort - 0.002 * smoothness
        elite = score.topk(elite_count).indices
        mean = 0.25 * mean + 0.75 * segment_actions[elite].mean(dim=0)
        std = (
            0.25 * std + 0.75 * segment_actions[elite].std(dim=0)
        ).clamp(0.04, 1.0)
        index = int(score.argmax())
        if float(score[index]) > best_score:
            best_score = float(score[index])
            best_segment_actions = segment_actions[index].clone()
    if best_segment_actions is None:
        raise RuntimeError("local CEM did not produce an action sequence")

    actions = best_segment_actions.repeat_interleave(segment_steps).unsqueeze(-1)
    state = start.view(1, 6)
    states = [state.squeeze(0).clone()]
    for action in actions:
        state = dynamics(state, action.view(1, 1))
        states.append(state.squeeze(0).clone())
    states = torch.stack(states)
    return LocalPlan(
        states=states,
        actions=actions,
        terminal_distance=float(local_state_distance(states[-1:], target.view(1, 6))),
    )


def variable_riccati_gains(
    dynamics: nn.Module,
    states: torch.Tensor,
    actions: torch.Tensor,
) -> torch.Tensor:
    horizon = actions.shape[0]
    state_jacobians, action_jacobians = [], []
    for step in range(horizon):
        state_jacobian, action_jacobian = _local_jacobians(
            dynamics, states[step], states[step + 1], actions[step],
        )
        state_jacobians.append(state_jacobian)
        action_jacobians.append(action_jacobian)
    state_jacobians = torch.stack(state_jacobians)
    action_jacobians = torch.stack(action_jacobians)
    state_cost = torch.diag(torch.tensor([2.0, 2.0, 0.5, 0.5]))
    value = torch.diag(torch.tensor([8.0, 8.0, 2.0, 2.0]))
    action_cost = torch.tensor(0.20)
    gains = torch.zeros(horizon, 1, 4)
    for step in reversed(range(horizon)):
        state_matrix = state_jacobians[step]
        action_matrix = action_jacobians[step]
        denominator = action_cost + (
            action_matrix.T @ value @ action_matrix
        ).squeeze()
        gain = -(
            action_matrix.T @ value @ state_matrix
        ) / denominator.clamp_min(1e-6)
        gains[step] = gain
        value = (
            state_cost
            + state_matrix.T @ value @ state_matrix
            + state_matrix.T @ value @ action_matrix @ gain
        )
        value = 0.5 * (value + value.T) + 1e-6 * torch.eye(4)
    return gains


def nearest_reference_progress(
    state: torch.Tensor,
    reference: torch.Tensor,
    current: int,
    search_window: int = 48,
) -> int:
    stop = min(reference.shape[0], current + search_window + 1)
    candidates = reference[current:stop]
    distance = local_state_distance(state.view(1, 6), candidates)
    return current + int(distance.argmin())
