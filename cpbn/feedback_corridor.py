"""Short-horizon corridor planning for feedback composition."""

from __future__ import annotations

import torch
from torch import nn

from cpbn.receding_tube import LocalPlan, local_state_distance


@torch.no_grad()
def plan_feedback_corridor(
    dynamics: nn.Module,
    start: torch.Tensor,
    target_path: torch.Tensor,
    action_segments: int = 4,
    population: int = 256,
    elite_count: int = 32,
    iterations: int = 5,
    seed: int = 0,
) -> LocalPlan:
    """Fit a whole short state corridor, not only its final waypoint."""
    horizon = target_path.shape[0]
    if horizon % action_segments:
        raise ValueError("target path length must be divisible by action_segments")
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
        path_cost = torch.zeros(population)
        terminal_distance = None
        for step in range(horizon):
            segment = min(action_segments - 1, step // segment_steps)
            action = segment_actions[:, segment:segment + 1]
            state = dynamics(state, action)
            distance = local_state_distance(
                state, target_path[step].view(1, 6),
            )
            path_cost = path_cost + distance.square()
            terminal_distance = distance
        path_cost = path_cost / horizon
        smoothness = (
            segment_actions[:, 1:] - segment_actions[:, :-1]
        ).square().mean(dim=-1)
        effort = segment_actions.square().mean(dim=-1)
        score = (
            -4.0 * terminal_distance.square()
            -0.5 * path_cost
            -0.002 * effort
            -0.002 * smoothness
        )
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
        raise RuntimeError("corridor CEM did not produce an action sequence")

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
        terminal_distance=float(local_state_distance(
            states[-1:], target_path[-1:].view(1, 6),
        )),
    )
