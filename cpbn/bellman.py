"""Bellman pullback and an actor-free implicit action layer."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn

from cpbn.acrobot import task_reward


RewardFunction = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor],
]


def bellman_return(
    value: nn.Module,
    dynamics: nn.Module,
    state: torch.Tensor,
    goal: torch.Tensor,
    action: torch.Tensor,
    gamma: float = 0.99,
    reward_function: RewardFunction = task_reward,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pull the future value back through the cognition/dynamics operator."""
    next_state = dynamics(state, action)
    reward, done = reward_function(state, next_state, action)
    continuation = value(next_state, goal)
    objective = reward + gamma * continuation * (~done).to(state.dtype)
    return objective, next_state, reward, done


class ImplicitBellmanAction(nn.Module):
    """Solve the projected stationarity equation dQ/da=0 for each state.

    The module owns no trainable actor parameters.  Every action therefore
    depends on the supplied dynamics model and value network in the forward
    path, which prevents a separate actor from bypassing cognition.
    """

    def __init__(
        self,
        iterations: int = 10,
        gradient_step: float = 0.20,
        max_step: float = 0.25,
        curvature_floor: float = 1e-3,
        gamma: float = 0.99,
    ):
        super().__init__()
        self.iterations = iterations
        self.gradient_step = gradient_step
        self.max_step = max_step
        self.curvature_floor = curvature_floor
        self.gamma = gamma

    def forward(
        self,
        value: nn.Module,
        dynamics: nn.Module,
        state: torch.Tensor,
        goal: torch.Tensor,
        initial_action: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if initial_action is None:
            action = state.new_zeros(state.shape[0], 1)
        else:
            action = initial_action.detach().clone().clamp(-1.0, 1.0)

        for _ in range(self.iterations):
            action.requires_grad_(True)
            objective, _, _, _ = bellman_return(
                value, dynamics, state, goal, action, self.gamma,
            )
            gradient = torch.autograd.grad(
                objective.sum(), action, create_graph=True,
            )[0]
            curvature = torch.autograd.grad(gradient.sum(), action)[0]
            newton_step = -gradient / curvature.clamp_max(-self.curvature_floor)
            ascent_step = self.gradient_step * gradient
            delta = torch.where(
                curvature < -self.curvature_floor,
                newton_step,
                ascent_step,
            ).clamp(-self.max_step, self.max_step)
            action = (action.detach() + delta.detach()).clamp(-1.0, 1.0)
        return action.detach()


@torch.no_grad()
def grid_best_action(
    value: nn.Module,
    dynamics: nn.Module,
    state: torch.Tensor,
    goal: torch.Tensor,
    grid_points: int = 129,
    gamma: float = 0.99,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Audit-only global grid approximation; never used to execute actions."""
    grid = torch.linspace(-1.0, 1.0, grid_points, device=state.device)
    batch = state.shape[0]
    tiled_state = state[:, None, :].expand(-1, grid_points, -1).reshape(
        -1, state.shape[-1],
    )
    tiled_goal = goal[:, None, :].expand(-1, grid_points, -1).reshape(
        -1, goal.shape[-1],
    )
    tiled_action = grid.view(1, -1, 1).expand(batch, -1, -1).reshape(-1, 1)
    objective, _, _, _ = bellman_return(
        value, dynamics, tiled_state, tiled_goal, tiled_action, gamma,
    )
    objective = objective.view(batch, grid_points)
    best_index = objective.argmax(dim=1)
    return grid[best_index].unsqueeze(-1), objective.max(dim=1).values
