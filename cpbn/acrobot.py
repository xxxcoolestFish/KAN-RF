"""Differentiable Acrobot task used by the current CPBN checkpoint."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


STATE_DIM = 6
ACTION_DIM = 1
SOURCE_FACTOR = (7.35, 0.0, 0.8, 0.8)
GOAL = torch.tensor([-1.0, 0.0, 1.0, 0.0, 0.0, 0.0])

L1 = L2 = 1.0
LC1 = LC2 = 0.5
I1 = I2 = 1.0
DT = 0.05
MAX_V1 = 6.0
MAX_V2 = 8.0


def _factor_batch(
    factor: torch.Tensor | Sequence[float], state: torch.Tensor,
) -> torch.Tensor:
    value = torch.as_tensor(factor, dtype=state.dtype, device=state.device)
    if value.ndim == 1:
        value = value.unsqueeze(0).expand(state.shape[0], -1)
    if value.shape != (state.shape[0], 4):
        raise ValueError("factor must have shape [4] or [batch, 4]")
    return value


def step(
    state: torch.Tensor,
    action: torch.Tensor,
    factor: torch.Tensor | Sequence[float] = SOURCE_FACTOR,
) -> torch.Tensor:
    """Advance a state batch under gravity/damping/actuation/inertia factors."""
    factor = _factor_batch(factor, state)
    gravity, damping, actuation, inertia_scale = factor.unbind(dim=-1)

    c1, s1, c2, s2 = state[:, 0], state[:, 1], state[:, 2], state[:, 3]
    theta1, theta2 = torch.atan2(s1, c1), torch.atan2(s2, c2)
    dtheta1, dtheta2 = state[:, 4] * MAX_V1, state[:, 5] * MAX_V2
    sin2, cos2 = torch.sin(theta2), torch.cos(theta2)
    i1, i2 = I1 * inertia_scale, I2 * inertia_scale

    d1 = LC1**2 + L1**2 + LC2**2 + 2.0 * L1 * LC2 * cos2 + i1 + i2
    d2 = LC2**2 + i2
    coupling = LC2**2 + L1 * LC2 * cos2 + i2
    phi2 = gravity * LC2 * torch.sin(theta1 + theta2)
    phi1 = (
        -L1 * LC2 * dtheta2.square() * sin2
        - 2.0 * L1 * LC2 * dtheta2 * dtheta1 * sin2
        + (LC1 + L1) * gravity * torch.sin(theta1)
        + phi2
        - damping * dtheta1
    )
    denominator = d2 - coupling.square() / d1
    ddtheta2 = (
        action[:, 0] * actuation
        + coupling / d1 * phi1
        - L1 * LC2 * dtheta1.square() * sin2
        - phi2
        - damping * dtheta2
    ) / (denominator + 1e-6)
    ddtheta1 = -(coupling * ddtheta2 + phi1) / d1

    dtheta1 = (dtheta1 + ddtheta1 * DT).clamp(-MAX_V1, MAX_V1)
    dtheta2 = (dtheta2 + ddtheta2 * DT).clamp(-MAX_V2, MAX_V2)
    theta1 = theta1 + dtheta1 * DT
    theta2 = theta2 + dtheta2 * DT
    return torch.stack(
        [
            torch.cos(theta1),
            torch.sin(theta1),
            torch.cos(theta2),
            torch.sin(theta2),
            dtheta1 / MAX_V1,
            dtheta2 / MAX_V2,
        ],
        dim=-1,
    )


class OracleAcrobotDynamics(nn.Module):
    """Exact dynamics behind the same interface as a learned cognition model."""

    def __init__(self, factor: Sequence[float] = SOURCE_FACTOR):
        super().__init__()
        self.register_buffer("factor", torch.as_tensor(factor, dtype=torch.float32))

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return step(state, action, self.factor)


def random_states(
    count: int, generator: torch.Generator | None = None,
) -> torch.Tensor:
    angles = (torch.rand(count, 2, generator=generator) * 2.0 - 1.0) * torch.pi
    velocity = torch.rand(count, 2, generator=generator) * 2.0 - 1.0
    return torch.stack(
        [
            torch.cos(angles[:, 0]),
            torch.sin(angles[:, 0]),
            torch.cos(angles[:, 1]),
            torch.sin(angles[:, 1]),
            velocity[:, 0],
            velocity[:, 1],
        ],
        dim=-1,
    )


def reset_down_states(
    count: int,
    noise: float = 0.04,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    angles = torch.randn(count, 2, generator=generator) * noise
    return torch.stack(
        [
            torch.cos(angles[:, 0]),
            torch.sin(angles[:, 0]),
            torch.cos(angles[:, 1]),
            torch.sin(angles[:, 1]),
            torch.randn(count, generator=generator) * noise,
            torch.randn(count, generator=generator) * noise,
        ],
        dim=-1,
    )


def tip_height(state: torch.Tensor) -> torch.Tensor:
    c1, s1, c2, s2 = state[:, 0], state[:, 1], state[:, 2], state[:, 3]
    return -c1 - (c1 * c2 - s1 * s2)


def task_reward(
    state: torch.Tensor, next_state: torch.Tensor, action: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Task-level reward; it contains no hand-designed energy teacher."""
    del state
    height = tip_height(next_state)
    success = height >= 1.0
    reward = 0.25 * (height + 2.0) + 5.0 * success.to(next_state.dtype)
    reward = reward - 0.005 * action.square().sum(dim=-1)
    return reward, success
