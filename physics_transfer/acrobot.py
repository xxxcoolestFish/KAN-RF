"""Differentiable continuous-action Acrobot used by the new framework."""

from __future__ import annotations

import math

import numpy as np
import torch


L1 = L2 = 1.0
LC1 = LC2 = 0.5
I1 = I2 = 1.0
DT = 0.05
MAX_V1 = 6.0
MAX_V2 = 8.0
TARGET_H = 1.0


def step(state: torch.Tensor, action: torch.Tensor, gravity: float = 9.8) -> torch.Tensor:
    """Advance normalized Acrobot states by one differentiable step."""
    c1, s1, c2, s2 = state[:, 0], state[:, 1], state[:, 2], state[:, 3]
    theta1, theta2 = torch.atan2(s1, c1), torch.atan2(s2, c2)
    dtheta1, dtheta2 = state[:, 4] * MAX_V1, state[:, 5] * MAX_V2
    sin2, cos2 = torch.sin(theta2), torch.cos(theta2)

    d1 = LC1**2 + L1**2 + LC2**2 + 2.0 * L1 * LC2 * cos2 + I1 + I2
    d2 = LC2**2 + I2
    coupling = LC2**2 + L1 * LC2 * cos2 + I2
    phi2 = gravity * LC2 * torch.sin(theta1 + theta2)
    phi1 = (
        -L1 * LC2 * dtheta2.square() * sin2
        - 2.0 * L1 * LC2 * dtheta2 * dtheta1 * sin2
        + (LC1 + L1) * gravity * torch.sin(theta1)
        + phi2
    )
    denominator = d2 - coupling.square() / d1
    ddtheta2 = (
        action[:, 0]
        + coupling / d1 * phi1
        - L1 * LC2 * dtheta1.square() * sin2
        - phi2
    ) / (denominator + 1e-6)
    ddtheta1 = -(coupling * ddtheta2 + phi1) / d1

    dtheta1 = (dtheta1 + ddtheta1 * DT).clamp(-MAX_V1, MAX_V1)
    dtheta2 = (dtheta2 + ddtheta2 * DT).clamp(-MAX_V2, MAX_V2)
    theta1 = theta1 + dtheta1 * DT
    theta2 = theta2 + dtheta2 * DT
    return torch.stack(
        [torch.cos(theta1), torch.sin(theta1), torch.cos(theta2), torch.sin(theta2),
         dtheta1 / MAX_V1, dtheta2 / MAX_V2], dim=-1,
    )


def tip_height(state: torch.Tensor) -> torch.Tensor:
    theta1 = torch.atan2(state[:, 1], state[:, 0])
    theta12 = theta1 + torch.atan2(state[:, 3], state[:, 2])
    return -L1 * torch.cos(theta1) - L2 * torch.cos(theta12)


def reset_state(rng: np.random.Generator) -> torch.Tensor:
    theta1, theta2 = rng.uniform(-0.1, 0.1, size=2)
    return torch.tensor(
        [[math.cos(theta1), math.sin(theta1), math.cos(theta2), math.sin(theta2), 0.0, 0.0]],
        dtype=torch.float32,
    )


def evaluation_starts(n: int, seed: int = 1042) -> list[torch.Tensor]:
    rng = np.random.default_rng(seed)
    return [reset_state(rng) for _ in range(n)]
