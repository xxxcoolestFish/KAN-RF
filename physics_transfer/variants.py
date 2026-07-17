"""Hidden multi-factor Acrobot variants for representation validation."""

from __future__ import annotations

import torch


L1 = L2 = 1.0
LC1 = LC2 = 0.5
I1 = I2 = 1.0
DT = 0.05
MAX_V1 = 6.0
MAX_V2 = 8.0


def _factor(value, state: torch.Tensor) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=state.dtype, device=state.device)
    if tensor.ndim == 0:
        tensor = tensor.expand(state.shape[0])
    return tensor.reshape(-1)


def step(
    state: torch.Tensor,
    action: torch.Tensor,
    gravity: torch.Tensor | float,
    damping: torch.Tensor | float,
    actuation: torch.Tensor | float,
    inertia_scale: torch.Tensor | float,
) -> torch.Tensor:
    """Advance a batch under hidden gravity/damping/actuation/inertia factors."""
    gravity = _factor(gravity, state)
    damping = _factor(damping, state)
    actuation = _factor(actuation, state)
    inertia_scale = _factor(inertia_scale, state)

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
    return torch.stack([
        torch.cos(theta1), torch.sin(theta1),
        torch.cos(theta2), torch.sin(theta2),
        dtheta1 / MAX_V1, dtheta2 / MAX_V2,
    ], dim=-1)
