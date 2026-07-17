"""Small formula-agnostic interaction batches for stage 1."""

from __future__ import annotations

import torch

from .acrobot import step


def _random_states(n: int, generator: torch.Generator | None = None) -> torch.Tensor:
    angles = (torch.rand(n, 2, generator=generator) * 2.0 - 1.0) * torch.pi
    velocity = torch.rand(n, 2, generator=generator) * 2.0 - 1.0
    return torch.stack([
        torch.cos(angles[:, 0]), torch.sin(angles[:, 0]),
        torch.cos(angles[:, 1]), torch.sin(angles[:, 1]),
        velocity[:, 0], velocity[:, 1],
    ], dim=-1)


def sample_acrobot_batch(batch_size: int, history_steps: int,
                         gravities: tuple[float, ...],
                         generator: torch.Generator | None = None):
    """Generate histories while hiding the gravity label from the model."""
    gravity_index = torch.randint(len(gravities), (batch_size,), generator=generator)
    gravity = torch.tensor(gravities, dtype=torch.float32)[gravity_index]
    state = _random_states(batch_size, generator)
    history = []

    for _ in range(history_steps):
        action = torch.rand(batch_size, 1, generator=generator) * 2.0 - 1.0
        history.append(torch.cat([state, action], dim=-1))
        next_state = state.clone()
        for index, gravity_value in enumerate(gravities):
            mask = gravity_index == index
            if mask.any():
                next_state[mask] = step(state[mask], action[mask], float(gravity_value))
        state = next_state

    current_action = torch.rand(batch_size, 1, generator=generator) * 2.0 - 1.0
    target = state.clone()
    for index, gravity_value in enumerate(gravities):
        mask = gravity_index == index
        if mask.any():
            target[mask] = step(state[mask], current_action[mask], float(gravity_value))

    return {
        "history": torch.cat(history, dim=-1),
        "state": state,
        "action": current_action,
        "next_state": target,
        "gravity": gravity,
    }
