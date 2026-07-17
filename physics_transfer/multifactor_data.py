"""Formula-hidden interaction batches for multi-factor Acrobot variants."""

from __future__ import annotations

import torch

from .variants import step


def _random_states(n: int, generator: torch.Generator | None = None) -> torch.Tensor:
    angles = (torch.rand(n, 2, generator=generator) * 2.0 - 1.0) * torch.pi
    velocity = torch.rand(n, 2, generator=generator) * 2.0 - 1.0
    return torch.stack([
        torch.cos(angles[:, 0]), torch.sin(angles[:, 0]),
        torch.cos(angles[:, 1]), torch.sin(angles[:, 1]),
        velocity[:, 0], velocity[:, 1],
    ], dim=-1)


def sample_multifactor_batch(
    batch_size: int,
    history_steps: int,
    factors: tuple[tuple[float, float, float, float], ...],
    generator: torch.Generator | None = None,
):
    """Generate transitions; factor labels are returned only for diagnostics."""
    factor_index = torch.randint(len(factors), (batch_size,), generator=generator)
    factor_tensor = torch.tensor(factors, dtype=torch.float32)[factor_index]
    state = _random_states(batch_size, generator)
    history = []

    for _ in range(history_steps):
        action = torch.rand(batch_size, 1, generator=generator) * 2.0 - 1.0
        history.append(torch.cat([state, action], dim=-1))
        state = step(
            state, action,
            factor_tensor[:, 0], factor_tensor[:, 1],
            factor_tensor[:, 2], factor_tensor[:, 3],
        )

    current_action = torch.rand(batch_size, 1, generator=generator) * 2.0 - 1.0
    target = step(
        state, current_action,
        factor_tensor[:, 0], factor_tensor[:, 1],
        factor_tensor[:, 2], factor_tensor[:, 3],
    )
    return {
        "history": torch.cat(history, dim=-1),
        "state": state,
        "action": current_action,
        "next_state": target,
        "factors": factor_tensor,
    }
