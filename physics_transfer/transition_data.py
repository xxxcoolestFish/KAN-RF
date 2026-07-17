"""Atomic transition data for the streaming cognitive interface.

The basic record is always ``(state_t, action_t, next_state_t)``.  Sequence
sampling is only a batching convenience for training an online updater; it does
not change the data unit or require a fixed history length at inference.
"""

from __future__ import annotations

import torch

from .multifactor_data import _random_states
from .variants import step


def _factor_batch(batch_size: int, factors, device=None):
    index = torch.randint(len(factors), (batch_size,), device=device)
    table = torch.tensor(factors, dtype=torch.float32, device=device)
    return table[index]


def sample_transition_batch(batch_size: int, factors, device=None):
    """Sample independent one-step records ``(s_t, a_t, s_{t+1})``."""
    factor_tensor = _factor_batch(batch_size, factors, device=device)
    state = _random_states(batch_size).to(device=device)
    action = torch.rand(batch_size, 1, device=device) * 2.0 - 1.0
    next_state = step(
        state, action, factor_tensor[:, 0], factor_tensor[:, 1],
        factor_tensor[:, 2], factor_tensor[:, 3],
    )
    return {
        "state": state,
        "action": action,
        "next_state": next_state,
        "factors": factor_tensor,
    }


def sample_transition_sequence_batch(batch_size: int, sequence_steps: int,
                                     factors, device=None):
    """Sample episodes as stacks of atomic transition records.

    All transitions in one sequence share a hidden environment context, while
    the context is never returned as a model input.  This is the minimal setup
    needed to train a streaming estimator without concatenating a fixed window.
    """
    factor_tensor = _factor_batch(batch_size, factors, device=device)
    state = _random_states(batch_size).to(device=device)
    states, actions, next_states = [], [], []
    for _ in range(sequence_steps):
        action = torch.rand(batch_size, 1, device=device) * 2.0 - 1.0
        next_state = step(
            state, action, factor_tensor[:, 0], factor_tensor[:, 1],
            factor_tensor[:, 2], factor_tensor[:, 3],
        )
        states.append(state)
        actions.append(action)
        next_states.append(next_state)
        state = next_state
    return {
        "state": torch.stack(states, dim=1),
        "action": torch.stack(actions, dim=1),
        "next_state": torch.stack(next_states, dim=1),
        "factors": factor_tensor,
    }
