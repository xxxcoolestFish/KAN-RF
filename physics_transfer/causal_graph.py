"""Local and temporal causal-effect extraction from a transition model.

The module deliberately uses interventional finite differences rather than
interpreting parameter magnitude or hidden-unit activation as causality.
"""

from __future__ import annotations

from typing import Callable

import torch


Predictor = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
Transition = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def finite_difference_input_effect(
    predictor: Predictor,
    state: torch.Tensor,
    action: torch.Tensor,
    epsilon: float = 1e-3,
) -> torch.Tensor:
    """Estimate d next_state / d [state, action] by interventions.

    Returns a tensor with shape ``(batch, input_dim, state_dim)``.  The
    predictor is evaluated at ``x + eps`` and ``x - eps`` independently for
    every input coordinate.
    """
    x = torch.cat([state, action], dim=-1)
    effects = []
    for coordinate in range(x.shape[-1]):
        delta = torch.zeros_like(x)
        delta[:, coordinate] = epsilon
        plus = predictor(x[:, : state.shape[-1]] + delta[:, : state.shape[-1]],
                         x[:, state.shape[-1] :] + delta[:, state.shape[-1] :])
        minus = predictor(x[:, : state.shape[-1]] - delta[:, : state.shape[-1]],
                          x[:, state.shape[-1] :] - delta[:, state.shape[-1] :])
        effects.append((plus - minus) / (2.0 * epsilon))
    return torch.stack(effects, dim=1)


def rollout_model(
    predictor: Predictor,
    state: torch.Tensor,
    actions: torch.Tensor,
) -> torch.Tensor:
    """Roll a predictor with a fixed action sequence.

    ``actions`` has shape ``(batch, horizon, action_dim)`` and the returned
    states have shape ``(batch, horizon, state_dim)``.
    """
    current = state
    outputs = []
    for index in range(actions.shape[1]):
        current = predictor(current, actions[:, index])
        outputs.append(current)
    return torch.stack(outputs, dim=1)


def temporal_action_effect(
    transition: Transition,
    state: torch.Tensor,
    actions: torch.Tensor,
    epsilon: float = 1e-3,
    action_time: int = 0,
    action_index: int = 0,
) -> torch.Tensor:
    """Estimate the effect of one action on every future state in a rollout.

    Returns ``(batch, horizon, state_dim)`` for the selected action coordinate.
    The transition may be either the learned cognitive model or the exact
    environment transition.
    """
    plus_actions = actions.clone()
    minus_actions = actions.clone()
    plus_actions[:, action_time, action_index] += epsilon
    minus_actions[:, action_time, action_index] -= epsilon
    plus_states = rollout_model(transition, state, plus_actions)
    minus_states = rollout_model(transition, state, minus_actions)
    return (plus_states - minus_states) / (2.0 * epsilon)


def summarize_effect(effect: torch.Tensor) -> dict:
    """Return robust strength, signed effect and sign stability statistics."""
    if effect.ndim < 2:
        raise ValueError("effect must include a batch dimension")
    signed_mean = effect.mean(dim=0)
    median_abs = effect.abs().median(dim=0).values
    sign_consistency = effect.sign().float().mean(dim=0).abs()
    return {
        "signed_mean": signed_mean.detach().cpu().tolist(),
        "median_abs": median_abs.detach().cpu().tolist(),
        "sign_consistency": sign_consistency.detach().cpu().tolist(),
    }


def cosine_by_sample(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Cosine similarity across the last dimensions, per batch sample."""
    first = first.reshape(first.shape[0], -1)
    second = second.reshape(second.shape[0], -1)
    numerator = (first * second).sum(dim=-1)
    denominator = first.norm(dim=-1) * second.norm(dim=-1)
    return numerator / denominator.clamp_min(1e-8)
