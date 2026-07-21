"""Implicit cognition-to-policy parameter transport for a feedback Actor."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import torch

from cpbn.corridor_policy import future_corridor
from cpbn.feedback_phase import bounded_nearest_phase
from cpbn.receding_tube import LOCAL_METRIC
from cpbn.time_varying_tube import tangent_error


@dataclass
class TransportDiagnostics:
    source_objective: float
    target_objective: float
    gradient_difference_norm: float
    raw_update_norm: float
    transported_update_norm: float
    fisher_metric_norm: float
    trust_scale: float
    parameter_block_norms: dict[str, float]


def transported_parameters(actor):
    """Return policy-mean parameters; exploration scale is not transported."""
    return [
        (name, parameter)
        for name, parameter in actor.named_parameters()
        if name != "log_std"
    ]


def closed_loop_corridor_objective(
    actor,
    dynamics,
    reference: torch.Tensor,
    initial_state: torch.Tensor,
    initial_phase: torch.Tensor,
    rollout_steps: int = 6,
    corridor_horizon: int = 12,
    action_penalty: float = 0.002,
) -> torch.Tensor:
    """Short-horizon closed-loop route objective under a cognitive model."""
    state = initial_state
    phase = initial_phase
    metric = LOCAL_METRIC.to(state)
    objective = torch.zeros((), dtype=state.dtype, device=state.device)
    for step in range(rollout_steps):
        corridor = future_corridor(reference, phase, corridor_horizon)
        raw_action = actor.distribution(state, corridor).mean
        action = torch.tanh(raw_action)
        next_state = dynamics(state, action)
        next_phase = bounded_nearest_phase(
            next_state.detach(), reference, phase,
            backtrack=4, advance=12,
        )
        desired_phase = (next_phase + 1).clamp_max(reference.shape[0] - 1)
        error = tangent_error(next_state, reference[desired_phase])
        route_cost = (error * metric).square().sum(dim=-1)
        effort = action.square().sum(dim=-1)
        objective = objective + (
            route_cost + action_penalty * effort
        ).mean()
        state = next_state
        phase = next_phase
    return objective / rollout_steps


def diagonal_action_fisher(
    actor,
    reference: torch.Tensor,
    state: torch.Tensor,
    phase: torch.Tensor,
    corridor_horizon: int,
    draws: int = 4,
    seed: int = 0,
) -> tuple[torch.Tensor, ...]:
    """Hutchinson estimate of the deterministic action Jacobian diagonal."""
    named = transported_parameters(actor)
    parameters = tuple(parameter for _, parameter in named)
    corridor = future_corridor(reference, phase, corridor_horizon)
    mean = actor.distribution(state, corridor).mean
    generator = torch.Generator(device=mean.device).manual_seed(seed)
    diagonal = [torch.zeros_like(parameter) for parameter in parameters]
    for draw in range(draws):
        sign = torch.randint(
            0, 2, mean.shape, generator=generator, device=mean.device,
        ).to(mean.dtype).mul_(2.0).sub_(1.0)
        projection = (mean * sign).sum() / math.sqrt(mean.shape[0])
        gradient = torch.autograd.grad(
            projection,
            parameters,
            retain_graph=draw + 1 < draws,
            allow_unused=False,
        )
        for index, value in enumerate(gradient):
            diagonal[index] = diagonal[index] + value.detach().square()
    return tuple(value / draws for value in diagonal)


def implicit_transport_delta(
    actor,
    source_dynamics,
    target_dynamics,
    reference: torch.Tensor,
    state: torch.Tensor,
    phase: torch.Tensor,
    rollout_steps: int = 6,
    corridor_horizon: int = 12,
    fisher_draws: int = 4,
    damping: float = 0.03,
    trust_radius: float = 0.20,
    seed: int = 0,
) -> tuple[dict[str, torch.Tensor], TransportDiagnostics]:
    """Compute a damped diagonal-Fisher approximation to implicit transport."""
    named = transported_parameters(actor)
    names = [name for name, _ in named]
    parameters = tuple(parameter for _, parameter in named)
    source_loss = closed_loop_corridor_objective(
        actor, source_dynamics, reference, state, phase,
        rollout_steps, corridor_horizon,
    )
    source_gradient = torch.autograd.grad(source_loss, parameters)
    target_loss = closed_loop_corridor_objective(
        actor, target_dynamics, reference, state, phase,
        rollout_steps, corridor_horizon,
    )
    target_gradient = torch.autograd.grad(target_loss, parameters)
    fisher = diagonal_action_fisher(
        actor, reference, state, phase, corridor_horizon,
        fisher_draws, seed,
    )
    gradient_difference = tuple(
        target.detach() - source.detach()
        for source, target in zip(source_gradient, target_gradient)
    )
    raw_delta = tuple(
        -gradient / (curvature + damping)
        for gradient, curvature in zip(gradient_difference, fisher)
    )
    raw_norm = torch.stack([
        value.square().sum() for value in raw_delta
    ]).sum().sqrt()
    metric_norm = torch.stack([
        (value.square() * (curvature + damping)).sum()
        for value, curvature in zip(raw_delta, fisher)
    ]).sum().sqrt()
    scale = min(1.0, trust_radius / max(float(metric_norm), 1e-12))
    transported = tuple(value * scale for value in raw_delta)
    transported_norm = torch.stack([
        value.square().sum() for value in transported
    ]).sum().sqrt()
    gradient_norm = torch.stack([
        value.square().sum() for value in gradient_difference
    ]).sum().sqrt()
    delta = {
        name: value for name, value in zip(names, transported)
    }
    diagnostics = TransportDiagnostics(
        source_objective=float(source_loss.detach()),
        target_objective=float(target_loss.detach()),
        gradient_difference_norm=float(gradient_norm),
        raw_update_norm=float(raw_norm),
        transported_update_norm=float(transported_norm),
        fisher_metric_norm=float(metric_norm),
        trust_scale=scale,
        parameter_block_norms={
            name: float(value.norm()) for name, value in delta.items()
        },
    )
    return delta, diagnostics


def apply_parameter_delta(actor, delta: dict[str, torch.Tensor], scale=1.0):
    """Return an Actor whose effective weights contain the cognitive transport."""
    transported = copy.deepcopy(actor)
    with torch.no_grad():
        for name, parameter in transported.named_parameters():
            if name in delta:
                parameter.add_(scale * delta[name].to(parameter))
    return transported
