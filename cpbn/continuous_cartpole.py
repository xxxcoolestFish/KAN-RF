"""Differentiable batched CartPole with continuous force control."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class CartPoleParameters:
    cart_mass: float = 1.0
    pole_mass: float = 0.1
    half_length: float = 0.5
    gravity: float = 9.8
    cart_friction: float = 0.0
    pole_friction: float = 0.0
    actuator_scale: float = 1.0


class CartPoleActor(nn.Module):
    def __init__(self, hidden_dim: int = 96, force_limit: float = 12.0):
        super().__init__()
        self.force_limit = force_limit
        self.network = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state):
        scale = state.new_tensor([2.4, 0.5, 3.0, 3.0])
        return self.force_limit * torch.tanh(self.network(state / scale))


def cartpole_acceleration(state, action, parameters: CartPoleParameters):
    """Return [cart acceleration, pole angular acceleration]."""
    theta = state[..., 1]
    x_dot = state[..., 2]
    theta_dot = state[..., 3]
    force = parameters.actuator_scale * action[..., 0]
    sine, cosine = theta.sin(), theta.cos()
    total_mass = parameters.cart_mass + parameters.pole_mass
    polemass_length = parameters.pole_mass * parameters.half_length
    effective_force = force - parameters.cart_friction * x_dot
    temp = (
        effective_force + polemass_length * theta_dot.square() * sine
    ) / total_mass
    theta_acc = (
        parameters.gravity * sine
        - cosine * temp
        - parameters.pole_friction * theta_dot
        / (parameters.pole_mass * parameters.half_length).clamp_min(1e-8)
        if isinstance(parameters.pole_mass, torch.Tensor)
        else parameters.gravity * sine
        - cosine * temp
        - parameters.pole_friction * theta_dot
        / max(parameters.pole_mass * parameters.half_length, 1e-8)
    ) / (
        parameters.half_length
        * (4.0 / 3.0 - parameters.pole_mass * cosine.square() / total_mass)
    )
    x_acc = temp - polemass_length * theta_acc * cosine / total_mass
    return torch.stack((x_acc, theta_acc), dim=-1)


def cartpole_step(
    state,
    action,
    parameters: CartPoleParameters,
    *,
    dt: float = 0.02,
):
    acceleration = cartpole_acceleration(state, action, parameters)
    velocity = state[..., 2:] + dt * acceleration
    position = state[..., :2] + dt * velocity
    return torch.cat((position, velocity), dim=-1)


def sample_cartpole_initial_states(count, device, *, generator=None):
    random = torch.rand(count, 4, device=device, generator=generator) * 2.0 - 1.0
    scale = torch.tensor([0.6, 0.18, 0.35, 0.35], device=device)
    return random * scale


def cartpole_rollout(actor, parameters, initial, *, steps: int = 300):
    state = initial
    trajectory = [state]
    actions = []
    for _ in range(steps):
        action = actor(state)
        state = cartpole_step(state, action, parameters)
        trajectory.append(state)
        actions.append(action)
    return torch.stack(trajectory), torch.stack(actions)


def cartpole_rollout_cost(trajectory, actions):
    state = trajectory[1:]
    return cartpole_running_cost(state, actions).mean()


def cartpole_running_cost(state, action):
    """Per-transition task cost used to train the source actor."""
    running = (
        1.0 * state[..., 0].square()
        + 18.0 * state[..., 1].square()
        + 0.08 * state[..., 2].square()
        + 0.15 * state[..., 3].square()
        + 2e-3 * action[..., 0].square()
    )
    boundary = (
        5.0 * torch.relu(state[..., 0].abs() - 2.0).square()
        + 20.0 * torch.relu(state[..., 1].abs() - 0.35).square()
    )
    return running + boundary


def cartpole_completion_ratio(trajectory):
    """Maximum normalized violation ratio of the formal success conditions."""
    state = trajectory[1:]
    path_ratio = torch.stack((
        state[..., 0].abs() / 2.4,
        state[..., 1].abs() / 0.35,
    ), dim=-1).amax(dim=(0, 2))
    terminal = state[-min(50, state.shape[0]):]
    terminal_ratio = torch.stack((
        terminal[..., 0].abs().mean(dim=0) / 0.35,
        terminal[..., 1].abs().mean(dim=0) / 0.08,
    ), dim=-1).amax(dim=-1)
    return torch.maximum(path_ratio, terminal_ratio)


@torch.no_grad()
def cartpole_success_rate(trajectory):
    state = trajectory[1:]
    finite = torch.isfinite(state).all(dim=(0, 2))
    inside = (
        (state[..., 0].abs() < 2.4)
        & (state[..., 1].abs() < 0.35)
    ).all(dim=0)
    terminal = (
        state[-50:, :, 0].abs().mean(dim=0) < 0.35
    ) & (
        state[-50:, :, 1].abs().mean(dim=0) < 0.08
    )
    return float((finite & inside & terminal).float().mean())
