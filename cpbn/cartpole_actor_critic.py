"""Feedback actor-critic components for continuous CartPole."""

from __future__ import annotations

import torch
from torch import nn


def cartpole_task_reward(state):
    """Dimensionless reward derived only from the formal terminal tolerances."""
    normalized = torch.stack((
        state[..., 0] / 0.35,
        state[..., 1] / 0.08,
    ), dim=-1)
    return torch.exp(-0.5 * normalized.square().sum(dim=-1))


def cartpole_path_failure(state):
    """Whether a state violates the formal path constraints or is non-finite."""
    return (
        ~torch.isfinite(state).all(dim=-1)
        | (state[..., 0].abs() >= 2.4)
        | (state[..., 1].abs() >= 0.35)
    )


class _CartPoleQ(nn.Module):
    def __init__(self, hidden_dim=256, force_limit=12.0):
        super().__init__()
        self.force_limit = force_limit
        self.register_buffer("state_scale", torch.tensor([2.4, 0.5, 3.0, 3.0]))
        self.network = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state, action):
        features = torch.cat((
            state / self.state_scale,
            action / self.force_limit,
        ), dim=-1)
        return self.network(features).squeeze(-1)


class TwinCartPoleQ(nn.Module):
    """Twin action-value networks used by clipped double-Q feedback learning."""

    def __init__(self, hidden_dim=256, force_limit=12.0):
        super().__init__()
        self.q1 = _CartPoleQ(hidden_dim, force_limit)
        self.q2 = _CartPoleQ(hidden_dim, force_limit)

    def forward(self, state, action):
        return self.q1(state, action), self.q2(state, action)


class CartPoleStateValue(nn.Module):
    """State value used when action effects are supplied by a dynamics model."""

    def __init__(self, hidden_dim=256):
        super().__init__()
        self.register_buffer("state_scale", torch.tensor([2.4, 0.5, 3.0, 3.0]))
        self.network = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state):
        return self.network(state / self.state_scale).squeeze(-1)


class TensorTransitionReplay:
    """Fixed-capacity circular replay stored entirely on one torch device."""

    def __init__(self, capacity, device):
        self.capacity = int(capacity)
        self.device = torch.device(device)
        self.state = torch.empty(self.capacity, 4, device=self.device)
        self.action = torch.empty(self.capacity, 1, device=self.device)
        self.reward = torch.empty(self.capacity, device=self.device)
        self.next_state = torch.empty(self.capacity, 4, device=self.device)
        self.done = torch.empty(self.capacity, device=self.device)
        self.position = 0
        self.size = 0

    def __len__(self):
        return self.size

    @torch.no_grad()
    def add(self, state, action, reward, next_state, done):
        if state.shape[0] >= self.capacity:
            state = state[-self.capacity:]
            action = action[-self.capacity:]
            reward = reward[-self.capacity:]
            next_state = next_state[-self.capacity:]
            done = done[-self.capacity:]
        count = state.shape[0]
        first = min(count, self.capacity - self.position)
        second = count - first
        for storage, value in (
            (self.state, state),
            (self.action, action),
            (self.reward, reward),
            (self.next_state, next_state),
            (self.done, done),
        ):
            storage[self.position:self.position + first].copy_(value[:first])
            if second:
                storage[:second].copy_(value[first:])
        self.position = (self.position + count) % self.capacity
        self.size = min(self.capacity, self.size + count)

    def sample(self, count, generator=None):
        index = torch.randint(
            self.size, (count,), device=self.device, generator=generator,
        )
        return (
            self.state[index],
            self.action[index],
            self.reward[index],
            self.next_state[index],
            self.done[index],
        )


@torch.no_grad()
def soft_update(target, online, rate):
    for target_parameter, online_parameter in zip(
        target.parameters(), online.parameters(), strict=True,
    ):
        target_parameter.lerp_(online_parameter, rate)
