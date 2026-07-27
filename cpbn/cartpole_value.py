"""Source-policy value model for continuous CartPole."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from cpbn.continuous_cartpole import (
    cartpole_running_cost,
    cartpole_step,
    sample_cartpole_initial_states,
)


class CartPoleValueCritic(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.register_buffer("state_scale", torch.tensor([2.4, 0.5, 3.0, 3.0]))
        self.network = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state):
        return self.network(state / self.state_scale).squeeze(-1)


@dataclass(frozen=True)
class CartPoleValueDataset:
    state: torch.Tensor
    returns: torch.Tensor


@torch.no_grad()
def collect_source_value_data(
    actor,
    parameters,
    count,
    device,
    *,
    steps=300,
    gamma=0.99,
    generator=None,
):
    state = sample_cartpole_initial_states(count, device, generator=generator)
    states, costs = [], []
    for _ in range(steps):
        action = actor(state)
        next_state = cartpole_step(state, action, parameters)
        states.append(state)
        costs.append(cartpole_running_cost(next_state, action))
        state = next_state
    cost = torch.stack(costs)
    returns = torch.empty_like(cost)
    future = torch.zeros(count, device=device)
    for index in reversed(range(steps)):
        future = cost[index] + gamma * future
        returns[index] = future
    return CartPoleValueDataset(
        state=torch.stack(states).flatten(0, 1),
        returns=returns.flatten(),
    )


@torch.no_grad()
def value_metrics(critic, dataset, mean, scale, batch_size=16384):
    prediction = torch.cat([
        critic(dataset.state[start:start + batch_size])
        for start in range(0, dataset.state.shape[0], batch_size)
    ])
    target = (dataset.returns - mean) / scale
    residual = prediction - target
    target_variance = target.var(unbiased=False).clamp_min(1e-12)
    covariance = (
        (prediction - prediction.mean()) * (target - target.mean())
    ).mean()
    correlation = covariance / (
        prediction.std(unbiased=False) * target.std(unbiased=False)
    ).clamp_min(1e-12)
    return {
        "normalized_rmse": float(residual.square().mean().sqrt()),
        "explained_variance": float(
            1.0 - residual.var(unbiased=False) / target_variance
        ),
        "correlation": float(correlation),
    }
