"""Direct stochastic policy whose forward pass consumes a state corridor."""

from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Normal

from cpbn.time_varying_tube import tangent_error


def future_corridor(
    reference: torch.Tensor,
    phase: torch.Tensor,
    horizon: int,
) -> torch.Tensor:
    offset = torch.arange(1, horizon + 1, device=phase.device)
    index = (phase.unsqueeze(-1) + offset).clamp_max(reference.shape[0] - 1)
    return reference.to(phase.device)[index]


def corridor_tokens(
    state: torch.Tensor,
    corridor: torch.Tensor,
) -> torch.Tensor:
    error = tangent_error(corridor, state.unsqueeze(1))
    time = torch.linspace(
        1.0 / corridor.shape[1], 1.0, corridor.shape[1],
        dtype=state.dtype, device=state.device,
    ).view(1, -1, 1).expand(state.shape[0], -1, -1)
    return torch.cat([corridor, error, time], dim=-1)


class CorridorEncoder(nn.Module):
    """Encode ordered corridor tokens without a state-only bypass."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.gru = nn.GRU(11, hidden_dim, batch_first=True)

    def forward(self, state, corridor):
        tokens = corridor_tokens(state, corridor)
        _, hidden = self.gru(tokens)
        return hidden.squeeze(0)


class DirectCorridorActor(nn.Module):
    def __init__(self, hidden_dim: int = 64, log_std: float = 0.0):
        super().__init__()
        self.encoder = CorridorEncoder(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.log_std = nn.Parameter(torch.tensor([log_std]))

    def distribution(self, state, corridor):
        mean = self.head(self.encoder(state, corridor))
        std = self.log_std.clamp(-5.0, 1.0).exp().expand_as(mean)
        return Normal(mean, std)

    @staticmethod
    def log_prob(distribution, raw, action):
        return (
            distribution.log_prob(raw)
            - torch.log(1.0 - action.square() + 1e-6)
        ).sum(dim=-1)

    def sample(self, state, corridor, deterministic=False):
        distribution = self.distribution(state, corridor)
        raw = distribution.mean if deterministic else distribution.rsample()
        action = torch.tanh(raw)
        return action, self.log_prob(distribution, raw, action)

    def evaluate(self, state, corridor, action):
        distribution = self.distribution(state, corridor)
        action = action.clamp(-0.999999, 0.999999)
        raw = torch.atanh(action)
        return self.log_prob(distribution, raw, action), distribution.entropy().sum(-1)


class CorridorCritic(nn.Module):
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.encoder = CorridorEncoder(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state, corridor):
        return self.head(self.encoder(state, corridor)).squeeze(-1)
