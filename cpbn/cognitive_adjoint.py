"""One-step Bellman-adjoint policy with a mandatory cognitive transition."""

from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Normal

from cpbn.corridor_policy import corridor_tokens
from cpbn.time_varying_tube import tangent_coordinates


class CorridorPotential(nn.Module):
    """Scalar task potential over an ordered state corridor."""

    def __init__(self, hidden_dim: int = 64, corridor_horizon: int = 12):
        super().__init__()
        self.corridor_horizon = corridor_horizon
        self.network = nn.Sequential(
            nn.Linear(11 * corridor_horizon, 2 * hidden_dim),
            nn.Tanh(),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state, corridor):
        if corridor.shape[1] != self.corridor_horizon:
            raise ValueError("corridor horizon does not match potential network")
        tokens = corridor_tokens(state, corridor)
        return self.network(tokens.flatten(start_dim=1)).squeeze(-1)


def action_pullback_and_gramian(
    potential: nn.Module,
    cognition: nn.Module,
    state: torch.Tensor,
    corridor: torch.Tensor,
    create_graph: bool,
):
    """Return d[V(F(s,a))]/da and the local one-step B^T B at a=0."""
    anchor = torch.zeros(
        state.shape[0], 1, dtype=state.dtype, device=state.device,
        requires_grad=True,
    )
    predicted = cognition(state.detach(), anchor)
    tangent = tangent_coordinates(predicted)
    control_rows = []
    for dimension in range(tangent.shape[-1]):
        row = torch.autograd.grad(
            tangent[:, dimension].sum(), anchor,
            retain_graph=True, create_graph=False,
        )[0]
        control_rows.append(row.detach())
    control_map = torch.stack(control_rows, dim=1)
    gramian = control_map.square().sum(dim=1)
    next_potential = potential(predicted, corridor.detach())
    pullback = torch.autograd.grad(
        next_potential.sum(), anchor,
        create_graph=create_graph, retain_graph=create_graph,
    )[0]
    return pullback, gramian, next_potential


class BellmanAdjointActor(nn.Module):
    """Actor whose action is a regularized pullback of a scalar value gradient."""

    def __init__(
        self,
        hidden_dim: int = 64,
        corridor_horizon: int = 12,
        log_std: float = 0.0,
        ridge: float = 1e-5,
        log_gain: float = -3.0,
    ):
        super().__init__()
        self.potential = CorridorPotential(hidden_dim, corridor_horizon)
        self.log_gain = nn.Parameter(torch.tensor(float(log_gain)))
        self.log_std = nn.Parameter(torch.tensor([float(log_std)]))
        self.register_buffer("ridge", torch.tensor(float(ridge)))

    def potential_value(self, state, corridor):
        return self.potential(state, corridor)

    def mean_with_diagnostics(self, state, corridor, cognition):
        build_graph = torch.is_grad_enabled()
        with torch.enable_grad():
            pullback, gramian, next_potential = action_pullback_and_gramian(
                self.potential, cognition, state, corridor, build_graph,
            )
            inverse_control = pullback / (gramian + self.ridge)
            unconstrained = self.log_gain.clamp(-7.0, 3.0).exp()
            unconstrained = unconstrained * inverse_control
            mean = 5.0 * torch.tanh(unconstrained / 5.0)
        if not build_graph:
            mean = mean.detach()
            pullback = pullback.detach()
            gramian = gramian.detach()
            next_potential = next_potential.detach()
        return mean, {
            "pullback": pullback,
            "gramian": gramian,
            "next_potential": next_potential,
        }

    def mean(self, state, corridor, cognition):
        return self.mean_with_diagnostics(state, corridor, cognition)[0]

    def distribution(self, state, corridor, cognition):
        mean = self.mean(state, corridor, cognition)
        std = self.log_std.clamp(-5.0, 1.0).exp().expand_as(mean)
        return Normal(mean, std)

    @staticmethod
    def log_prob(distribution, raw, action):
        return (
            distribution.log_prob(raw)
            - torch.log(1.0 - action.square() + 1e-6)
        ).sum(dim=-1)

    def sample(
        self, state, corridor, cognition, deterministic: bool = False,
    ):
        distribution = self.distribution(state, corridor, cognition)
        raw = distribution.mean if deterministic else distribution.rsample()
        action = torch.tanh(raw)
        return action, self.log_prob(distribution, raw, action)

    def evaluate(self, state, corridor, cognition, action):
        distribution = self.distribution(state, corridor, cognition)
        action = action.clamp(-0.999999, 0.999999)
        raw = torch.atanh(action)
        return self.log_prob(distribution, raw, action), distribution.entropy().sum(-1)
