"""Direct policy layers whose action must pass through cognitive Jacobians."""

from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Normal

from cpbn.corridor_policy import corridor_tokens
from cpbn.time_varying_tube import apply_tangent_error, tangent_coordinates


def local_jacobians_batch(
    dynamics: nn.Module,
    centers: torch.Tensor,
    action_anchor: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return tangent-space A=dF/dx and B=dF/da for independent states."""
    if action_anchor is None:
        action_anchor = torch.zeros(
            centers.shape[0], 1, dtype=centers.dtype, device=centers.device,
        )
    with torch.enable_grad():
        error = torch.zeros(
            centers.shape[0], 4, dtype=centers.dtype,
            device=centers.device, requires_grad=True,
        )
        action = action_anchor.detach().clone().requires_grad_(True)
        state = apply_tangent_error(centers.detach(), error)
        output = tangent_coordinates(dynamics(state, action))
        state_rows, action_rows = [], []
        for dimension in range(4):
            state_grad, action_grad = torch.autograd.grad(
                output[:, dimension].sum(), (error, action),
                retain_graph=dimension < 3,
            )
            state_rows.append(state_grad)
            action_rows.append(action_grad)
    state_jacobian = torch.stack(state_rows, dim=1)
    action_jacobian = torch.stack(action_rows, dim=1)
    return state_jacobian.detach(), action_jacobian.detach()


def future_route_jacobians(
    state_jacobians: torch.Tensor,
    action_jacobians: torch.Tensor,
    phase: torch.Tensor,
    horizon: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    offset = torch.arange(horizon, device=phase.device)
    index = (phase.unsqueeze(-1) + offset).clamp_max(
        state_jacobians.shape[0] - 1,
    )
    return (
        state_jacobians.to(phase.device)[index],
        action_jacobians.to(phase.device)[index],
    )


def cognitive_tokens(state, corridor, state_jacobians, action_jacobians):
    batch, horizon = corridor.shape[:2]
    identity = torch.eye(4, dtype=state.dtype, device=state.device)
    centered_a = state_jacobians - identity.view(1, 1, 4, 4)
    return torch.cat([
        corridor_tokens(state, corridor),
        centered_a.reshape(batch, horizon, 16),
        50.0 * action_jacobians.reshape(batch, horizon, 4),
    ], dim=-1)


class PullbackEncoder(nn.Module):
    """Create long-horizon state covectors using an adjoint recurrence."""

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.gru = nn.GRU(31, hidden_dim, batch_first=True)
        self.local_costate = nn.Linear(hidden_dim, 4)
        self.adjoint_gate = nn.Linear(hidden_dim, 1)
        self.summary = nn.Linear(hidden_dim + 4, hidden_dim)

    def forward(self, state, corridor, state_jacobians, action_jacobians):
        tokens = cognitive_tokens(
            state, corridor, state_jacobians, action_jacobians,
        )
        encoded, hidden = self.gru(tokens)
        local = self.local_costate(encoded)
        gate = 0.99 * torch.sigmoid(self.adjoint_gate(encoded))
        costate = torch.zeros(
            state.shape[0], 4, dtype=state.dtype, device=state.device,
        )
        for step in reversed(range(corridor.shape[1])):
            propagated = torch.bmm(
                state_jacobians[:, step].transpose(1, 2),
                costate.unsqueeze(-1),
            ).squeeze(-1)
            costate = local[:, step] + gate[:, step] * propagated
            costate = 8.0 * torch.tanh(costate / 8.0)
        summary = torch.tanh(self.summary(torch.cat([
            hidden.squeeze(0), costate,
        ], dim=-1)))
        return summary, costate


class CognitivePullbackActor(nn.Module):
    """Actor with no action path except B_theta^T times a state covector."""

    def __init__(
        self,
        hidden_dim: int = 64,
        log_std: float = 0.0,
        log_pullback_scale: float = 4.0,
    ):
        super().__init__()
        self.encoder = PullbackEncoder(hidden_dim)
        self.costate_residual = nn.Linear(hidden_dim, 4)
        self.scale_modulation = nn.Linear(hidden_dim, 1)
        self.log_pullback_scale = nn.Parameter(torch.tensor(log_pullback_scale))
        self.log_std = nn.Parameter(torch.tensor([log_std]))

    def mean(self, state, corridor, state_jacobians, action_jacobians):
        summary, costate = self.encoder(
            state, corridor, state_jacobians, action_jacobians,
        )
        costate = costate + self.costate_residual(summary)
        scale_log = self.log_pullback_scale + 0.5 * torch.tanh(
            self.scale_modulation(summary),
        )
        scale = scale_log.clamp(-4.0, 8.0).exp()
        current_b = action_jacobians[:, 0]
        pullback = torch.bmm(
            current_b.transpose(1, 2), costate.unsqueeze(-1),
        ).squeeze(-1)
        return -scale * pullback

    def distribution(self, state, corridor, state_jacobians, action_jacobians):
        mean = self.mean(state, corridor, state_jacobians, action_jacobians)
        std = self.log_std.clamp(-5.0, 1.0).exp().expand_as(mean)
        return Normal(mean, std)

    @staticmethod
    def log_prob(distribution, raw, action):
        return (
            distribution.log_prob(raw)
            - torch.log(1.0 - action.square() + 1e-6)
        ).sum(dim=-1)

    def sample(
        self, state, corridor, state_jacobians, action_jacobians,
        deterministic=False,
    ):
        distribution = self.distribution(
            state, corridor, state_jacobians, action_jacobians,
        )
        raw = distribution.mean if deterministic else distribution.rsample()
        action = torch.tanh(raw)
        return action, self.log_prob(distribution, raw, action)

    def evaluate(self, state, corridor, state_jacobians, action_jacobians, action):
        distribution = self.distribution(
            state, corridor, state_jacobians, action_jacobians,
        )
        action = action.clamp(-0.999999, 0.999999)
        raw = torch.atanh(action)
        return self.log_prob(distribution, raw, action), distribution.entropy().sum(-1)


class CognitivePullbackCritic(nn.Module):
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.encoder = PullbackEncoder(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1),
        )

    def forward(self, state, corridor, state_jacobians, action_jacobians):
        summary, _ = self.encoder(
            state, corridor, state_jacobians, action_jacobians,
        )
        return self.head(summary).squeeze(-1)
