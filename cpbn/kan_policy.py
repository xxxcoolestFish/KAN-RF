"""KAN-based policy network with B-spline features and smoothness prior."""

from __future__ import annotations

import torch
import torch.nn as nn
from itertools import combinations

from kanrf import bspline_basis


class BsplineFeatureExtractor(nn.Module):
    """B-spline feature expansion + pair interactions (same as cognition dict)."""

    def __init__(
        self,
        state_dim: int,
        state_scale,
        *,
        grid_size: int = 4,
        spline_order: int = 2,
        pair_modes: int = 3,
        learnable: bool = True,
    ):
        super().__init__()
        state_scale = torch.as_tensor(state_scale, dtype=torch.float32)
        self.state_dim = state_dim
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.contrast_dim = grid_size + spline_order - 1
        self.pair_modes = min(pair_modes, self.contrast_dim)
        self.register_buffer("state_scale", state_scale)
        self.register_buffer("grid", torch.linspace(-1.0, 1.0, grid_size + 1))

        # DCT compression for pair interactions
        index = torch.arange(self.contrast_dim, dtype=torch.float32)[:, None]
        mode = torch.arange(self.pair_modes, dtype=torch.float32)[None, :]
        projection = torch.cos(torch.pi * (index + 0.5) * mode / self.contrast_dim)
        self.register_buffer("pair_projection",
                             torch.linalg.qr(projection, mode="reduced").Q)

        self.feature_dim = (
            1
            + self.state_dim * self.contrast_dim
            + len(tuple(combinations(range(self.state_dim), 2))) * self.pair_modes**2
        )

        # ── learnable B-spline coefficients (feature_dim × 1 for each output) ─
        if learnable:
            self.coefficients = nn.Parameter(
                torch.zeros(self.feature_dim) * 0.01,
            )

    def forward(self, state):
        normalized = (state / self.state_scale).clamp(-1.0, 1.0)
        local = bspline_basis(
            normalized.reshape(-1), self.grid, self.spline_order,
        ).reshape(*state.shape, -1)[..., :-1]
        blocks = [local.flatten(start_dim=-2)]
        compressed = local @ self.pair_projection
        for first, second in combinations(range(self.state_dim), 2):
            blocks.append(torch.einsum(
                "...i,...j->...ij",
                compressed[..., first, :],
                compressed[..., second, :],
            ).flatten(start_dim=-2))
        constant = torch.ones(*state.shape[:-1], 1, device=state.device)
        return torch.cat((constant, *blocks), dim=-1)

    def smoothness_loss(self):
        """Laplacian smoothness penalty on per-dimension B-spline coefficients."""
        if not hasattr(self, "coefficients") or self.coefficients is None:
            return torch.tensor(0.0, device=self.grid.device)
        loss = torch.tensor(0.0, device=self.grid.device)
        for dim_idx in range(self.state_dim):
            start = 1 + dim_idx * self.contrast_dim
            end = start + self.contrast_dim
            coeff = self.coefficients[start:end]
            diff = coeff[1:] - coeff[:-1]
            loss = loss + (diff**2).sum()
        return loss


class KANPolicy(nn.Module):
    """PPO-compatible KAN policy: B-spline features → linear actor + MLP critic."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        state_scale,
        *,
        grid_size: int = 5,
        spline_order: int = 3,
        pair_modes: int = 3,
        critic_hidden: int = 256,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.extractor = BsplineFeatureExtractor(
            state_dim=state_dim,
            state_scale=state_scale,
            grid_size=grid_size,
            spline_order=spline_order,
            pair_modes=pair_modes,
            learnable=True,
        )
        feature_dim = self.extractor.feature_dim

        # ── Actor: linear on B-spline features ────────────────────────
        self.actor = nn.Linear(feature_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        # ── Critic: MLP on B-spline features ──────────────────────────
        self.critic = nn.Sequential(
            nn.Linear(feature_dim, critic_hidden),
            nn.ReLU(),
            nn.Linear(critic_hidden, critic_hidden),
            nn.ReLU(),
            nn.Linear(critic_hidden, 1),
        )

    def features(self, obs):
        return self.extractor(obs)

    def forward_actor(self, obs, deterministic=False):
        feat = self.features(obs)
        mean = self.actor(feat)
        std = self.log_std.exp().expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        return mean if deterministic else dist.sample()

    def forward_critic(self, obs):
        return self.critic(self.features(obs))

    def evaluate_actions(self, obs, actions):
        feat = self.features(obs)
        mean = self.actor(feat)
        std = self.log_std.exp().expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        values = self.critic(feat)
        return values, log_prob, entropy

    def smoothness_loss(self):
        return self.extractor.smoothness_loss()
