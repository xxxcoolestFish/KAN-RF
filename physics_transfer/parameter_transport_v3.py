"""Numerically stable full-parameter transport."""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn


def cognitive_parameter_vector(cognitive: nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in cognitive.parameters()])


class ParameterTransport(nn.Module):
    def __init__(self, cognitive: nn.Module, layer_shapes: Iterable[tuple[int, int]],
                 rank: int = 4, hidden_dim: int = 64, code_dim: int = 32,
                 output_scale: float = 0.05):
        super().__init__()
        self.layer_shapes, self.rank, self.code_dim = tuple(layer_shapes), rank, code_dim
        reference = cognitive_parameter_vector(cognitive)
        self.register_buffer("reference_scale", reference.abs().clamp_min(0.1))
        self.theta_dim = int(reference.numel())
        factor_dim = sum(out_dim * rank + rank * in_dim
                         for out_dim, in_dim in self.layer_shapes)
        self.encoder = nn.Sequential(
            nn.Linear(self.theta_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
        )
        self.code_head, self.factor_head = nn.Linear(hidden_dim, code_dim), nn.Linear(hidden_dim, factor_dim)
        self.output_scale = output_scale

    def forward(self, cognitive: nn.Module):
        theta = cognitive_parameter_vector(cognitive).to(self.reference_scale)
        normalized = theta / self.reference_scale
        normalized = normalized / normalized.square().mean().sqrt().clamp_min(1e-6)
        hidden = self.encoder(normalized)
        code, raw = torch.tanh(self.code_head(hidden)), torch.tanh(self.factor_head(hidden))
        updates, offset = [], 0
        for out_dim, in_dim in self.layer_shapes:
            left_count = out_dim * self.rank
            right_count = self.rank * in_dim
            left = raw[offset:offset + left_count].view(out_dim, self.rank)
            offset += left_count
            right = raw[offset:offset + right_count].view(self.rank, in_dim)
            offset += right_count
            updates.append(self.output_scale * (left @ right))
        return {"code": code, "updates": updates}


class TransportedMLPPolicy(nn.Module):
    def __init__(self, cognitive: nn.Module, transport: ParameterTransport,
                 state_dim: int = 6, goal_dim: int = 6, hidden_dim: int = 32,
                 action_dim: int = 1):
        super().__init__()
        self.cognitive, self.transport = cognitive, transport
        self.in_dim = state_dim + goal_dim
        self.base_weight_1 = nn.Parameter(torch.empty(hidden_dim, self.in_dim))
        self.base_bias_1 = nn.Parameter(torch.zeros(hidden_dim))
        self.base_weight_2 = nn.Parameter(torch.empty(action_dim, hidden_dim))
        self.base_bias_2 = nn.Parameter(torch.zeros(action_dim))
        nn.init.xavier_uniform_(self.base_weight_1)
        nn.init.xavier_uniform_(self.base_weight_2)

    def forward(self, state, goal, use_transport=True):
        if goal.shape[0] == 1 and state.shape[0] != 1:
            goal = goal.expand(state.shape[0], -1)
        x = torch.cat([state, goal], dim=-1)
        if use_transport:
            updates = self.transport(self.cognitive)["updates"]
        else:
            updates = [torch.zeros_like(self.base_weight_1), torch.zeros_like(self.base_weight_2)]
        hidden = torch.tanh(F.linear(x, self.base_weight_1 + updates[0], self.base_bias_1))
        return torch.tanh(F.linear(hidden, self.base_weight_2 + updates[1], self.base_bias_2))

    def transport_code(self):
        return self.transport(self.cognitive)["code"]
