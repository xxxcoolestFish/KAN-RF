"""Parameter-space transport from a cognitive predictor to a policy.

The transport module deliberately operates on the complete cognitive
parameter vector.  It does not query the predictor with hand-written action
probes.  Instead, it generates low-rank weight modulations for a decision
decoder.  A separate reconstruction head can be used to calibrate the
transport against the frozen cognitive predictor before policy training.
"""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn


def cognitive_parameter_vector(cognitive: nn.Module) -> torch.Tensor:
    """Return all cognitive parameters in a deterministic flat order."""
    return torch.cat([parameter.detach().reshape(-1) for parameter in cognitive.parameters()])


class ParameterTransport(nn.Module):
    """Map all cognitive parameters to policy-layer low-rank updates.

    The mapper is intentionally small compared with the policy.  Its input is
    the complete cognitive parameter vector, while its output is a collection
    of low-rank factors for the policy receiver layers plus a compact code used
    by the optional transport-reconstruction head.
    """

    def __init__(self, cognitive: nn.Module, layer_shapes: Iterable[tuple[int, int]],
                 rank: int = 4, hidden_dim: int = 64, code_dim: int = 32,
                 scale: float = 0.05):
        super().__init__()
        self.layer_shapes = tuple(layer_shapes)
        self.rank = rank
        self.code_dim = code_dim
        reference = cognitive_parameter_vector(cognitive)
        self.register_buffer("reference_scale", reference.abs().clamp_min(1e-3))
        self.theta_dim = int(reference.numel())
        factor_dim = sum(out_dim * rank + rank * in_dim
                         for out_dim, in_dim in self.layer_shapes)
        self.encoder = nn.Sequential(
            nn.Linear(self.theta_dim, hidden_dim),
            nn.Tanh(),
        )
        self.code_head = nn.Linear(hidden_dim, code_dim)
        self.factor_head = nn.Linear(hidden_dim, factor_dim)
        self.output_scale = scale

    def forward(self, cognitive: nn.Module):
        theta = cognitive_parameter_vector(cognitive).to(self.reference_scale)
        normalized = theta / self.reference_scale
        hidden = self.encoder(normalized)
        code = torch.tanh(self.code_head(hidden))
        raw = torch.tanh(self.factor_head(hidden))
        updates = []
        offset = 0
        for out_dim, in_dim in self.layer_shapes:
            left_count = out_dim * self.rank
            right_count = self.rank * in_dim
            left = raw[offset:offset + left_count].view(out_dim, self.rank)
            offset += left_count
            right = raw[offset:offset + right_count].view(self.rank, in_dim)
            offset += right_count
            updates.append(self.output_scale * (left @ right))
        return {"code": code, "updates": updates, "normalized_parameters": normalized}


class TransportedMLPPolicy(nn.Module):
    """Direct action policy whose effective weights depend on cognition."""

    def __init__(self, cognitive: nn.Module, transport: ParameterTransport,
                 state_dim: int = 6, goal_dim: int = 6, hidden_dim: int = 32,
                 action_dim: int = 1):
        super().__init__()
        self.cognitive = cognitive
        self.transport = transport
        self.in_dim = state_dim + goal_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.base_weight_1 = nn.Parameter(torch.empty(hidden_dim, self.in_dim))
        self.base_bias_1 = nn.Parameter(torch.zeros(hidden_dim))
        self.base_weight_2 = nn.Parameter(torch.empty(action_dim, hidden_dim))
        self.base_bias_2 = nn.Parameter(torch.zeros(action_dim))
        nn.init.xavier_uniform_(self.base_weight_1)
        nn.init.xavier_uniform_(self.base_weight_2)

    def forward(self, state: torch.Tensor, goal: torch.Tensor,
                use_transport: bool = True) -> torch.Tensor:
        if goal.shape[0] == 1 and state.shape[0] != 1:
            goal = goal.expand(state.shape[0], -1)
        x = torch.cat([state, goal], dim=-1)
        if use_transport:
            transported = self.transport(self.cognitive)
            update_1, update_2 = transported["updates"]
        else:
            update_1 = torch.zeros_like(self.base_weight_1)
            update_2 = torch.zeros_like(self.base_weight_2)
        hidden = torch.tanh(F.linear(x, self.base_weight_1 + update_1,
                                     self.base_bias_1))
        logits = F.linear(hidden, self.base_weight_2 + update_2,
                          self.base_bias_2)
        return torch.tanh(logits)

    def transport_code(self):
        return self.transport(self.cognitive)["code"]


class TransportReconstructionHead(nn.Module):
    """Read a next-state prediction from the transported physical code."""

    def __init__(self, code_dim: int, state_dim: int = 6, action_dim: int = 1,
                 hidden_dim: int = 32):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim + code_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, state, action, code):
        return self.network(torch.cat([state, action, code.expand(state.shape[0], -1)], dim=-1))
