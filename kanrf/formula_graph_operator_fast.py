"""Efficient formula-graph operator for ProtoKAN.

Compared with the first prototype, this version encodes each edge function
with a shared small encoder and pools edge tokens.  The complete function
graph still participates, but the large flattened edge matrix is avoided.
The detached edge signatures are cached while the cognitive trunk is frozen
and can be invalidated after online cognition updates.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from kanrf._protokan import ProtoKAN, ProtoKANLayer


class ProtoKANFormulaOperatorFast(nn.Module):
    def __init__(self, network: ProtoKAN, basis_count: int = 8,
                 token_dim: int = 32, edge_hidden: int = 16,
                 basis_range: float = 1.0):
        super().__init__()
        if basis_count < 2 or token_dim < 1:
            raise ValueError("basis_count >= 2 and token_dim >= 1 are required")
        self.network = network
        self.basis_count = basis_count
        self.token_dim = token_dim
        self.basis_range = basis_range
        self.register_buffer("basis", torch.linspace(
            -basis_range, basis_range, basis_count,
        ))
        self.edge_encoders = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(basis_count),
                nn.Linear(basis_count, edge_hidden),
                nn.Tanh(),
                nn.Linear(edge_hidden, token_dim),
            )
            for _ in network.layers
        ])
        self.dynamic_encoders = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(layer.in_dim + layer.out_dim),
                nn.Linear(layer.in_dim + layer.out_dim, token_dim),
                nn.Tanh(), nn.Linear(token_dim, token_dim),
            )
            for layer in network.layers
        ])
        self._signature_cache: list[torch.Tensor | None] = [
            None for _ in network.layers
        ]

    @staticmethod
    def edge_values(layer: ProtoKANLayer, basis: torch.Tensor) -> torch.Tensor:
        diff = basis[:, None] - layer.proto_pos[None, :]
        sigma = torch.exp(layer.log_sigma).clamp(1e-4, 10.0)
        weights = torch.softmax(-diff.square() / (2.0 * sigma.square()), dim=-1)
        preds = (
            layer.proto_val[:, :, None, :] +
            layer.proto_der[:, :, None, :] * diff[None, None, :, :]
        )
        values = (weights[None, None, :, :] * preds).sum(dim=-1)
        return values + layer.base_weight[:, :, None] * F.silu(basis)[None, None, :]

    @torch.no_grad()
    def _signature(self, index: int) -> torch.Tensor:
        if self._signature_cache[index] is None:
            values = self.edge_values(self.network.layers[index], self.basis)
            self._signature_cache[index] = values.reshape(-1, self.basis_count)
        return self._signature_cache[index]

    def clear_cache(self):
        self._signature_cache = [None for _ in self._signature_cache]

    def layer_formula_values(self, layer_index: int) -> torch.Tensor:
        """Return the current (uncached) edge-function values."""
        return self.edge_values(self.network.layers[layer_index], self.basis)

    def forward(self, x: torch.Tensor, return_tokens: bool = False):
        h = x
        tokens = []
        for index, layer in enumerate(self.network.layers):
            y = layer(h)
            edge_tokens = self.edge_encoders[index](self._signature(index))
            static = edge_tokens.mean(dim=0, keepdim=True).expand(h.shape[0], -1)
            dynamic = self.dynamic_encoders[index](torch.cat([h, y], dim=-1))
            tokens.append(static + dynamic)
            h = y
        if return_tokens:
            return h, torch.stack(tokens, dim=1)
        return h
