"""Canonical, anchor-relative formula operator for ProtoKAN.

The source cognitive function is captured as an anchor.  Decision-side
tokens contain a stable source-function component and a normalized residual
component for the currently adapted cognitive function.  Edge statistics are
pooled over the function graph, so hidden-edge ordering does not define the
token coordinate system.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from kanrf._protokan import ProtoKAN, ProtoKANLayer


class CanonicalFormulaOperator(nn.Module):
    """Expose a ProtoKAN prediction and stable anchor-relative graph tokens."""

    def __init__(self, network: ProtoKAN, basis_count: int = 8,
                 token_dim: int = 32, basis_range: float = 1.0,
                 edge_hidden: int = 32):
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

        # mean, standard deviation, absolute mean, maximum, minimum per basis
        self.stat_dim = 5 * basis_count
        self.base_projectors = nn.ModuleList()
        self.residual_projectors = nn.ModuleList()
        self.dynamic_encoders = nn.ModuleList()
        for layer in network.layers:
            self.base_projectors.append(nn.Sequential(
                nn.LayerNorm(self.stat_dim),
                nn.Linear(self.stat_dim, edge_hidden), nn.Tanh(),
                nn.Linear(edge_hidden, token_dim),
            ))
            self.residual_projectors.append(nn.Sequential(
                nn.LayerNorm(self.stat_dim),
                nn.Linear(self.stat_dim, edge_hidden), nn.Tanh(),
                nn.Linear(edge_hidden, token_dim),
            ))
            self.dynamic_encoders.append(nn.Sequential(
                nn.LayerNorm(layer.in_dim + layer.out_dim),
                nn.Linear(layer.in_dim + layer.out_dim, token_dim),
                nn.Tanh(), nn.Linear(token_dim, token_dim),
            ))

        # Capture the source function after cognitive pretraining.  Buffers
        # are deliberately detached: the anchor never moves online.
        for index, layer in enumerate(network.layers):
            anchor = self._statistics(self.edge_values(layer, self.basis)).detach()
            scale = anchor.abs() + 0.05
            self.register_buffer(f"anchor_stats_{index}", anchor)
            self.register_buffer(f"anchor_scale_{index}", scale)
        self._current_cache: list[torch.Tensor | None] = [
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

    @staticmethod
    def _statistics(values: torch.Tensor) -> torch.Tensor:
        # Pool all edges.  This removes dependence on arbitrary hidden-edge
        # ordering while retaining the function response at every basis point.
        flat = values.reshape(-1, values.shape[-1])
        return torch.cat([
            flat.mean(dim=0),
            flat.std(dim=0, unbiased=False),
            flat.abs().mean(dim=0),
            flat.amax(dim=0),
            flat.amin(dim=0),
        ], dim=-1).unsqueeze(0)

    @torch.no_grad()
    def _current_statistics(self, index: int) -> torch.Tensor:
        if self._current_cache[index] is None:
            values = self.edge_values(self.network.layers[index], self.basis)
            self._current_cache[index] = self._statistics(values)
        return self._current_cache[index]

    def clear_cache(self):
        self._current_cache = [None for _ in self._current_cache]

    def token_drift(self) -> torch.Tensor:
        drifts = []
        for index in range(len(self.network.layers)):
            current = self._current_statistics(index)
            anchor = getattr(self, f"anchor_stats_{index}")
            scale = getattr(self, f"anchor_scale_{index}")
            drifts.append(((current - anchor) / scale).square().mean().sqrt())
        return torch.stack(drifts).mean()

    def layer_formula_values(self, layer_index: int) -> torch.Tensor:
        return self.edge_values(self.network.layers[layer_index], self.basis)

    def forward(self, x: torch.Tensor, return_tokens: bool = False):
        h = x
        tokens = []
        for index, layer in enumerate(self.network.layers):
            y = layer(h)
            anchor = getattr(self, f"anchor_stats_{index}")
            scale = getattr(self, f"anchor_scale_{index}")
            current = self._current_statistics(index)
            residual = torch.clamp((current - anchor) / scale, -5.0, 5.0)
            base_token = self.base_projectors[index](anchor)
            residual_token = self.residual_projectors[index](residual)
            static = (base_token + residual_token).expand(h.shape[0], -1)
            dynamic = self.dynamic_encoders[index](torch.cat([h, y], dim=-1))
            tokens.append(static + dynamic)
            h = y
        if return_tokens:
            return h, torch.stack(tokens, dim=1)
        return h
