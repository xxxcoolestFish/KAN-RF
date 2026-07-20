"""Differentiable formula-graph interface for ProtoKAN.

The cognitive model remains an ordinary ProtoKAN predictor.  This module
adds a second, decision-side view of the same function: every edge function
is evaluated on a fixed analytic basis and encoded as a graph token.  The
token is a function-level representation (not a raw flattened parameter
vector), while the wrapped network's numerical prediction is unchanged.

The basis is only a representation grid; it is not a list of physical
parameters and has no environment-specific semantic meaning.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from kanrf._protokan import ProtoKAN, ProtoKANLayer


@dataclass(frozen=True)
class FormulaNode:
    """Static metadata for one ProtoKAN layer in the formula graph."""

    layer: int
    in_dim: int
    out_dim: int
    n_prototypes: int


class ProtoKANFormulaGraph(nn.Module):
    """Wrap a ProtoKAN and expose differentiable formula-graph tokens.

    ``forward`` returns exactly the same prediction as the wrapped network.
    When ``return_tokens=True``, it additionally returns a tensor of shape
    ``(batch, n_layers, token_dim)``.  Each token combines:

    * a static encoding of the complete edge-function graph, obtained by
      evaluating every edge on the analytic basis; and
    * a dynamic encoding of the current layer activation.

    All ProtoKAN parameters occur in the edge-function evaluation: prototype
    positions, values, derivatives, kernel width, and the SiLU base weights.
    """

    def __init__(self, network: ProtoKAN, basis_count: int = 8,
                 token_dim: int = 32, basis_range: float = 1.0):
        super().__init__()
        if basis_count < 2:
            raise ValueError("basis_count must be at least 2")
        if token_dim < 1:
            raise ValueError("token_dim must be positive")
        self.network = network
        self.basis_count = basis_count
        self.token_dim = token_dim
        self.basis_range = basis_range
        self.register_buffer(
            "basis",
            torch.linspace(-basis_range, basis_range, basis_count),
        )

        self.nodes = tuple(
            FormulaNode(i, layer.in_dim, layer.out_dim, layer.n_prototypes)
            for i, layer in enumerate(network.layers)
        )
        self.static_encoders = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(node.out_dim * node.in_dim * basis_count),
                nn.Linear(node.out_dim * node.in_dim * basis_count, token_dim),
                nn.Tanh(),
                nn.Linear(token_dim, token_dim),
            )
            for node in self.nodes
        ])
        self.dynamic_encoders = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(node.in_dim + node.out_dim),
                nn.Linear(node.in_dim + node.out_dim, token_dim),
                nn.Tanh(),
                nn.Linear(token_dim, token_dim),
            )
            for node in self.nodes
        ])

    @staticmethod
    def _edge_values(layer: ProtoKANLayer, basis: torch.Tensor) -> torch.Tensor:
        """Evaluate every edge function on the analytic basis.

        Returns ``(out_dim, in_dim, basis_count)``.  This is intentionally
        expressed with the same differentiable formula as ``ProtoKANLayer``
        rather than reading a detached parameter vector.
        """

        # basis: (M,), prototype positions: (N,)
        diff = basis[:, None] - layer.proto_pos[None, :]
        sigma = torch.exp(layer.log_sigma).clamp(1e-4, 10.0)
        weights = torch.softmax(-diff.pow(2) / (2.0 * sigma ** 2), dim=-1)
        # (out_dim, in_dim, M, N)
        preds = (
            layer.proto_val[:, :, None, :] +
            layer.proto_der[:, :, None, :] * diff[None, None, :, :]
        )
        edge = (weights[None, None, :, :] * preds).sum(dim=-1)
        # Add the base SiLU edge path.
        edge = edge + layer.base_weight[:, :, None] * F.silu(basis)[None, None, :]
        return edge.permute(0, 1, 2).contiguous()

    def layer_formula_values(self, layer_index: int) -> torch.Tensor:
        """Return differentiable edge-function values for one layer."""

        return self._edge_values(self.network.layers[layer_index], self.basis)

    def forward(self, x: torch.Tensor, return_tokens: bool = False):
        h = x
        tokens = []
        for index, layer in enumerate(self.network.layers):
            # Numerical path: this is exactly the original ProtoKAN layer.
            y = layer(h)

            # Static graph path: complete edge functions, not flattened raw
            # parameters.  It is expanded over the current batch.
            edge_values = self.layer_formula_values(index).reshape(1, -1)
            static = self.static_encoders[index](edge_values)
            static = static.expand(h.shape[0], -1)

            # Dynamic path binds the graph token to the current state/action
            # representation at this layer.
            dynamic = self.dynamic_encoders[index](torch.cat([h, y], dim=-1))
            tokens.append(static + dynamic)
            h = y

        if return_tokens:
            return h, torch.stack(tokens, dim=1)
        return h

    def nodes_metadata(self) -> list[dict[str, int]]:
        return [node.__dict__.copy() for node in self.nodes]

    def edge_formula(self, layer_index: int, out_index: int,
                     in_index: int) -> str:
        """Return a symbolic template for one ProtoKAN edge function."""

        layer = self.network.layers[layer_index]
        if not (0 <= out_index < layer.out_dim and 0 <= in_index < layer.in_dim):
            raise IndexError("edge index outside layer dimensions")
        terms = []
        for n in range(layer.n_prototypes):
            terms.append(
                "softmax_n(-(x-p_{n})^2/(2*sigma^2))"
                "*(v_{o,i,n}+d_{o,i,n}*(x-p_n))".format(n=n, o=out_index, i=in_index)
            )
        return (
            f"phi_{layer_index}_{out_index}_{in_index}(x)="
            f"w_{out_index}_{in_index}*SiLU(x)+" + "+".join(terms)
        )

    def export(self, path: str | Path | None = None) -> dict[str, Any]:
        """Export graph metadata and symbolic edge templates as JSON data."""

        data: dict[str, Any] = {
            "type": "ProtoKANFormulaGraph",
            "basis_count": self.basis_count,
            "basis_range": self.basis_range,
            "token_dim": self.token_dim,
            "layers": [],
        }
        for index, node in enumerate(self.nodes):
            layer = self.network.layers[index]
            layer_data = {
                **node.__dict__,
                "proto_pos": layer.proto_pos.detach().cpu().tolist(),
                "log_sigma": float(layer.log_sigma.detach().cpu()),
                "edges": [
                    self.edge_formula(index, o, i)
                    for o in range(layer.out_dim)
                    for i in range(layer.in_dim)
                ],
            }
            data["layers"].append(layer_data)
        if path is not None:
            Path(path).write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        return data
