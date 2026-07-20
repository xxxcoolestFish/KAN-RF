"""Differentiable formula-graph operator for ProtoKAN.

This is the decision-side interface for a ProtoKAN cognitive predictor.  It
evaluates each learned edge function on an analytic basis, encodes the
resulting function graph into tokens, and keeps the original numerical
prediction path unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from kanrf._protokan import ProtoKAN, ProtoKANLayer


class ProtoKANFormulaOperator(nn.Module):
    """Expose a ProtoKAN prediction together with function-graph tokens."""

    def __init__(self, network: ProtoKAN, basis_count: int = 8,
                 token_dim: int = 32, basis_range: float = 1.0):
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
        self.static_encoders = nn.ModuleList()
        self.dynamic_encoders = nn.ModuleList()
        for layer in network.layers:
            self.static_encoders.append(nn.Sequential(
                nn.LayerNorm(layer.out_dim * layer.in_dim * basis_count),
                nn.Linear(layer.out_dim * layer.in_dim * basis_count, token_dim),
                nn.Tanh(), nn.Linear(token_dim, token_dim),
            ))
            self.dynamic_encoders.append(nn.Sequential(
                nn.LayerNorm(layer.in_dim + layer.out_dim),
                nn.Linear(layer.in_dim + layer.out_dim, token_dim),
                nn.Tanh(), nn.Linear(token_dim, token_dim),
            ))

    @staticmethod
    def edge_values(layer: ProtoKANLayer, basis: torch.Tensor) -> torch.Tensor:
        """Evaluate all ``(out_dim, in_dim)`` edge functions on ``basis``."""
        diff = basis[:, None] - layer.proto_pos[None, :]
        sigma = torch.exp(layer.log_sigma).clamp(1e-4, 10.0)
        weights = torch.softmax(-diff.square() / (2.0 * sigma.square()), dim=-1)
        preds = (
            layer.proto_val[:, :, None, :] +
            layer.proto_der[:, :, None, :] * diff[None, None, :, :]
        )
        values = (weights[None, None, :, :] * preds).sum(dim=-1)
        values = values + layer.base_weight[:, :, None] * F.silu(basis)[None, None, :]
        return values

    def layer_formula_values(self, layer_index: int) -> torch.Tensor:
        return self.edge_values(self.network.layers[layer_index], self.basis)

    def forward(self, x: torch.Tensor, return_tokens: bool = False):
        h = x
        tokens = []
        for index, layer in enumerate(self.network.layers):
            y = layer(h)
            signature = self.layer_formula_values(index).reshape(1, -1)
            static = self.static_encoders[index](signature).expand(h.shape[0], -1)
            dynamic = self.dynamic_encoders[index](torch.cat([h, y], dim=-1))
            tokens.append(static + dynamic)
            h = y
        if return_tokens:
            return h, torch.stack(tokens, dim=1)
        return h

    def edge_formula(self, layer_index: int, out_index: int,
                     in_index: int) -> str:
        layer = self.network.layers[layer_index]
        if not (0 <= out_index < layer.out_dim and 0 <= in_index < layer.in_dim):
            raise IndexError("edge index outside layer dimensions")
        terms = []
        for n in range(layer.n_prototypes):
            terms.append(
                f"softmax_n(-(x-p_{n})^2/(2*sigma^2))"
                f"*(v_{{{out_index},{in_index},{n}}}"
                f"+d_{{{out_index},{in_index},{n}}}*(x-p_{n}))"
            )
        return (
            f"phi_{layer_index}_{out_index}_{in_index}(x)="
            f"w_{{{out_index},{in_index}}}*SiLU(x)+" + "+".join(terms)
        )

    def export(self, path: str | Path | None = None) -> dict:
        data = {
            "type": "ProtoKANFormulaOperator",
            "basis_count": self.basis_count,
            "basis_range": self.basis_range,
            "token_dim": self.token_dim,
            "layers": [],
        }
        for index, layer in enumerate(self.network.layers):
            data["layers"].append({
                "layer": index,
                "in_dim": layer.in_dim,
                "out_dim": layer.out_dim,
                "n_prototypes": layer.n_prototypes,
                "proto_pos": layer.proto_pos.detach().cpu().tolist(),
                "log_sigma": float(layer.log_sigma.detach().cpu()),
                "edges": [
                    self.edge_formula(index, o, i)
                    for o in range(layer.out_dim)
                    for i in range(layer.in_dim)
                ],
            })
        if path is not None:
            Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
        return data
