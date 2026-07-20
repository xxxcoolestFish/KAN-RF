"""Stable nonlinear residual routing on top of exact ProtoKAN edge routes."""

from __future__ import annotations

import torch

from kanrf.protokan_causal_router import ProtoKANNonlinearEdgeRouter


class StableProtoKANNonlinearEdgeRouter(ProtoKANNonlinearEdgeRouter):
    def __init__(self, hidden_dim: int = 16, delta: float = 0.05,
                 correction_scale: float = 0.1):
        super().__init__(hidden_dim=hidden_dim, delta=delta)
        self.correction_scale = correction_scale

    def forward(self, traces, output_message):
        message = output_message
        layer_messages = []
        for trace in reversed(traces):
            downstream = message.unsqueeze(-1).expand_as(trace.values)
            features = torch.stack([
                downstream,
                trace.values,
                trace.derivatives,
                trace.positive_response / self.delta,
                trace.negative_response / self.delta,
                trace.curvature * self.delta,
            ], dim=-1)
            correction = self.edge_gate(features).squeeze(-1)
            linear = (downstream * trace.derivatives).sum(dim=1)
            residual = self.correction_scale * correction.mean(dim=1)
            message = linear + residual
            layer_messages.append(message)
        return message, list(reversed(layer_messages))
