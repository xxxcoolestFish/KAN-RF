"""Native function-edge traces and causal message routing for ProtoKAN."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from kanrf._protokan import ProtoKAN, ProtoKANLayer


@dataclass
class ProtoKANEdgeTrace:
    inputs: torch.Tensor
    outputs: torch.Tensor
    values: torch.Tensor
    derivatives: torch.Tensor
    positive_response: torch.Tensor
    negative_response: torch.Tensor
    curvature: torch.Tensor


def _edge_values(layer: ProtoKANLayer, x: torch.Tensor) -> torch.Tensor:
    diff = x.unsqueeze(-1) - layer.proto_pos.view(1, 1, -1)
    sigma = torch.exp(layer.log_sigma).clamp(1e-4, 10.0)
    weights = torch.softmax(-diff.square() / (2.0 * sigma.square()), dim=-1)
    predictions = (
        layer.proto_val.unsqueeze(0)
        + layer.proto_der.unsqueeze(0) * diff.unsqueeze(1)
    )
    prototype_edges = (weights.unsqueeze(1) * predictions).sum(dim=-1)
    base_edges = (
        F.silu(x).unsqueeze(1) * layer.base_weight.unsqueeze(0)
    )
    return prototype_edges + base_edges


def edge_values_and_derivatives(layer: ProtoKANLayer, x: torch.Tensor,
                                delta: float = 0.05):
    """Evaluate every function edge and its exact first derivative."""
    diff = x.unsqueeze(-1) - layer.proto_pos.view(1, 1, -1)
    sigma = torch.exp(layer.log_sigma).clamp(1e-4, 10.0)
    weights = torch.softmax(-diff.square() / (2.0 * sigma.square()), dim=-1)
    predictions = (
        layer.proto_val.unsqueeze(0)
        + layer.proto_der.unsqueeze(0) * diff.unsqueeze(1)
    )
    values = (weights.unsqueeze(1) * predictions).sum(dim=-1)

    logit_derivative = -diff / sigma.square()
    centered = logit_derivative - (
        weights * logit_derivative
    ).sum(dim=-1, keepdim=True)
    weight_derivative = weights * centered
    derivatives = (
        weight_derivative.unsqueeze(1) * predictions
        + weights.unsqueeze(1) * layer.proto_der.unsqueeze(0)
    ).sum(dim=-1)

    sigmoid = torch.sigmoid(x)
    silu_derivative = sigmoid * (1.0 + x * (1.0 - sigmoid))
    values = values + F.silu(x).unsqueeze(1) * layer.base_weight.unsqueeze(0)
    derivatives = derivatives + (
        silu_derivative.unsqueeze(1) * layer.base_weight.unsqueeze(0)
    )

    positive = _edge_values(layer, x + delta) - values
    negative = _edge_values(layer, x - delta) - values
    curvature = (positive + negative) / (delta * delta)
    return values, derivatives, positive, negative, curvature


def trace_protokan(network: ProtoKAN, x: torch.Tensor,
                   delta: float = 0.05):
    traces = []
    hidden = x
    for layer in network.layers:
        values, derivatives, positive, negative, curvature = (
            edge_values_and_derivatives(layer, hidden, delta)
        )
        output = values.sum(dim=-1)
        traces.append(ProtoKANEdgeTrace(
            inputs=hidden,
            outputs=output,
            values=values,
            derivatives=derivatives,
            positive_response=positive,
            negative_response=negative,
            curvature=curvature,
        ))
        hidden = output
    return hidden, traces


def linear_causal_route(traces: list[ProtoKANEdgeTrace],
                        output_message: torch.Tensor):
    """Exact Jacobian-vector routing through native ProtoKAN edges."""
    message = output_message
    layer_messages = []
    for trace in reversed(traces):
        message = (
            trace.derivatives * message.unsqueeze(-1)
        ).sum(dim=1)
        layer_messages.append(message)
    return message, list(reversed(layer_messages))


class ProtoKANNonlinearEdgeRouter(nn.Module):
    """Learned residual routing over nonlinear ProtoKAN edge responses.

    With the residual gate initialized to zero this is exactly the analytic
    Jacobian route.  Training can then use value, signed finite responses and
    curvature to depart from the local linear approximation.
    """

    def __init__(self, hidden_dim: int = 16, delta: float = 0.05):
        super().__init__()
        self.delta = delta
        self.edge_gate = nn.Sequential(
            nn.Linear(6, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.edge_gate[-1].weight)
        nn.init.zeros_(self.edge_gate[-1].bias)

    def forward(self, traces: list[ProtoKANEdgeTrace],
                output_message: torch.Tensor):
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
            edge_message = downstream * trace.derivatives + correction
            message = edge_message.sum(dim=1)
            layer_messages.append(message)
        return message, list(reversed(layer_messages))
