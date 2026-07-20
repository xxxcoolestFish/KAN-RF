"""Native Jacobian and path attribution for ProtoKAN layers."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _silu_derivative(x: torch.Tensor) -> torch.Tensor:
    sigmoid = torch.sigmoid(x)
    return sigmoid * (1.0 + x * (1.0 - sigmoid))


def layer_forward_jacobian(layer, x: torch.Tensor):
    """Return a ProtoKAN layer output and its per-sample local Jacobian.

    The returned Jacobian has shape ``(batch, out_dim, in_dim)``.  It is
    derived analytically from prototype interpolation and the SiLU base path.
    """
    sigma = torch.exp(layer.log_sigma).clamp(1e-4, 10.0)
    diff = x.unsqueeze(-1) - layer.proto_pos.view(1, 1, -1)
    logits = -diff.pow(2) / (2.0 * sigma.pow(2))
    weights = torch.softmax(logits, dim=-1)
    preds = (
        layer.proto_val.unsqueeze(0)
        + layer.proto_der.unsqueeze(0) * diff.unsqueeze(1)
    )
    edge_out = (weights.unsqueeze(1) * preds).sum(dim=-1)
    proto_out = edge_out.sum(dim=-1)
    base_out = F.silu(x) @ layer.base_weight.T

    d_logits = -diff / sigma.pow(2)
    weighted_mean = (weights * d_logits).sum(dim=-1, keepdim=True)
    d_weights = weights * (d_logits - weighted_mean)
    d_edge = (
        d_weights.unsqueeze(1) * preds
        + weights.unsqueeze(1) * layer.proto_der.unsqueeze(0)
    ).sum(dim=-1)
    d_base = layer.base_weight.unsqueeze(0) * _silu_derivative(x).unsqueeze(1)
    local_jacobian = d_edge + d_base
    return base_out + proto_out, local_jacobian


def protokan_forward_jacobian(network, x: torch.Tensor):
    """Return ProtoKAN output and the full input-output Jacobian."""
    current = x
    batch, input_dim = x.shape
    total = torch.eye(input_dim, dtype=x.dtype, device=x.device)
    total = total.unsqueeze(0).expand(batch, -1, -1).clone()
    for layer in network.layers:
        current, local = layer_forward_jacobian(layer, current)
        total = torch.bmm(local, total)
    return current, total


def cognitive_forward_jacobian(cognitive, state: torch.Tensor, action: torch.Tensor):
    """Convenience wrapper for a SimpleCognitiveKAN-like model."""
    x = torch.cat([state, action], dim=-1)
    return protokan_forward_jacobian(cognitive.network, x)


def native_temporal_action_effect(cognitive, state: torch.Tensor, actions: torch.Tensor):
    """Propagate the first-action Jacobian through a fixed model rollout.

    Returns ``(batch, horizon, state_dim)`` for ``d s_{t+k} / d a_t``.
    """
    current = state
    influence = None
    effects = []
    state_dim = state.shape[-1]
    for index in range(actions.shape[1]):
        current, jacobian = cognitive_forward_jacobian(
            cognitive, current, actions[:, index],
        )
        state_jacobian = jacobian[:, :, :state_dim]
        action_jacobian = jacobian[:, :, state_dim:]
        if influence is None:
            influence = action_jacobian
        else:
            influence = torch.bmm(state_jacobian, influence)
        effects.append(influence.squeeze(-1))
    return torch.stack(effects, dim=1)
