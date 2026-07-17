"""Losses for the adaptive cognitive representation."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def cognitive_prediction_loss(predicted: torch.Tensor,
                              target: torch.Tensor) -> torch.Tensor:
    return F.smooth_l1_loss(predicted, target)


def token_consistency_loss(pooled_a: torch.Tensor,
                           pooled_b: torch.Tensor,
                           gates_a: torch.Tensor,
                           gates_b: torch.Tensor) -> torch.Tensor:
    """Trajectory-invariant consistency without assigning token semantics."""
    gate_a = torch.sort(gates_a, dim=-1).values
    gate_b = torch.sort(gates_b, dim=-1).values
    return F.mse_loss(pooled_a, pooled_b) + F.mse_loss(gate_a, gate_b)


def gate_sparsity_loss(gates: torch.Tensor) -> torch.Tensor:
    return gates.mean()
