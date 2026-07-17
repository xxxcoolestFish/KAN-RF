"""Bounded physical residual wrapper for the separated cognitive predictor."""

from __future__ import annotations

import torch
from torch import nn

from .split_cognitive_v2 import SplitCognitivePredictorV2


class BoundedSplitCognitivePredictor(nn.Module):
    """Bound the physical residual without changing the branch structure."""

    def __init__(self, residual_scale: float = 0.1, **kwargs):
        super().__init__()
        self.residual_scale = residual_scale
        self.core = SplitCognitivePredictorV2(**kwargs)

    def forward(self, history: torch.Tensor, state: torch.Tensor,
                action: torch.Tensor):
        output = self.core(history, state, action)
        residual = self.residual_scale * torch.tanh(output["physics_residual"])
        output["physics_residual"] = residual
        output["next_state"] = output["base_next_state"] + residual
        return output
