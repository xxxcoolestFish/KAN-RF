"""Actor layers whose only environment-dependent path is low-rank modulation."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class LowRankCognitiveLinear(nn.Module):
    """A linear layer with a cognition-gated, rank-limited weight update.

    For one sample with detached cognition ``c``, the layer computes

    ``Wx + b + scale * A diag(Tc) Bx``.

    Thus cognition cannot enter through an unconstrained feature-concatenation
    shortcut, and the environment-dependent weight update has rank at most
    ``rank``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        cognition_dim: int,
        rank: int = 4,
        modulation_scale: float | None = None,
    ) -> None:
        super().__init__()
        if min(in_features, out_features, cognition_dim, rank) <= 0:
            raise ValueError("all dimensions and rank must be positive")
        self.in_features = in_features
        self.out_features = out_features
        self.cognition_dim = cognition_dim
        self.rank = rank
        self.modulation_scale = (
            rank**-0.5 if modulation_scale is None else modulation_scale
        )

        self.base = nn.Linear(in_features, out_features)
        self.right = nn.Parameter(torch.empty(rank, in_features))
        self.left = nn.Parameter(torch.empty(out_features, rank))
        self.cognition_to_gate = nn.Linear(cognition_dim, rank, bias=False)
        nn.init.normal_(self.right, std=0.02)
        nn.init.normal_(self.left, std=0.02)
        # Start exactly at the shared actor while retaining gradients for T.
        nn.init.zeros_(self.cognition_to_gate.weight)

    def _prepare_cognition(
        self,
        cognition: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        if cognition.ndim == 1:
            cognition = cognition.unsqueeze(0)
        if cognition.ndim != 2 or cognition.shape[-1] != self.cognition_dim:
            raise ValueError("cognition must have shape (batch, cognition_dim)")
        if cognition.shape[0] == 1 and batch_size != 1:
            cognition = cognition.expand(batch_size, -1)
        if cognition.shape[0] != batch_size:
            raise ValueError("cognition batch size must match the state batch")
        return cognition.detach()

    def gates(
        self,
        cognition: torch.Tensor,
        batch_size: int | None = None,
    ) -> torch.Tensor:
        """Return detached-cognition rank gates."""
        if batch_size is None:
            batch_size = 1 if cognition.ndim == 1 else cognition.shape[0]
        cognition = self._prepare_cognition(cognition, batch_size)
        return self.cognition_to_gate(cognition)

    def adaptation_matrix(self, cognition: torch.Tensor) -> torch.Tensor:
        """Materialize ``A diag(Tc) B`` for diagnostics.

        A vector input returns ``(out_features, in_features)``; a batched input
        returns ``(batch, out_features, in_features)``.
        """
        vector_input = cognition.ndim == 1
        gates = self.gates(cognition)
        matrices = torch.einsum(
            "or,br,ri->boi", self.left, gates, self.right
        )
        matrices = matrices * self.modulation_scale
        return matrices[0] if vector_input else matrices

    def forward(
        self,
        inputs: torch.Tensor,
        cognition: torch.Tensor,
    ) -> torch.Tensor:
        if inputs.ndim != 2 or inputs.shape[-1] != self.in_features:
            raise ValueError("inputs must have shape (batch, in_features)")
        gates = self.gates(cognition, inputs.shape[0])
        low_rank_features = F.linear(inputs, self.right) * gates
        modulation = F.linear(low_rank_features, self.left)
        return self.base(inputs) + self.modulation_scale * modulation


class CognitionModulatedActor(nn.Module):
    """Deterministic actor with cognition-required low-rank layer modulation."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        cognition_dim: int,
        hidden_dims: Sequence[int] = (256, 256),
        rank: int = 4,
        action_limit: float = 1.0,
    ) -> None:
        super().__init__()
        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one layer")
        dimensions = [state_dim, *hidden_dims, action_dim]
        self.layers = nn.ModuleList(
            LowRankCognitiveLinear(
                dimensions[index],
                dimensions[index + 1],
                cognition_dim,
                rank,
            )
            for index in range(len(dimensions) - 1)
        )
        self.action_limit = action_limit

    def base_forward(self, states: torch.Tensor) -> torch.Tensor:
        """Evaluate the shared, cognition-independent actor."""
        hidden = states
        for layer in self.layers[:-1]:
            hidden = F.relu(layer.base(hidden))
        return torch.tanh(self.layers[-1].base(hidden)) * self.action_limit

    def forward(
        self,
        states: torch.Tensor,
        cognition: torch.Tensor,
    ) -> torch.Tensor:
        # Detaching once here makes the actor/recognition optimization boundary
        # explicit; individual layers repeat the guard for standalone use.
        cognition = cognition.detach()
        hidden = states
        for layer in self.layers[:-1]:
            hidden = F.relu(layer(hidden, cognition))
        return torch.tanh(self.layers[-1](hidden, cognition)) * self.action_limit
