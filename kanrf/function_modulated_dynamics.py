"""Function-aligned ProtoKAN dynamics with low-dimensional environment modes."""

from __future__ import annotations

import torch
import torch.nn as nn

from kanrf._protokan import ProtoKAN


class ActionModulatedProtoKAN(nn.Module):
    """Let an environment latent modulate only first-layer action functions.

    All environments share the same ProtoKAN and prototype coordinates.  The
    latent combines aligned value/derivative functions on action edges, rather
    than interpolating independently trained parameter vectors.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        action_start: int,
        action_dim: int,
        latent_dim: int = 1,
        n_prototypes: int = 8,
        grid_range: float = 4.0,
    ) -> None:
        super().__init__()
        if action_start < 0 or action_start + action_dim > input_dim:
            raise ValueError("action slice must lie inside the input")
        self.action_start = action_start
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.backbone = ProtoKAN(
            [input_dim, hidden_dim, output_dim],
            n_prototypes=n_prototypes,
            grid_range=grid_range,
        )
        for layer in self.backbone.layers:
            layer.proto_pos.requires_grad_(False)
            layer.log_sigma.requires_grad_(False)

        shape = (latent_dim, hidden_dim, action_dim, n_prototypes)
        self.action_mode_value = nn.Parameter(torch.empty(shape))
        self.action_mode_derivative = nn.Parameter(torch.empty(shape))
        nn.init.normal_(self.action_mode_value, std=0.01)
        nn.init.normal_(self.action_mode_derivative, std=0.01)

    def action_mode_outputs(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return aligned action-function modes with shape ``(B, R, H)``."""
        first = self.backbone.layers[0]
        actions = inputs[
            :, self.action_start:self.action_start + self.action_dim
        ]
        sigma = torch.exp(first.log_sigma).clamp(1e-4, 10.0)
        differences = actions.unsqueeze(-1) - first.proto_pos.view(1, 1, -1)
        weights = torch.softmax(
            -differences.square() / (2.0 * sigma.square()),
            dim=-1,
        )
        local_predictions = (
            self.action_mode_value.unsqueeze(0)
            + self.action_mode_derivative.unsqueeze(0)
            * differences[:, None, None, :, :]
        )
        edge_outputs = (
            weights[:, None, None, :, :] * local_predictions
        ).sum(dim=-1)
        return edge_outputs.sum(dim=-1)

    def forward(
        self,
        inputs: torch.Tensor,
        environment_latent: torch.Tensor,
    ) -> torch.Tensor:
        if environment_latent.ndim == 1:
            environment_latent = environment_latent.unsqueeze(-1)
        if environment_latent.shape[-1] != self.latent_dim:
            raise ValueError("environment latent has the wrong dimension")
        if environment_latent.shape[0] == 1 and inputs.shape[0] != 1:
            environment_latent = environment_latent.expand(
                inputs.shape[0], -1
            )
        if environment_latent.shape[0] != inputs.shape[0]:
            raise ValueError("latent batch size must match input batch size")

        hidden = self.backbone.layers[0](inputs)
        modes = self.action_mode_outputs(inputs)
        hidden = hidden + torch.einsum(
            "br,brh->bh", environment_latent, modes
        )
        return self.backbone.layers[1](hidden)

