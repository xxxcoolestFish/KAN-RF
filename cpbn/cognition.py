"""ProtoKAN transition cognition with manifold-preserving state output."""

from __future__ import annotations

import torch
from torch import nn

from cpbn.time_varying_tube import apply_tangent_error, tangent_error
from kanrf import ProtoKAN


class ProtoKANDynamics(nn.Module):
    """Predict an intrinsic state increment and reconstruct the next state.

    The network remains a single, unsplit ProtoKAN.  The wrapper only enforces
    the known representation geometry: angle pairs stay on the unit circle and
    normalized velocities remain inside their valid interval.
    """

    def __init__(
        self,
        hidden_dim: int = 32,
        n_prototypes: int = 8,
    ):
        super().__init__()
        self.network = ProtoKAN(
            [7, hidden_dim, 4],
            n_prototypes=n_prototypes,
        )
        # Bounds cover every observed one-step source transition without
        # forcing the tanh output to sit exactly on its saturation boundary.
        self.register_buffer(
            "delta_limit", torch.tensor([0.35, 0.45, 0.50, 0.50]),
        )

    def transition_delta(
        self, state: torch.Tensor, action: torch.Tensor,
    ) -> torch.Tensor:
        network_input = torch.cat([state, action], dim=-1)
        return torch.tanh(self.network(network_input)) * self.delta_limit

    def forward(
        self, state: torch.Tensor, action: torch.Tensor,
    ) -> torch.Tensor:
        return apply_tangent_error(state, self.transition_delta(state, action))

    def prediction_loss(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor,
    ) -> torch.Tensor:
        prediction = self(state, action)
        return tangent_error(prediction, next_state).square().mean()
