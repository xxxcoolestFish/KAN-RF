"""Streaming cognitive world model based on atomic transition records."""

from __future__ import annotations

import torch
from torch import nn

from kanrf._protokan import ProtoKAN


class StreamingCognitiveWorldModel(nn.Module):
    """Predict dynamics while maintaining an online physical context state.

    ``observe_transition`` consumes one observed ``(s, a, s_next)`` record and
    updates ``z``.  ``predict_next`` uses the context accumulated before the
    current action, so the target next state is never fed into its own
    prediction.  The latent width is a capacity choice, not a physical
    parameter count or a semantic slot assignment.
    """

    def __init__(self, state_dim: int = 6, action_dim: int = 1,
                 latent_dim: int = 16, hidden_dim: int = 32,
                 n_prototypes: int = 8):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.transition_encoder = ProtoKAN(
            [2 * state_dim + action_dim, hidden_dim, latent_dim],
            n_prototypes=n_prototypes,
        )
        self.latent_update = nn.GRUCell(latent_dim, latent_dim)
        self.dynamics = ProtoKAN(
            [state_dim + action_dim + latent_dim, hidden_dim, state_dim],
            n_prototypes=n_prototypes,
        )

    def initial_latent(self, batch_size: int, device=None):
        return torch.zeros(batch_size, self.latent_dim, device=device)

    def predict_next(self, state, action, latent):
        return self.dynamics(torch.cat([state, action, latent], dim=-1))

    def observe_transition(self, state, action, next_state, latent):
        evidence = torch.tanh(
            self.transition_encoder(torch.cat([state, action, next_state], dim=-1))
        )
        return self.latent_update(evidence, latent)

    def forward_sequence(self, states, actions, next_states=None):
        """Unroll predictions and optional online updates over a sequence."""
        latent = self.initial_latent(states.shape[0], states.device)
        predictions, pre_latents, post_latents = [], [], []
        for index in range(states.shape[1]):
            pre_latents.append(latent)
            prediction = self.predict_next(states[:, index], actions[:, index], latent)
            predictions.append(prediction)
            if next_states is not None:
                latent = self.observe_transition(
                    states[:, index], actions[:, index],
                    next_states[:, index], latent,
                )
            post_latents.append(latent)
        return {
            "predictions": torch.stack(predictions, dim=1),
            "pre_latents": torch.stack(pre_latents, dim=1),
            "post_latents": torch.stack(post_latents, dim=1),
            "latent": latent,
        }

    def physics_code(self, latent):
        """Bounded code exposed to the decision receiver."""
        return torch.tanh(latent)
