"""Low-rank physical residual for identifiable cognitive decoding."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from kanrf._protokan import ProtoKAN

from .split_cognitive_v2 import SplitCognitivePredictorV2


class LowRankSplitCognitivePredictor(SplitCognitivePredictorV2):
    """Decode history-derived physics coefficients through a state/action basis.

    The rank is an over-complete capacity choice, not a claim about the number
    of physical parameters.  Coefficients depend only on the history branch;
    the basis can depend on the current state and action.  Column normalization
    makes the coefficient scale comparable across states, while the returned
    Gram error can be used as an orthogonality regularizer during training.
    """

    def __init__(self, state_dim: int, action_dim: int, history_steps: int,
                 physics_rank: int = 4, residual_scale: float = 0.1,
                 token_count: int = 8, token_dim: int = 8,
                 state_context_dim: int = 16, hidden_dim: int = 24,
                 n_prototypes: int = 8):
        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            history_steps=history_steps,
            token_count=token_count,
            token_dim=token_dim,
            state_context_dim=state_context_dim,
            hidden_dim=hidden_dim,
            n_prototypes=n_prototypes,
        )
        if physics_rank > state_dim:
            raise ValueError("physics_rank must not exceed state_dim")
        self.physics_rank = physics_rank
        self.residual_scale = residual_scale
        self.physics_coefficients = ProtoKAN(
            [token_count * token_dim, hidden_dim, physics_rank],
            n_prototypes=n_prototypes,
        )
        self.physics_basis = ProtoKAN(
            [state_dim + action_dim, hidden_dim, state_dim * physics_rank],
            n_prototypes=n_prototypes,
        )
        # The unconstrained full residual belongs to the parent architecture;
        # remove it so this variant has only the low-rank transport path.
        del self.physics_dynamics

    def forward(self, history: torch.Tensor, state: torch.Tensor,
                action: torch.Tensor):
        tokens, gates = self.physics_encoder(self._physics_input(history))
        weighted = (tokens * gates.unsqueeze(-1)).flatten(start_dim=1)
        context = torch.tanh(self.state_context(state))
        base = self.base_dynamics(torch.cat([state, action, context], dim=-1))

        coefficients = torch.tanh(self.physics_coefficients(weighted))
        basis = self.physics_basis(torch.cat([state, action], dim=-1))
        basis = basis.view(state.shape[0], self.state_dim, self.physics_rank)
        basis = F.normalize(basis, dim=1, eps=1e-6)
        transported = torch.bmm(basis, coefficients.unsqueeze(-1)).squeeze(-1)
        residual = self.residual_scale * torch.tanh(transported)
        identity = torch.eye(
            self.physics_rank, dtype=basis.dtype, device=basis.device
        ).unsqueeze(0)
        gram_error = torch.bmm(basis.transpose(1, 2), basis) - identity
        return {
            "physics_tokens": tokens,
            "physics_gates": gates,
            "physics_pooled": self.physics_encoder.pool(tokens, gates),
            "physics_coefficients": coefficients,
            "physics_basis": basis,
            "basis_gram_error": gram_error,
            "state_context": context,
            "base_next_state": base,
            "physics_residual": residual,
            "next_state": base + residual,
        }
