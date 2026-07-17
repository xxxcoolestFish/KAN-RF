"""Shared physical dictionary used by cognitive and decision branches."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from kanrf._protokan import ProtoKAN

from .low_rank_split import LowRankSplitCognitivePredictor


class SharedPhysicsDictionary(nn.Module):
    """A normalized coefficient-to-latent physical dictionary."""

    def __init__(self, coefficient_dim: int, latent_dim: int):
        super().__init__()
        self.coefficient_dim = coefficient_dim
        self.latent_dim = latent_dim
        self.dictionary = nn.Parameter(torch.empty(coefficient_dim, latent_dim))
        nn.init.orthogonal_(self.dictionary)

    def normalized(self) -> torch.Tensor:
        return F.normalize(self.dictionary, dim=0)

    def forward(self, coefficients: torch.Tensor) -> torch.Tensor:
        return coefficients @ self.normalized()

    def orthogonality_loss(self) -> torch.Tensor:
        matrix = self.normalized()
        gram = matrix.T @ matrix
        eye = torch.eye(self.latent_dim, dtype=matrix.dtype, device=matrix.device)
        return (gram - eye).square().mean()


class SharedDictionaryCognitivePredictor(nn.Module):
    """Cognitive predictor whose residual decoder consumes the shared latent."""

    def __init__(self, shared_dictionary: SharedPhysicsDictionary,
                 state_dim: int, action_dim: int, history_steps: int,
                 latent_dim: int = 4, residual_scale: float = 0.1,
                 token_count: int = 8, token_dim: int = 8,
                 hidden_dim: int = 24, n_prototypes: int = 8):
        super().__init__()
        self.shared_dictionary = shared_dictionary
        self.residual_scale = residual_scale
        self.core = LowRankSplitCognitivePredictor(
            physics_rank=latent_dim, residual_scale=residual_scale,
            state_dim=state_dim, action_dim=action_dim,
            history_steps=history_steps, token_count=token_count,
            token_dim=token_dim, hidden_dim=hidden_dim,
            n_prototypes=n_prototypes,
        )
        self.shared_residual_decoder = ProtoKAN(
            [state_dim + action_dim + latent_dim, hidden_dim, state_dim],
            n_prototypes=n_prototypes,
        )

    def forward(self, history: torch.Tensor, state: torch.Tensor,
                action: torch.Tensor):
        output = self.core(history, state, action)
        coefficients = output["physics_coefficients"]
        latent = self.shared_dictionary(coefficients)
        raw_residual = self.shared_residual_decoder(
            torch.cat([state, action, latent], dim=-1)
        )
        residual = self.residual_scale * torch.tanh(raw_residual)
        output["physics_latent"] = latent
        output["physics_residual"] = residual
        output["next_state"] = output["base_next_state"] + residual
        output["dictionary_orthogonality"] = self.shared_dictionary.orthogonality_loss()
        return output
