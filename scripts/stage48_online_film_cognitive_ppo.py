"""Stage 48: online cognitive PPO using the FiLM-gated interface."""

from __future__ import annotations

import torch
from torch import nn

from scripts import stage46_online_cognitive_ppo as base
from scripts.stage41_ppo_cognitive_actor import GaussianActorBase, _mlp

STATE_DIM = 6
ACTION_DIM = 1


class CognitiveFiLMActor(GaussianActorBase):
    def __init__(self, cognitive: nn.Module, hidden_dim: int = 64):
        super().__init__(hidden_dim)
        self.cognitive = cognitive
        self.query_net = _mlp(STATE_DIM * 2, hidden_dim, ACTION_DIM)
        self.modulation = nn.Linear(ACTION_DIM, STATE_DIM * 2)
        self.decision_head = _mlp(STATE_DIM * 3, hidden_dim, ACTION_DIM)
        for parameter in self.cognitive.parameters():
            parameter.requires_grad = False

    def mean_action(self, state: torch.Tensor, goal: torch.Tensor):
        query = torch.tanh(self.query_net(torch.cat([state, goal], dim=-1)))
        predicted_next = self.cognitive(state, query)
        scale, bias = self.modulation(query).chunk(2, dim=-1)
        scale = 0.5 * torch.tanh(scale)
        bias = 0.5 * torch.tanh(bias)
        modulated = predicted_next * (1.0 + scale) + bias
        delta = modulated - state
        return self.decision_head(torch.cat([modulated, delta, goal], dim=-1))


base.CognitiveEmbeddedGaussianActor = CognitiveFiLMActor
base.main()
