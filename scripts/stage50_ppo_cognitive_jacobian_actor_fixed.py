"""FiLM actor with a control-relevant cognitive action-sensitivity feature.

The decision head never receives the internal query directly.  It receives the
ProtoKAN prediction, its state delta, and a finite-difference approximation of
the ProtoKAN Jacobian with respect to the query action.
"""

from __future__ import annotations

import torch
from torch import nn

from physics_transfer.variants import step
from scripts import stage44_ppo_embedded_cognitive as base
from scripts.stage41_ppo_cognitive_actor import GaussianActorBase, _mlp, tip_height

STATE_DIM = 6
ACTION_DIM = 1
QUERY_EPS = 0.05


class CognitiveJacobianFiLMActor(GaussianActorBase):
    def __init__(self, cognitive: nn.Module, hidden_dim: int = 64):
        super().__init__(hidden_dim)
        self.cognitive = cognitive
        self.query_net = _mlp(STATE_DIM * 2, hidden_dim, ACTION_DIM)
        self.modulation = nn.Linear(ACTION_DIM, STATE_DIM * 2)
        # prediction (6) + delta (6) + action sensitivity (6) + goal (6)
        self.decision_head = _mlp(STATE_DIM * 4, hidden_dim, ACTION_DIM)
        for parameter in self.cognitive.parameters():
            parameter.requires_grad = False

    def mean_action(self, state: torch.Tensor, goal: torch.Tensor):
        query = torch.tanh(self.query_net(torch.cat([state, goal], dim=-1)))
        predicted_next = self.cognitive(state, query)
        plus = self.cognitive(state, (query + QUERY_EPS).clamp(-1.0, 1.0))
        minus = self.cognitive(state, (query - QUERY_EPS).clamp(-1.0, 1.0))
        sensitivity = (plus - minus) / (2.0 * QUERY_EPS)
        scale, bias = self.modulation(query).chunk(2, dim=-1)
        scale = 0.5 * torch.tanh(scale)
        bias = 0.5 * torch.tanh(bias)
        modulated = predicted_next * (1.0 + scale) + bias
        delta = modulated - state
        features = torch.cat([modulated, delta, sensitivity, goal], dim=-1)
        return self.decision_head(features)


def fixed_states(count: int, generator=None, noise: float = 0.04):
    angles = torch.randn(count, 2, generator=generator) * noise
    return torch.stack([
        torch.cos(angles[:, 0]), torch.sin(angles[:, 0]),
        torch.cos(angles[:, 1]), torch.sin(angles[:, 1]),
        torch.randn(count, generator=generator) * noise,
        torch.randn(count, generator=generator) * noise,
    ], dim=-1)


@torch.no_grad()
def evaluate_fixed(actor, factor, goal, states, steps):
    state = states.clone()
    factor_tensor = torch.tensor(factor, dtype=state.dtype).view(1, 4).expand(state.shape[0], -1)
    maximum = torch.full((state.shape[0],), -float("inf"))
    success = torch.zeros(state.shape[0], dtype=torch.bool)
    for _ in range(steps):
        action, _, _ = actor.sample(state, goal.expand(state.shape[0], -1), deterministic=True)
        state = step(state, action, factor_tensor[:, 0], factor_tensor[:, 1],
                     factor_tensor[:, 2], factor_tensor[:, 3])
        height = tip_height(state)
        maximum = torch.maximum(maximum, height)
        success |= height >= 1.0
    return {
        "success_count": int(success.sum()),
        "success_rate": float(success.float().mean()),
        "mean_max_height": float(maximum.mean()),
        "max_height": float(maximum.max()),
    }


base.CognitiveEmbeddedGaussianActor = CognitiveJacobianFiLMActor
base._random_states = fixed_states
base.evaluate = evaluate_fixed
base.main()
