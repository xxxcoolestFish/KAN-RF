"""Strict delta-interface ablation without a direct query-to-action shortcut."""

from __future__ import annotations

import torch
from torch import nn

from physics_transfer.variants import step
from scripts import stage44_ppo_embedded_cognitive as base
from scripts.stage41_ppo_cognitive_actor import GaussianActorBase, _mlp, tip_height

STATE_DIM = 6
ACTION_DIM = 1


class CognitiveDeltaNoQueryActor(GaussianActorBase):
    def __init__(self, cognitive: nn.Module, hidden_dim: int = 64):
        super().__init__(hidden_dim)
        self.cognitive = cognitive
        self.query_net = _mlp(STATE_DIM * 2, hidden_dim, ACTION_DIM)
        self.decision_head = _mlp(STATE_DIM * 3, hidden_dim, ACTION_DIM)
        for parameter in self.cognitive.parameters():
            parameter.requires_grad = False

    def mean_action(self, state: torch.Tensor, goal: torch.Tensor):
        query = torch.tanh(self.query_net(torch.cat([state, goal], dim=-1)))
        predicted_next = self.cognitive(state, query)
        delta = predicted_next - state
        return self.decision_head(torch.cat([predicted_next, delta, goal], dim=-1))


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


base.CognitiveEmbeddedGaussianActor = CognitiveDeltaNoQueryActor
base._random_states = fixed_states
base.evaluate = evaluate_fixed
base.main()
