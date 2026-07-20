"""Stage 58: multi-action oracle for the DeepONet policy.

The policy receives exact target transition outcomes for a small internal
action basis.  These are virtual model evaluations, not executed exploratory
actions.  The purpose is to test whether comparing multiple action effects
solves the single-query bottleneck.
"""

from __future__ import annotations

import torch
from torch import nn

from scripts import stage56_oracle_ppo as oracle


base = oracle.base
OracleCognitive = oracle.OracleCognitive
PRETRAIN_FACTOR = oracle.PRETRAIN_FACTOR

ACTION_BASIS_COUNT = 9
TOKEN_DIM = 32


class OracleCognitiveDefault(OracleCognitive):
    def __init__(self, factor=None):
        super().__init__(PRETRAIN_FACTOR[0] if factor is None else factor)


class MultiActionOracleActor(base.GaussianActorBase):
    """Direct policy over exact virtual action consequences."""

    def __init__(self, cognitive: OracleCognitiveDefault,
                 hidden_dim: int = 64):
        super().__init__(hidden_dim)
        self.cognitive = cognitive
        self.register_buffer(
            "action_basis", torch.linspace(-1.0, 1.0, ACTION_BASIS_COUNT),
        )
        self.effect_encoder = base._mlp(
            base.ACTION_DIM + base.STATE_DIM * 2,
            hidden_dim,
            TOKEN_DIM,
        )
        self.trunk = base._mlp(
            base.ACTOR_STATE_DIM + base.STATE_DIM + base.ACTION_DIM,
            hidden_dim,
            TOKEN_DIM,
        )
        self.attention = nn.MultiheadAttention(
            TOKEN_DIM, num_heads=4, batch_first=True,
        )
        self.gate = nn.Linear(TOKEN_DIM, TOKEN_DIM)
        self.score_head = base._mlp(TOKEN_DIM, hidden_dim, 1)
        for parameter in self.cognitive.parameters():
            parameter.requires_grad = False

    def action_scores(self, actor_state, goal):
        batch = actor_state.shape[0]
        state = actor_state[:, :base.STATE_DIM]
        action = self.action_basis.to(actor_state).view(1, -1, 1).expand(batch, -1, -1)
        state_repeated = state.unsqueeze(1).expand(-1, ACTION_BASIS_COUNT, -1)
        flat_state = state_repeated.reshape(-1, base.STATE_DIM)
        flat_action = action.reshape(-1, base.ACTION_DIM)
        # Exact target dynamics; no hidden factor is passed to the actor.
        flat_next = self.cognitive(flat_state, flat_action)
        next_state = flat_next.reshape(batch, ACTION_BASIS_COUNT, base.STATE_DIM)
        delta = next_state - state_repeated
        effects = self.effect_encoder(torch.cat([action, next_state, delta], dim=-1))

        context_goal = torch.cat([actor_state, goal], dim=-1)
        trunk_input = torch.cat([
            context_goal.unsqueeze(1).expand(-1, ACTION_BASIS_COUNT, -1),
            action,
        ], dim=-1)
        trunk = self.trunk(trunk_input)
        attended, _ = self.attention(trunk, effects, effects)
        fused = trunk * (1.0 + torch.tanh(self.gate(attended)))
        return self.score_head(fused).squeeze(-1)

    def mean_action(self, actor_state, goal):
        scores = self.action_scores(actor_state, goal)
        probabilities = torch.softmax(scores, dim=-1)
        expected = (probabilities * self.action_basis.to(actor_state)).sum(dim=-1, keepdim=True)
        return torch.atanh(expected.clamp(-0.999, 0.999))


base.ContextCognitiveKAN = OracleCognitiveDefault
base.ContextFiLMActor = MultiActionOracleActor

# stage56 already installed the no-pretrain/no-update hooks and target-factor
# switch in base.evaluate.  Reuse its parser and output protocol.
oracle.main()
