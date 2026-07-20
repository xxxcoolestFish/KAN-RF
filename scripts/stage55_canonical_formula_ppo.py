"""Stage 55: anchor-relative canonical formula-graph decision operator."""

from __future__ import annotations

import torch
from torch import nn

from kanrf.canonical_formula_operator import CanonicalFormulaOperator
from scripts import stage51_context_cognitive_ppo as base


TOKEN_DIM = 32
_ACTIVE_ACTORS = []


class CanonicalFormulaActor(base.GaussianActorBase):
    def __init__(self, cognitive: base.ContextCognitiveKAN,
                 hidden_dim: int = 64):
        super().__init__(hidden_dim)
        self.cognitive = cognitive
        self.formula = CanonicalFormulaOperator(
            cognitive.network, basis_count=8, token_dim=TOKEN_DIM,
        )
        self.query = base._mlp(
            base.ACTOR_STATE_DIM + base.STATE_DIM, hidden_dim, TOKEN_DIM,
        )
        self.attention = nn.MultiheadAttention(
            TOKEN_DIM, num_heads=4, batch_first=True,
        )
        self.state_goal = base._mlp(
            base.ACTOR_STATE_DIM + base.STATE_DIM, hidden_dim, hidden_dim,
        )
        self.gate = nn.Linear(TOKEN_DIM, hidden_dim)
        self.decision_head = base._mlp(
            hidden_dim + TOKEN_DIM, hidden_dim, base.ACTION_DIM,
        )
        for parameter in self.cognitive.parameters():
            parameter.requires_grad = False
        _ACTIVE_ACTORS.append(self)

    def mean_action(self, actor_state, goal):
        state = actor_state[:, :base.STATE_DIM]
        context = actor_state[:, base.STATE_DIM:]
        # The formula interface does not use a policy-generated action probe.
        # Zero action is only a current-state anchor for the dynamic token.
        graph_input = torch.cat([
            state, torch.zeros(state.shape[0], base.ACTION_DIM), context,
        ], dim=-1)
        _, formula_tokens = self.formula(graph_input, return_tokens=True)
        query = self.query(torch.cat([actor_state, goal], dim=-1)).unsqueeze(1)
        summary, _ = self.attention(query, formula_tokens, formula_tokens)
        summary = summary.squeeze(1)
        state_goal = self.state_goal(torch.cat([actor_state, goal], dim=-1))
        fused = state_goal * (1.0 + torch.tanh(self.gate(summary)))
        return self.decision_head(torch.cat([fused, summary], dim=-1))


_base_update_cognitive = base.update_cognitive


def update_cognitive_and_refresh(cognitive, optimizer, transitions, epochs, seed):
    result = _base_update_cognitive(
        cognitive, optimizer, transitions, epochs, seed,
    )
    for actor in _ACTIVE_ACTORS:
        if actor.cognitive is cognitive:
            actor.formula.clear_cache()
    return result


base.ContextFiLMActor = CanonicalFormulaActor
base.update_cognitive = update_cognitive_and_refresh
base.main()
