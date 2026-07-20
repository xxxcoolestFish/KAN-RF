"""Stage 54: decision network driven by a ProtoKAN formula graph.

The cognition network remains a pure next-state predictor.  The actor does
not propose an action query for the cognition network.  Instead, a
``ProtoKANFormulaOperator`` evaluates the complete learned edge-function
graph on an analytic basis and exposes layer tokens to an attention decoder.
The decoder directly emits the action.
"""

from __future__ import annotations

import torch
from torch import nn

from kanrf.formula_graph_operator import ProtoKANFormulaOperator
from scripts import stage51_context_cognitive_ppo as base


FORMULA_TOKEN_DIM = 32


class FormulaGraphActor(base.GaussianActorBase):
    def __init__(self, cognitive: base.ContextCognitiveKAN,
                 hidden_dim: int = 64):
        super().__init__(hidden_dim)
        self.cognitive = cognitive
        self.formula = ProtoKANFormulaOperator(
            cognitive.network,
            basis_count=8,
            token_dim=FORMULA_TOKEN_DIM,
        )
        self.query = base._mlp(
            base.ACTOR_STATE_DIM + base.STATE_DIM,
            hidden_dim,
            FORMULA_TOKEN_DIM,
        )
        self.attention = nn.MultiheadAttention(
            FORMULA_TOKEN_DIM, num_heads=4, batch_first=True,
        )
        self.state_goal = base._mlp(
            base.ACTOR_STATE_DIM + base.STATE_DIM,
            hidden_dim,
            hidden_dim,
        )
        self.gate = nn.Linear(FORMULA_TOKEN_DIM, hidden_dim)
        self.decision_head = base._mlp(
            hidden_dim + FORMULA_TOKEN_DIM,
            hidden_dim,
            base.ACTION_DIM,
        )
        for parameter in self.cognitive.parameters():
            parameter.requires_grad = False

    def mean_action(self, actor_state, goal):
        state = actor_state[:, :base.STATE_DIM]
        context = actor_state[:, base.STATE_DIM:]
        # Zero is only a coordinate anchor for the current-state graph token;
        # it is not an action emitted or executed by the policy.
        graph_input = torch.cat([
            state, torch.zeros(state.shape[0], base.ACTION_DIM), context,
        ], dim=-1)
        _, formula_tokens = self.formula(graph_input, return_tokens=True)
        query = self.query(torch.cat([actor_state, goal], dim=-1)).unsqueeze(1)
        summary, _ = self.attention(query, formula_tokens, formula_tokens)
        summary = summary.squeeze(1)

        state_goal = self.state_goal(torch.cat([actor_state, goal], dim=-1))
        # Every action path is multiplicatively modulated by the formula
        # token; there is no actor head that bypasses the formula graph.
        fused = state_goal * (1.0 + torch.tanh(self.gate(summary)))
        return self.decision_head(torch.cat([fused, summary], dim=-1))


base.ContextFiLMActor = FormulaGraphActor
base.main()
