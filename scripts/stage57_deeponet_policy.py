"""Stage 57: DeepONet-style physics-function-to-policy operator.

The branch network reads the canonical formula graph of the cognitive
ProtoKAN.  The trunk network evaluates state/goal/action-basis features.  A
cross-attention operator produces a score for every action basis point, and
the actor emits the expected action directly.  The cognitive model never
receives a policy-generated action query.
"""

from __future__ import annotations

import torch
from torch import nn

from kanrf.canonical_formula_operator import CanonicalFormulaOperator
from scripts import stage51_context_cognitive_ppo as base


TOKEN_DIM = 32
ACTION_BASIS_COUNT = 17
_ACTIVE_ACTORS = []


class DeepONetPolicyActor(base.GaussianActorBase):
    def __init__(self, cognitive: base.ContextCognitiveKAN,
                 hidden_dim: int = 64):
        super().__init__(hidden_dim)
        self.cognitive = cognitive
        self.formula = CanonicalFormulaOperator(
            cognitive.network, basis_count=8, token_dim=TOKEN_DIM,
        )
        self.register_buffer(
            "action_basis",
            torch.linspace(-1.0, 1.0, ACTION_BASIS_COUNT),
        )
        self.trunk = base._mlp(
            base.ACTOR_STATE_DIM + base.STATE_DIM + 1,
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
        _ACTIVE_ACTORS.append(self)

    def _branch_tokens(self, actor_state):
        state = actor_state[:, :base.STATE_DIM]
        context = actor_state[:, base.STATE_DIM:]
        # Zero action is only a coordinate anchor for dynamic formula tokens;
        # it is not executed and is not selected as the policy action.
        graph_input = torch.cat([
            state, torch.zeros(state.shape[0], base.ACTION_DIM), context,
        ], dim=-1)
        _, tokens = self.formula(graph_input, return_tokens=True)
        return tokens

    def action_scores(self, actor_state, goal):
        batch = actor_state.shape[0]
        tokens = self._branch_tokens(actor_state)
        action = self.action_basis.to(actor_state).view(1, -1, 1).expand(batch, -1, -1)
        context = torch.cat([actor_state, goal], dim=-1)
        trunk_input = torch.cat([
            context.unsqueeze(1).expand(-1, ACTION_BASIS_COUNT, -1), action,
        ], dim=-1)
        trunk = self.trunk(trunk_input)
        attended, _ = self.attention(trunk, tokens, tokens)
        fused = trunk * (1.0 + torch.tanh(self.gate(attended)))
        return self.score_head(fused).squeeze(-1)

    def mean_action(self, actor_state, goal):
        scores = self.action_scores(actor_state, goal)
        probabilities = torch.softmax(scores, dim=-1)
        expected = (probabilities * self.action_basis.to(actor_state)).sum(dim=-1, keepdim=True)
        return torch.atanh(expected.clamp(-0.999, 0.999))


_base_update_cognitive = base.update_cognitive


def update_cognitive_and_refresh(cognitive, optimizer, transitions, epochs, seed):
    result = _base_update_cognitive(
        cognitive, optimizer, transitions, epochs, seed,
    )
    for actor in _ACTIVE_ACTORS:
        if actor.cognitive is cognitive:
            actor.formula.clear_cache()
    return result


base.ContextFiLMActor = DeepONetPolicyActor
base.update_cognitive = update_cognitive_and_refresh
base.main()
