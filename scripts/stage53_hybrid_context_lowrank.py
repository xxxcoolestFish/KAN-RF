"""Stage 53: history context plus a small low-rank dynamics adapter.

The pretrained ProtoKAN trunk is kept fixed during parameter-shift
adaptation.  Only the recurrent context encoder and a zero-initialized
low-rank residual adapter are updated from real transitions.  This tests
whether a small amount of controlled plasticity is a better compromise than
either full cognitive updates or context-only updates.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from scripts import stage51_context_cognitive_ppo as base


ADAPTER_RANK = 4
ADAPTER_SCALE = 0.25


class HybridCognitiveKAN(base.ContextCognitiveKAN):
    """ContextCognitiveKAN with a zero-start low-rank output residual."""

    def __init__(self, context_dim: int = base.CONTEXT_DIM,
                 hidden_dim: int = 64, rank: int = ADAPTER_RANK):
        super().__init__(context_dim=context_dim, hidden_dim=hidden_dim)
        input_dim = base.STATE_DIM + base.ACTION_DIM + context_dim
        self.adapter_down = nn.Linear(input_dim, rank, bias=False)
        self.adapter_up = nn.Linear(rank, base.STATE_DIM, bias=False)
        nn.init.normal_(self.adapter_down.weight, mean=0.0, std=0.02)
        # A zero residual means the source policy starts exactly as the
        # pretrained ProtoKAN policy, while gradients can grow the adapter.
        nn.init.zeros_(self.adapter_up.weight)

    def forward(self, state, action, context):
        features = torch.cat([state, action, context], dim=-1)
        base_prediction = self.network(features)
        residual = self.adapter_up(torch.tanh(self.adapter_down(features)))
        return base_prediction + ADAPTER_SCALE * residual


def update_hybrid(cognitive, optimizer, transitions, epochs, seed):
    states, actions, next_states, dones = transitions
    # Keep the learned physical trunk fixed; adapt only context and the
    # explicitly capacity-limited residual pathway.
    for parameter in cognitive.parameters():
        parameter.requires_grad = False
    trainable = list(cognitive.context_encoder.parameters())
    trainable += list(cognitive.adapter_down.parameters())
    trainable += list(cognitive.adapter_up.parameters())
    for parameter in trainable:
        parameter.requires_grad = True

    losses = []
    for _ in range(epochs):
        context = torch.zeros(states.shape[1], base.CONTEXT_DIM)
        loss = torch.zeros(())
        for t in range(states.shape[0]):
            prediction = cognitive(states[t], actions[t], context)
            loss = loss + F.smooth_l1_loss(prediction, next_states[t])
            next_context = cognitive.update_context(
                context, states[t], actions[t], next_states[t],
            )
            context = torch.where(dones[t].unsqueeze(-1),
                                  torch.zeros_like(context), next_context)
        loss = loss / states.shape[0]
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 5.0)
        optimizer.step()
        losses.append(float(loss.detach()))

    for parameter in cognitive.parameters():
        parameter.requires_grad = False
    return sum(losses) / max(1, len(losses))


# The base training loop constructs the class through this module global.
base.ContextCognitiveKAN = HybridCognitiveKAN
base.update_cognitive = update_hybrid
base.main()
