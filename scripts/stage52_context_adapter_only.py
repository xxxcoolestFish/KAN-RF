"""Stage 52: freeze ProtoKAN and adapt only the history context encoder."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from scripts import stage51_context_cognitive_ppo as base


def update_context_only(cognitive, optimizer, transitions, epochs, seed):
    states, actions, next_states, dones = transitions
    for parameter in cognitive.parameters():
        parameter.requires_grad = False
    for parameter in cognitive.context_encoder.parameters():
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
            context = torch.where(dones[t].unsqueeze(-1), torch.zeros_like(context), next_context)
        loss = loss / states.shape[0]
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(cognitive.context_encoder.parameters(), 5.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    for parameter in cognitive.parameters():
        parameter.requires_grad = False
    return sum(losses) / max(1, len(losses))


base.update_cognitive = update_context_only
base.main()
