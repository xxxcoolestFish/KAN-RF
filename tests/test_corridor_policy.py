"""Checks for the direct state-corridor policy interface."""

import torch

from cpbn.corridor_policy import DirectCorridorActor, future_corridor


def test_future_corridor_clamps_at_terminal_state():
    reference = torch.randn(10, 6)
    phase = torch.tensor([0, 8, 9])
    corridor = future_corridor(reference, phase, horizon=4)
    assert corridor.shape == (3, 4, 6)
    assert torch.equal(corridor[-1, -1], reference[-1])


def test_actor_action_changes_with_corridor():
    torch.manual_seed(11)
    actor = DirectCorridorActor(hidden_dim=16)
    state = torch.randn(5, 6)
    first = torch.randn(5, 6, 6)
    second = first + 0.5
    action_a, _ = actor.sample(state, first, deterministic=True)
    action_b, _ = actor.sample(state, second, deterministic=True)
    assert float((action_a - action_b).abs().max()) > 1e-5
