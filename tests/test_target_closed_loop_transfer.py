from types import SimpleNamespace

import torch

from scripts.validate_target_closed_loop_bellman_transfer import (
    actor_from_config,
    actor_parameter_vector,
)


def test_actor_factory_reconstructs_frozen_source_architecture():
    config = SimpleNamespace(
        hidden_dim=8,
        corridor_horizon=4,
        action_bins=5,
        backup_depth=2,
        macro_steps=4,
        temperature=0.25,
        gamma=0.99,
        progress_reward=1.0,
        progress_clip=0.25,
        inside_reward=0.08,
        success_reward=3.0,
        action_penalty=0.002,
        corridor_radius=0.12,
    )
    actor = actor_from_config(config)
    assert actor.action_bins == 5
    assert actor.backup_depth == 2
    assert actor.macro_steps == 4
    assert actor.potential.corridor_horizon == 4


def test_parameter_vector_detects_any_actor_update():
    config = SimpleNamespace(
        hidden_dim=8,
        corridor_horizon=3,
        action_bins=3,
        backup_depth=2,
        macro_steps=2,
        temperature=0.25,
        gamma=0.99,
        progress_reward=1.0,
        progress_clip=0.25,
        inside_reward=0.08,
        success_reward=3.0,
        action_penalty=0.002,
        corridor_radius=0.12,
    )
    actor = actor_from_config(config)
    before = actor_parameter_vector(actor)
    with torch.no_grad():
        next(actor.parameters()).add_(0.1)
    after = actor_parameter_vector(actor)
    assert not torch.equal(before, after)
