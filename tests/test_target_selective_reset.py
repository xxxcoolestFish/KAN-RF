"""Checks for controlled source-policy parameter resets."""

import torch

from cpbn.corridor_policy import DirectCorridorActor
from scripts.validate_target_selective_reset import (
    RESET_BLOCKS,
    selective_reset_actor,
)


def _changed_names(before, after):
    source = dict(before.named_parameters())
    reset = dict(after.named_parameters())
    return {name for name in source if not torch.equal(source[name], reset[name])}


def test_head_reset_changes_only_action_head():
    torch.manual_seed(11)
    source = DirectCorridorActor(hidden_dim=8, log_std=-0.3)
    reset, names = selective_reset_actor(source, "head", seed=29)
    assert set(names) == set(RESET_BLOCKS["head"])
    assert _changed_names(source, reset) == set(RESET_BLOCKS["head"])
    assert torch.equal(source.log_std, reset.log_std)


def test_recurrent_reset_changes_only_hidden_to_hidden_gru_parameters():
    torch.manual_seed(13)
    source = DirectCorridorActor(hidden_dim=8, log_std=0.2)
    reset, names = selective_reset_actor(source, "recurrent", seed=31)
    assert set(names) == set(RESET_BLOCKS["recurrent"])
    assert _changed_names(source, reset) == set(RESET_BLOCKS["recurrent"])
    assert torch.equal(
        source.encoder.gru.weight_ih_l0,
        reset.encoder.gru.weight_ih_l0,
    )
    assert torch.equal(source.log_std, reset.log_std)


def test_combined_reset_is_exact_union_of_both_blocks():
    torch.manual_seed(17)
    source = DirectCorridorActor(hidden_dim=8)
    reset, names = selective_reset_actor(
        source, "recurrent_and_head", seed=37,
    )
    expected = set(RESET_BLOCKS["head"]) | set(RESET_BLOCKS["recurrent"])
    assert set(names) == expected
    assert _changed_names(source, reset) == expected
