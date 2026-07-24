import pytest
import torch

from cpbn.cognitive_ablation import intervene_on_coordinates


def test_inferred_and_frozen_preserve_coordinates():
    coordinates = torch.tensor([1.0, 2.0, 3.0])
    torch.testing.assert_close(
        intervene_on_coordinates(coordinates, "inferred"),
        coordinates,
    )
    torch.testing.assert_close(
        intervene_on_coordinates(coordinates, "frozen"),
        coordinates,
    )


def test_zero_removes_all_cognitive_information():
    coordinates = torch.tensor([1.0, -2.0, 3.0])
    torch.testing.assert_close(
        intervene_on_coordinates(coordinates, "zero"),
        torch.zeros_like(coordinates),
    )


def test_shuffle_is_deterministic_and_nonidentity():
    coordinates = torch.tensor([1.0, 2.0, 3.0])
    shuffled = intervene_on_coordinates(coordinates, "shuffled")
    torch.testing.assert_close(shuffled, torch.tensor([3.0, 1.0, 2.0]))
    assert not torch.equal(shuffled, coordinates)


def test_unknown_condition_is_rejected():
    with pytest.raises(ValueError):
        intervene_on_coordinates(torch.ones(3), "unknown")
