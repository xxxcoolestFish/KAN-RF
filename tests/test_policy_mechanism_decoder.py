import numpy as np
import torch

from cpbn.policy_mechanism_decoder import (
    ContextConcatPolicyDecoder,
    PolicyMechanismDecoder,
)


def test_parameter_matched_concat_decoder_shape_and_budget():
    mean = np.zeros(11, dtype=np.float32)
    variance = np.ones(11, dtype=np.float32)
    mechanism = PolicyMechanismDecoder(mean, variance)
    concat = ContextConcatPolicyDecoder(mean, variance)
    state = torch.randn(7, 11)
    coordinates = torch.randn(7, 3)

    assert mechanism(state, coordinates).shape == (7, 3)
    assert concat(state, coordinates).shape == (7, 3)
    torch.testing.assert_close(
        concat(state, torch.zeros_like(coordinates)),
        torch.zeros(7, 3),
    )
    mechanism_parameters = sum(p.numel() for p in mechanism.parameters())
    concat_parameters = sum(p.numel() for p in concat.parameters())
    assert abs(concat_parameters - mechanism_parameters) / mechanism_parameters < 0.01


def test_mechanism_decoder_one_hot_selects_corresponding_branch():
    decoder = PolicyMechanismDecoder(
        np.zeros(11, dtype=np.float32),
        np.ones(11, dtype=np.float32),
    )
    state = torch.randn(5, 11)
    coordinates = torch.zeros(5, 3)
    coordinates[:, 1] = 1.0

    expected = decoder.mechanism_effects(state)[:, 1, :]
    torch.testing.assert_close(decoder(state, coordinates), expected)
