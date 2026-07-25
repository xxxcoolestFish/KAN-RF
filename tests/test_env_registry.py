"""Environment registry and Walker2d generalization smoke tests."""

import numpy as np
import torch

from cpbn.generic_affine_kan import CompactInteractionKANDictionary
from scripts.prescreen_hopper_physics_shifts import (
    ENVS,
    SHIFTS,
    make_shifted_env,
)


def test_registry_contains_both_families():
    assert ENVS["hopper"]["gym_id"] == "Hopper-v5"
    assert ENVS["walker2d"]["gym_id"] == "Walker2d-v5"
    assert set(SHIFTS) == {
        "source",
        "payload_125",
        "payload_150",
        "friction_070",
        "actuator_080",
        "actuator_065",
        "combo_mild",
        "combo_medium",
    }


def test_walker2d_dictionary_dimensions():
    basis = CompactInteractionKANDictionary(
        torch.ones(17),
        torch.ones(6),
        pair_modes=1,
    )
    state = torch.randn(8, 17)
    action = torch.randn(8, 6)
    assert basis.feature_dim == 222
    assert basis(state).shape == (8, 222)
    assert basis.context_features(state, action).shape == (8, 7 * 222)


def test_walker2d_shift_factory_steps_and_scales():
    source = make_shifted_env(SHIFTS["source"], 123, "walker2d")()
    shifted = make_shifted_env(SHIFTS["combo_medium"], 123, "walker2d")()
    source_model = source.unwrapped.model
    shifted_model = shifted.unwrapped.model
    torso_id = source_model.body("torso").id
    mass_ratio = (
        shifted_model.body_mass[torso_id] / source_model.body_mass[torso_id]
    )
    assert np.isclose(mass_ratio, 1.35)
    friction_ratio = (
        shifted_model.geom_friction[:, 0] / source_model.geom_friction[:, 0]
    )
    assert np.allclose(friction_ratio, 0.70)
    gear_ratio = (
        shifted_model.actuator_gear[:, 0] / source_model.actuator_gear[:, 0]
    )
    assert np.allclose(gear_ratio, 0.75)
    observation, _ = shifted.reset(seed=123)
    assert observation.shape == (17,)
    following, _, terminated, truncated, _ = shifted.step(
        shifted.action_space.sample()
    )
    assert following.shape == (17,)
    source.close()
    shifted.close()


def test_default_env_remains_hopper():
    environment = make_shifted_env(SHIFTS["source"], 123)()
    assert environment.unwrapped.spec.id == "Hopper-v5"
    environment.close()
