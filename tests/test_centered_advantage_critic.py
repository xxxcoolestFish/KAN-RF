import gymnasium as gym
import torch
from stable_baselines3.common.torch_layers import FlattenExtractor

from cpbn.centered_advantage_critic import BaseActionCenteredCritic


def test_zero_residual_equals_learned_base_value():
    observation_space = gym.spaces.Box(
        -1.0, 1.0, shape=(5,), dtype=float,
    )
    action_space = gym.spaces.Box(
        -1.0, 1.0, shape=(2,), dtype=float,
    )
    extractor = FlattenExtractor(observation_space)
    critic = BaseActionCenteredCritic(
        observation_space,
        action_space,
        net_arch=[16, 16],
        features_extractor=extractor,
        features_dim=5,
        n_critics=2,
        share_features_extractor=True,
    )
    observations = torch.randn(7, 5)
    actions = torch.zeros(7, 2)
    q_values = critic(observations, actions)
    values = critic.values(observations)
    for q_value, value in zip(q_values, values):
        torch.testing.assert_close(q_value, value)


def test_nonzero_residual_has_finite_centered_advantage():
    observation_space = gym.spaces.Box(
        -1.0, 1.0, shape=(5,), dtype=float,
    )
    action_space = gym.spaces.Box(
        -1.0, 1.0, shape=(2,), dtype=float,
    )
    extractor = FlattenExtractor(observation_space)
    critic = BaseActionCenteredCritic(
        observation_space,
        action_space,
        net_arch=[8],
        features_extractor=extractor,
        features_dim=5,
        n_critics=2,
        share_features_extractor=True,
    )
    observations = torch.randn(4, 5)
    actions = torch.randn(4, 2).clamp(-1.0, 1.0)
    for q_value in critic(observations, actions):
        assert torch.isfinite(q_value).all()
