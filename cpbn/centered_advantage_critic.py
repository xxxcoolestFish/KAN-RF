"""Continuous critic whose action advantage is exact at a base action."""

from __future__ import annotations

import torch
from stable_baselines3.common.policies import ContinuousCritic
from stable_baselines3.common.torch_layers import create_mlp
from stable_baselines3.td3.policies import TD3Policy
from torch import nn


class BaseActionCenteredCritic(ContinuousCritic):
    """Represent Q(s, u) as V(s) + A(s, u) - A(s, 0)."""

    def __init__(
        self,
        observation_space,
        action_space,
        net_arch,
        features_extractor,
        features_dim,
        activation_fn=nn.ReLU,
        normalize_images=True,
        n_critics=2,
        share_features_extractor=True,
    ):
        super().__init__(
            observation_space,
            action_space,
            net_arch,
            features_extractor,
            features_dim,
            activation_fn,
            normalize_images,
            n_critics,
            share_features_extractor,
        )
        self.value_networks = nn.ModuleList(
            [
                nn.Sequential(
                    *create_mlp(
                        features_dim,
                        1,
                        net_arch,
                        activation_fn,
                    )
                )
                for _ in range(self.n_critics)
            ]
        )

    def _features(self, observations):
        with torch.set_grad_enabled(
            not self.share_features_extractor,
        ):
            return self.extract_features(
                observations,
                self.features_extractor,
            )

    def forward(self, observations, actions):
        features = self._features(observations)
        zeros = torch.zeros_like(actions)
        action_input = torch.cat((features, actions), dim=1)
        base_input = torch.cat((features, zeros), dim=1)
        return tuple(
            value(features)
            + advantage(action_input)
            - advantage(base_input)
            for value, advantage in zip(
                self.value_networks,
                self.q_networks,
            )
        )

    def values(self, observations):
        features = self._features(observations)
        return tuple(value(features) for value in self.value_networks)

    def q1_forward(self, observations, actions):
        with torch.no_grad():
            return self.forward(observations, actions)[0]


class CenteredAdvantageTD3Policy(TD3Policy):
    """TD3 policy using an exact zero-residual advantage origin."""

    def make_critic(self, features_extractor=None):
        critic_kwargs = self._update_features_extractor(
            self.critic_kwargs,
            features_extractor,
        )
        return BaseActionCenteredCritic(**critic_kwargs).to(self.device)
