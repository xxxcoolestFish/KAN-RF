"""SB3-compatible ProtoKAN feature extractor."""

import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from kanrf._protokan import ProtoKANLayer


class ProtoKANFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, *,
                 n_prototypes=16, out_dim=256, grid_range=1.0):
        state_dim = observation_space.shape[0]
        super().__init__(observation_space, features_dim=out_dim)
        self.layer = ProtoKANLayer(state_dim, out_dim,
                                   n_prototypes=n_prototypes,
                                   grid_range=grid_range)
        self._features_dim = out_dim

    def forward(self, obs):
        return self.layer(obs)
