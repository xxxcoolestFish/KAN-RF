"""Environment wrapper that appends physics latent z to observation.

Used by CPPE training: the policy sees (s, z) as its observation, where z
can be set externally (z=z_source during PPO rollout, z=z' during physics
supervised training).
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np


class PhysicsConditionedEnv(gym.Wrapper):
    """Appends physics latent z to the observation.

    The policy sees obs = [s, z] instead of just s.  During PPO rollout
    z is fixed to z_source; during physics-supervised training z is
    sampled from the PCA physics manifold.
    """

    def __init__(self, env: gym.Env, z_dim: int = 5):
        super().__init__(env)
        s_dim = env.observation_space.shape[0]
        self.z_dim = z_dim
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(s_dim + z_dim,),
            dtype=np.float32,
        )
        self._current_z: np.ndarray = np.zeros(z_dim, dtype=np.float32)

    def set_z(self, z: np.ndarray) -> None:
        """Set the physics latent for the next step(s)."""
        self._current_z = np.asarray(z, dtype=np.float32).flatten()

    def _augment_obs(self, obs: np.ndarray) -> np.ndarray:
        return np.concatenate([obs, self._current_z])

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._augment_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._augment_obs(obs), reward, terminated, truncated, info


class BatchedConditionedEnv:
    """Batched wrapper that sets z for all parallel envs simultaneously."""

    def __init__(self, vec_env, z_dim: int = 5):
        self.vec_env = vec_env
        self.z_dim = z_dim
        # Expand observation space
        s_dim = vec_env.observation_space.shape[0]
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(s_dim + z_dim,),
            dtype=np.float32,
        )
        self.action_space = vec_env.action_space
        self._current_z = np.zeros((vec_env.num_envs, z_dim), dtype=np.float32)

    @property
    def num_envs(self):
        return self.vec_env.num_envs

    def set_z(self, z: np.ndarray) -> None:
        """Set z for all envs. z can be (z_dim,) or (num_envs, z_dim)."""
        z = np.asarray(z, dtype=np.float32)
        if z.ndim == 1:
            z = np.tile(z[None, :], (self.num_envs, 1))
        self._current_z = z

    def _augment_obs(self, obs: np.ndarray) -> np.ndarray:
        return np.concatenate([obs, self._current_z], axis=-1)

    def reset(self):
        obs = self.vec_env.reset()
        return self._augment_obs(obs)

    def step(self, actions):
        obs, rewards, dones, infos = self.vec_env.step(actions)
        return self._augment_obs(obs), rewards, dones, infos
