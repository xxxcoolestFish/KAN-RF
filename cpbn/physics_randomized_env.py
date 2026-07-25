"""Physics-randomised Hopper environment for training physics-conditioned policies.

Each episode samples new physics parameters (mass, friction, actuator) from
pre-configured ranges, resets the MuJoCo model, and exposes the oracle z
alongside the observation to the policy.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np


class PhysicsRandomizedHopper(gym.Wrapper):
    """Hopper-v5 with per-episode physics randomisation.

    At each ``reset()``, samples:
        torso_mass  ~ [mass_min, mass_max]
        friction    ~ [friction_min, friction_max]
        actuator    ~ [actuator_min, actuator_max]

    The oracle physics vector  z = [mass_ratio, friction_ratio, actuator_ratio]
    is stored in ``self.current_z`` and can be passed to the policy.

    Notes
    -----
    The observation returned by ``step()`` / ``reset()`` is the **original**
    11-dim state.  The caller is responsible for concatenating z if the
    policy expects [s, z].
    """

    def __init__(
        self,
        *,
        mass_range=(0.8, 1.5),
        friction_range=(0.6, 1.3),
        actuator_range=(0.6, 1.0),
        seed=0,
    ):
        env = gym.make("Hopper-v5")
        super().__init__(env)
        self.mass_range = tuple(mass_range)
        self.friction_range = tuple(friction_range)
        self.actuator_range = tuple(actuator_range)
        self._rng = np.random.default_rng(seed)
        self.current_z = np.ones(3, dtype=np.float32)  # source default

    @property
    def observation_space(self):
        return self.env.observation_space

    def reset(self, *, seed=None, options=None):
        self._sample_physics()
        return self.env.reset(seed=seed, options=options)

    def _sample_physics(self):
        mass = float(self._rng.uniform(*self.mass_range))
        friction = float(self._rng.uniform(*self.friction_range))
        actuator = float(self._rng.uniform(*self.actuator_range))
        self.current_z = np.array(
            [mass, friction, actuator], dtype=np.float32,
        )
        model = self.unwrapped.model
        torso_id = model.body("torso").id
        model.body_mass[torso_id] = mass
        model.geom_friction[:, 0] = friction
        model.actuator_gear[:, 0] = actuator
