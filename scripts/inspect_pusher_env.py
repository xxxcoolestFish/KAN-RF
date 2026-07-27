"""Print reproducible environment, dependency and Pusher state diagnostics."""

from __future__ import annotations

import platform
import sys

import gymnasium as gym
import mujoco
import numpy as np
import stable_baselines3
import torch


def main() -> None:
    print("=== Runtime ===", flush=True)
    print(f"python={sys.version}", flush=True)
    print(f"platform={platform.platform()}", flush=True)
    print(f"torch={torch.__version__}", flush=True)
    print(f"cuda_available={torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"cuda_device={torch.cuda.get_device_name(0)}", flush=True)
    print(f"gymnasium={gym.__version__}", flush=True)
    print(f"mujoco={mujoco.__version__}", flush=True)
    print(f"stable_baselines3={stable_baselines3.__version__}", flush=True)

    env = gym.make("Pusher-v5")
    observation, info = env.reset(seed=1811)
    print("\n=== Pusher-v5 ===", flush=True)
    print(f"observation_shape={observation.shape}", flush=True)
    print(f"action_shape={env.action_space.shape}", flush=True)
    print(f"action_low={env.action_space.low}", flush=True)
    print(f"action_high={env.action_space.high}", flush=True)
    print(f"qpos_shape={env.unwrapped.data.qpos.shape}", flush=True)
    print(f"qvel_shape={env.unwrapped.data.qvel.shape}", flush=True)
    print(f"dt={env.unwrapped.dt}", flush=True)
    print(f"reset_info={info}", flush=True)

    action = np.zeros(env.action_space.shape, dtype=np.float32)
    next_observation, reward, terminated, truncated, step_info = env.step(action)
    print("\n=== Deterministic zero step ===", flush=True)
    print(f"reward={reward:.6f}", flush=True)
    print(f"reward_terms={step_info}", flush=True)
    print(f"terminated={terminated} truncated={truncated}", flush=True)
    print(
        f"observation_delta_norm={np.linalg.norm(next_observation-observation):.6f}",
        flush=True,
    )
    env.close()


if __name__ == "__main__":
    main()

