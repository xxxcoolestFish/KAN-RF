"""Generate training data with better stabilization coverage.

v1 problem: energy controller swung up but couldn't stabilize, so training data
had almost no examples of fine control near upright.

v2 fix:
  1. Better-tuned controller that can actually stabilize
  2. Extra stabilization-only episodes: start near upright, apply small torques
  3. Data augmentation: oversample transitions with sin>0.7
"""
import torch
import gymnasium as gym
import numpy as np


class BetterController:
    """Energy swing-up + properly-tuned PD stabilization."""

    def __init__(self):
        self.k_e = 8.0          # energy gain
        self.k_p = 80.0         # PD proportional (higher = tighter stabilization)
        self.k_d = 15.0         # PD derivative (higher = more damping)
        self.g = 10.0
        self.E_des = self.g
        self.max_torque = 2.0

    def __call__(self, obs):
        cos_th, sin_th, thd = obs

        # Angle error from upright: -cos_th is approximately the angular error
        # for small deviations (since cos(π/2 + φ) ≈ -φ)
        angle_err = -cos_th

        # Energy-based swing-up
        E = 0.5 * thd * thd + self.g * sin_th
        u_energy = self.k_e * (E - self.E_des) * thd

        # PD stabilization
        u_pd = self.k_p * angle_err + self.k_d * (-thd)

        # Blend near upright
        near_upright = abs(cos_th) < 0.7 and abs(thd) < 5.0
        if near_upright:
            alpha = min(1.0, (0.7 - abs(cos_th)) / 0.7)
            u = alpha * u_pd + (1 - alpha) * u_energy
        else:
            u = u_energy

        return np.clip(u, -self.max_torque, self.max_torque)


def collect_episodes(controller, env, n_episodes):
    """Run full episodes."""
    states, actions, next_states = [], [], []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        for _ in range(200):
            a = np.array([controller(obs)], dtype=np.float32)
            next_obs, _, term, trunc, _ = env.step(a)
            states.append(obs.copy())
            actions.append(a.item())
            next_states.append(next_obs.copy())
            obs = next_obs
            if term or trunc:
                break
    return states, actions, next_states


def collect_stabilization(env, n_episodes=50, steps_per_ep=100):
    """Stabilization-only episodes: start randomly, controller drives to upright,
    then we collect transitions specifically from the stabilization phase."""
    ctrl = BetterController()
    states, actions, next_states = [], [], []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        # First, swing up to upright (skip collection during swing-up)
        for _ in range(50):
            a = ctrl(obs)
            obs, _, term, trunc, _ = env.step([a])
            if term or trunc:
                break

        # Now collect stabilization data
        for _ in range(steps_per_ep):
            # Add small exploratory noise
            torque = ctrl(obs) + np.random.uniform(-0.3, 0.3)
            torque = np.clip(torque, -2.0, 2.0)
            a = np.array([torque], dtype=np.float32)
            next_obs, _, term, trunc, _ = env.step(a)
            states.append(obs.copy())
            actions.append(a.item())
            next_states.append(next_obs.copy())
            obs = next_obs
            if term or trunc:
                break

    return states, actions, next_states


def to_tensors(states, actions, next_states):
    s = torch.tensor(np.array(states), dtype=torch.float32)
    a = torch.tensor(np.array(actions), dtype=torch.float32).unsqueeze(-1)
    sn = torch.tensor(np.array(next_states), dtype=torch.float32)

    s_norm = s.clone(); s_norm[:, 2] /= 8.0
    a_norm = a / 2.0
    sn_norm = sn.clone(); sn_norm[:, 2] /= 8.0
    x = torch.cat([s_norm, a_norm], dim=-1)
    return x, sn_norm


def main():
    np.random.seed(42)
    env = gym.make("Pendulum-v1")

    # 1. Full episodes with better controller
    ctrl = BetterController()
    s1, a1, ns1 = collect_episodes(ctrl, env, n_episodes=40)
    x1, y1 = to_tensors(s1, a1, ns1)
    print(f"Full episodes:  {len(x1)} transitions")

    # 2. Stabilization-only data
    s2, a2, ns2 = collect_stabilization(env, n_episodes=120, steps_per_ep=100)
    x2, y2 = to_tensors(s2, a2, ns2)
    print(f"Stabilization:  {len(x2)} transitions")

    # 3. Random data from v1
    x_rand, y_rand = torch.load("pendulum_data.pt", weights_only=True)

    # Quality check: fraction of stabilization data near upright
    near = (y2[:, 1] > 0.7).float().mean().item()
    print(f"Stab data near upright (sin>0.7): {near:.1%}")

    # Merge: 30% full episodes + 30% stabilization + 40% random
    n_use = min(len(x1), 5000)
    n_stab = min(len(x2), 10000)
    n_rand = min(len(x_rand), 8000)

    x = torch.cat([x1[:n_use], x2[:n_stab], x_rand[:n_rand]], dim=0)
    y = torch.cat([y1[:n_use], y2[:n_stab], y_rand[:n_rand]], dim=0)
    perm = torch.randperm(len(x))
    x, y = x[perm], y[perm]

    print(f"Merged: {len(x)} transitions ({n_use} episodes + {n_stab} stab + {n_rand} random)")

    torch.save((x, y), "pendulum_data_v3.pt")
    print("Saved: pendulum_data_v3.pt")

    # Demo: show trajectory quality
    obs, _ = env.reset()
    traj = [obs.copy()]
    for _ in range(200):
        a = ctrl(obs)
        obs, _, term, trunc, _ = env.step([a])
        traj.append(obs.copy())
        if term or trunc:
            break
    traj = np.array(traj)
    final = traj[-1]
    # Count steps near upright (sin>0.9)
    near_steps = (traj[:, 1] > 0.9).sum()
    print(f"\nDemo: {len(traj)} steps, {near_steps} steps near upright (sin>0.9)")
    print(f"  Final: [{final[0]:+.3f}, {final[1]:+.3f}, {final[2]:+.3f}]")
    print(f"  |Δθ|={abs(np.arctan2(final[1], final[0]) - np.pi/2):.3f} rad")

    env.close()


if __name__ == "__main__":
    main()
