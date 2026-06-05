"""Generate training data using energy-based swing-up controller.

Adds swing-up + stabilization trajectories to supplement random data,
so the KAN world model learns the full dynamics including upright balancing.
"""
import torch
import gymnasium as gym
import numpy as np


class EnergyController:
    """Energy-based swing-up + PD stabilization for Pendulum-v1.

    Energy shaping pumps/removes energy by applying torque aligned with velocity.
    Near upright, switches to PD for fine stabilization.
    """

    def __init__(self, k_e=8.0, k_p=40.0, k_d=4.0):
        self.k_e = k_e      # energy gain
        self.k_p = k_p      # PD proportional gain
        self.k_d = k_d      # PD derivative gain
        self.g = 10.0       # gravity
        self.E_des = self.g  # desired energy at upright (sin=1, thetadot=0)
        self.max_torque = 2.0

    def __call__(self, obs):
        cos_th, sin_th, thd = obs

        # Current energy: E = 0.5*thd^2 + g*sin(th)
        E = 0.5 * thd * thd + self.g * sin_th

        # Energy-based swing-up torque
        u_energy = self.k_e * (E - self.E_des) * thd

        # PD stabilization (target: cos=0, sin=1, thd=0)
        u_pd = self.k_p * (-cos_th) + self.k_d * (-thd)

        # Blend: near upright and slow → more PD; otherwise → pure energy
        near_upright = abs(cos_th) < 0.3 and abs(thd) < 2.0
        if near_upright:
            alpha = min(1.0, (0.3 - abs(cos_th)) / 0.3)  # 0→1 as cos→0
            u = alpha * u_pd + (1 - alpha) * u_energy
        else:
            u = u_energy

        return np.clip(u, -self.max_torque, self.max_torque)


def collect(controller, env, n_episodes=50):
    """Run episodes with the controller, return (s, a, s') transitions."""
    states, actions, next_states = [], [], []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        step = 0
        while not done and step < 200:
            a = np.array([controller(obs)], dtype=np.float32)
            next_obs, _, terminated, truncated, _ = env.step(a)

            states.append(obs.copy())
            actions.append(a.item())
            next_states.append(next_obs.copy())

            obs = next_obs
            step += 1
            if terminated or truncated:
                break

    s = torch.tensor(np.array(states), dtype=torch.float32)
    a = torch.tensor(np.array(actions), dtype=torch.float32).unsqueeze(-1)
    s_next = torch.tensor(np.array(next_states), dtype=torch.float32)

    # Normalize
    s_norm = s.clone(); s_norm[:, 2] /= 8.0
    a_norm = a / 2.0
    s_next_norm = s_next.clone(); s_next_norm[:, 2] /= 8.0
    x = torch.cat([s_norm, a_norm], dim=-1)  # (N, 4)

    return x, s_next_norm


def main():
    np.random.seed(42)
    env = gym.make("Pendulum-v1")

    # Collect with energy controller
    ctrl = EnergyController()
    x_ctrl, y_ctrl = collect(ctrl, env, n_episodes=60)
    print(f"Controller data: {len(x_ctrl)} transitions")

    # Collect random data
    x_rand, y_rand = torch.load("pendulum_data.pt", weights_only=True)
    print(f"Random data:     {len(x_rand)} transitions")

    # Quick controller quality check
    s_norm = y_ctrl[:, :2]
    near_upright = (s_norm[:, 1] > 0.8).float().mean().item()
    print(f"Fraction near upright (sin>0.8): {near_upright:.3f}")

    # Merge — 50/50 split
    n_use = min(len(x_ctrl), 8000)
    x_mixed = torch.cat([x_ctrl[:n_use], x_rand], dim=0)
    y_mixed = torch.cat([y_ctrl[:n_use], y_rand], dim=0)

    # Shuffle
    perm = torch.randperm(len(x_mixed))
    x_mixed, y_mixed = x_mixed[perm], y_mixed[perm]

    print(f"Merged data:     {len(x_mixed)} transitions "
          f"({n_use} ctrl + {len(x_rand)} random)")

    torch.save((x_mixed, y_mixed), "pendulum_data_mixed.pt")
    print("Saved: pendulum_data_mixed.pt")

    # Quick test: run one episode and show states
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
    print(f"\nDemo episode ({len(traj)} steps):")
    print(f"  Final state: [{final[0]:+.3f}, {final[1]:+.3f}, {final[2]:+.3f}]")
    print(f"  |Δθ| from upright: {abs(np.arctan2(final[1], final[0]) - np.pi/2):.3f} rad")

    env.close()


if __name__ == "__main__":
    main()
