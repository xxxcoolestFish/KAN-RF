"""Collect (s, a, s') transitions from Pendulum-v1 with random actions."""
import torch
import gymnasium as gym


def collect_data(n_transitions: int = 15000) -> tuple[torch.Tensor, torch.Tensor]:
    env = gym.make('Pendulum-v1')

    states, actions, next_states = [], [], []

    obs, _ = env.reset()
    for _ in range(n_transitions):
        a = env.action_space.sample()  # uniform in [-2, 2]
        next_obs, _, terminated, truncated, _ = env.step(a)

        states.append(obs.copy())
        actions.append(a.item())
        next_states.append(next_obs.copy())

        if terminated or truncated:
            obs, _ = env.reset()
        else:
            obs = next_obs

    s = torch.tensor(states, dtype=torch.float32)        # (N, 3)
    a = torch.tensor(actions, dtype=torch.float32).unsqueeze(-1)  # (N, 1)
    s_next = torch.tensor(next_states, dtype=torch.float32)  # (N, 3)

    # Normalize theta_dot (range ~ [-8, 8]) and torque (range [-2, 2]) to [-1, 1]
    s_norm = s.clone()
    s_norm[:, 2] = s[:, 2] / 8.0       # theta_dot

    a_norm = a / 2.0                    # torque

    s_next_norm = s_next.clone()
    s_next_norm[:, 2] = s_next[:, 2] / 8.0

    # Input to world model: concat(s_norm, a_norm)
    x = torch.cat([s_norm, a_norm], dim=-1)  # (N, 4)

    print(f"Collected {n_transitions} transitions")
    print(f"  x: {x.shape}, range [{x.min():.3f}, {x.max():.3f}]")
    print(f"  y: {s_next_norm.shape}, range [{s_next_norm.min():.3f}, {s_next_norm.max():.3f}]")

    return x, s_next_norm


if __name__ == "__main__":
    x, y = collect_data()
    torch.save((x, y), "pendulum_data.pt")
    print("Saved: pendulum_data.pt")
