"""Generate multi-scale CartPole training data via analytical simulation.

CartPole-v1 dynamics (from gymnasium source):
  gravity=9.8, masscart=1.0, masspole=0.1, length=0.5
  force_mag=10.0, tau=0.02 (dt)

State: [cart_pos, cart_vel, pole_angle, pole_ang_vel] (4D)
Action: discrete {0=push_left, 1=push_right}
"""
import torch, numpy as np, time

G = 9.8; MC = 1.0; MP = 0.1; L = 0.5; FM = 10.0; DT = 0.02
K_VALS = [1, 2, 4, 8, 16]
TOTAL_MASS = MC + MP
POLE_MASS_LEN = MP * L


def step_cartpole(state, action):
    """Single CartPole step.  Returns next state.  Vectorized over batch."""
    x, x_dot, theta, theta_dot = (state[:, i] for i in range(4))
    force = torch.where(action.squeeze() == 1, FM, -FM)

    costheta = torch.cos(theta)
    sintheta = torch.sin(theta)

    temp = (force + POLE_MASS_LEN * theta_dot**2 * sintheta) / TOTAL_MASS
    theta_acc = (G * sintheta - costheta * temp) / \
                (L * (4.0/3.0 - MP * costheta**2 / TOTAL_MASS))
    x_acc = temp - POLE_MASS_LEN * theta_acc * costheta / TOTAL_MASS

    x_new = x + x_dot * DT
    x_dot_new = x_dot + x_acc * DT
    theta_new = theta + theta_dot * DT
    theta_dot_new = theta_dot + theta_acc * DT

    return torch.stack([x_new, x_dot_new, theta_new, theta_dot_new], dim=-1)


def simulate_k_steps(state, action, k):
    """Simulate CartPole forward k steps with constant action."""
    s = state
    for _ in range(k):
        s = step_cartpole(s, action)
    return s


def main():
    torch.manual_seed(42)
    t0 = time.time()

    # Sample random states + actions
    N = 20000
    state = torch.rand(N, 4)
    state[:, 0] = state[:, 0] * 5.0 - 2.5    # cart_pos ∈ [-2.5, 2.5]
    state[:, 1] = state[:, 1] * 6.0 - 3.0    # cart_vel ∈ [-3.0, 3.0]
    state[:, 2] = state[:, 2] * 0.6 - 0.3    # pole_angle ∈ [-0.3, 0.3]
    state[:, 3] = state[:, 3] * 6.0 - 3.0    # pole_ang_vel ∈ [-3.0, 3.0]
    action = torch.randint(0, 2, (N,))

    # Normalize state to approx [-1, 1]
    s_norm = state.clone()
    s_norm[:, 0] /= 2.5
    s_norm[:, 1] /= 3.0
    s_norm[:, 2] /= 0.3
    s_norm[:, 3] /= 3.0

    xs, ys = [], []
    for k in K_VALS:
        s_next = simulate_k_steps(state, action, k)
        y_norm = s_next.clone()
        y_norm[:, 0] /= 2.5
        y_norm[:, 1] /= 3.0
        y_norm[:, 2] /= 0.3
        y_norm[:, 3] /= 3.0

        a_onehot = torch.zeros(N, 2)
        a_onehot[torch.arange(N), action] = 1.0
        k_norm = torch.full((N, 1), k / 16.0)
        x_k = torch.cat([s_norm, a_onehot, k_norm], dim=-1)  # (N, 7)
        xs.append(x_k)
        ys.append(y_norm)

        pole_angle_change = (s_next[:, 2] - state[:, 2]).abs()
        print(f'  k={k:2d}: {N} samples, pole_angle |delta| '
              f'mean={pole_angle_change.mean():.4f} max={pole_angle_change.max():.4f} rad')

    x_ms = torch.cat(xs, dim=0)
    y_ms = torch.cat(ys, dim=0)
    perm = torch.randperm(len(x_ms))
    x_ms, y_ms = x_ms[perm], y_ms[perm]

    torch.save((x_ms, y_ms), 'cartpole_data_ms.pt')
    print(f'\nSaved cartpole_data_ms.pt: {x_ms.shape[0]} samples, '
          f'{x_ms.shape[1]}d -> {y_ms.shape[1]}d, {time.time()-t0:.1f}s')


if __name__ == '__main__':
    main()
