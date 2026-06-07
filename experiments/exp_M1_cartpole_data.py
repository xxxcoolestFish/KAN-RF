#!/usr/bin/env python3
"""Generate CartPole training data via analytical simulation.

Output format:
  Input:  [cosθ, sinθ, x, x_dot, th_dot, force] — 6D
  Target: [x_ddot, th_ddot]                      — 2D (acceleration)

CartPole has x as a cyclic coordinate (∂L/∂x = 0), so the dynamics
depend only on (cosθ, sinθ, x_dot, th_dot, force).  We include x
for the controller's MPC cost (cart centering penalty).

Physics: dt=0.02, gravity=9.8, masscart=1.0, masspole=0.1, length=0.5
"""
import numpy as np, torch, time, argparse

G = 9.8; MC = 1.0; MP = 0.1; L = 0.5; FM = 10.0; DT = 0.02
TOTAL_MASS = MC + MP
POLE_MASS_LEN = MP * L
I_POLE_PIVOT = (4.0/3.0) * MP * L**2  # rotational inertia about pivot


def step_cartpole(state, action):
    """Single-step CartPole dynamics (vectorized). Returns (next_state, accelerations)."""
    x, x_dot, theta, theta_dot = (state[:, i] for i in range(4))
    force = np.where(action == 1, FM, -FM)
    costheta = np.cos(theta)
    sintheta = np.sin(theta)
    temp = (force + POLE_MASS_LEN * theta_dot**2 * sintheta) / TOTAL_MASS
    theta_acc = (G * sintheta - costheta * temp) / \
                (L * (4.0/3.0 - MP * costheta**2 / TOTAL_MASS))
    x_acc = temp - POLE_MASS_LEN * theta_acc * costheta / TOTAL_MASS

    next_state = np.stack([
        x + x_dot * DT,
        x_dot + x_acc * DT,
        theta + theta_dot * DT,
        theta_dot + theta_acc * DT,
    ], axis=-1)

    return next_state, np.stack([x_acc, theta_acc], axis=-1)


def generate_random(n_samples, seed=42):
    """Generate data from random states + random actions."""
    rng = np.random.RandomState(seed)
    state = np.zeros((n_samples, 4))
    state[:, 0] = rng.uniform(-2.0, 2.0, n_samples)     # cart position
    state[:, 1] = rng.uniform(-3.0, 3.0, n_samples)     # cart velocity
    state[:, 2] = rng.uniform(-0.3, 0.3, n_samples)     # pole angle (rad)
    state[:, 3] = rng.uniform(-4.0, 4.0, n_samples)     # pole ang velocity
    action = rng.randint(0, 2, n_samples)

    next_s, acc = step_cartpole(state, action)
    return state, action, acc


def generate_energy_trajectories(n_episodes=500, max_steps=200, seed=123):
    """Generate swing-up-like trajectories using energy-pump heuristic.

    Energy E = ½(M+m)ẋ² + ½Iθ̇² + mLẋθ̇cosθ + mgLcosθ
    Energy-pump: choose action that maximizes instantaneous dE/dt (energy rate).

    dE/dt = F·ẋ  (power input = force × cart velocity)
    → If ẋ > 0, push right (F=+10) to add energy; if ẋ < 0, push left.
    → But also need to account for the pole's energy exchange.

    Simplified heuristic: action = sign(ẋ) when pole is swinging toward upright,
    action = -sign(ẋ) when pole is moving away.
    """
    rng = np.random.RandomState(seed)
    all_state = []
    all_action = []
    all_acc = []

    for ep in range(n_episodes):
        # Random initial state
        state_i = np.zeros(4)
        state_i[0] = rng.uniform(-1.5, 1.5)
        state_i[1] = rng.uniform(-2.0, 2.0)
        state_i[2] = rng.uniform(-0.25, 0.25)
        state_i[3] = rng.uniform(-3.0, 3.0)

        for step in range(max_steps):
            costh, sinth = np.cos(state_i[2]), np.sin(state_i[2])
            thd = state_i[3]

            # Energy-pump: push in direction of cart velocity to maximize power input
            # But also consider the pole's dynamics:
            #   θ̈ ≈ (g sinθ - cosθ·F/TOTAL_MASS) / (4/3·L)
            # If we want θ toward upright (sinθ=0), push when cosθ·sign(θ) is favorable
            if abs(state_i[2]) < 0.02 and abs(thd) < 0.5:
                # Near upright — random small corrections
                action_i = rng.randint(0, 2)
            elif abs(thd) > 2.0:
                # Large angular velocity — recenter cart
                action_i = 1 if state_i[0] < -0.5 else (0 if state_i[0] > 0.5 else rng.randint(0, 2))
            else:
                # Energy pump: push with cart velocity
                if abs(state_i[1]) > 0.1:
                    action_i = 1 if state_i[1] > 0 else 0
                else:
                    action_i = rng.randint(0, 2)

            next_s, acc = step_cartpole(state_i.reshape(1, -1), np.array([action_i]))
            all_state.append(state_i.copy())
            all_action.append(action_i)
            all_acc.append(acc[0].copy())

            state_i = next_s[0]

            # Terminate if pole angle too large or cart out of bounds
            if abs(state_i[2]) > 0.5 or abs(state_i[0]) > 3.0:
                break

    return (np.array(all_state), np.array(all_action), np.array(all_acc))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-random', type=int, default=80000)
    parser.add_argument('--n-energy-episodes', type=int, default=300)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save', type=str, default='saved_data/cartpole_lag_data.pt')
    args = parser.parse_args()

    t0 = time.time()

    # 1. Random data
    print(f"Generating {args.n_random} random samples...")
    state_r, action_r, acc_r = generate_random(args.n_random, seed=args.seed)
    print(f"  x_ddot range: [{acc_r[:,0].min():.1f}, {acc_r[:,0].max():.1f}]")
    print(f"  th_ddot range: [{acc_r[:,1].min():.1f}, {acc_r[:,1].max():.1f}]")

    # 2. Energy-pump trajectory data
    print(f"Generating {args.n_energy_episodes} energy-pump episodes...")
    state_e, action_e, acc_e = generate_energy_trajectories(
        args.n_energy_episodes, seed=args.seed + 1)
    print(f"  {len(state_e)} transitions from {args.n_energy_episodes} episodes")
    print(f"  avg steps/episode: {len(state_e)/args.n_energy_episodes:.0f}")

    # 3. Combine
    all_state = np.concatenate([state_r, state_e], axis=0)
    all_action = np.concatenate([action_r, action_e], axis=0)
    all_acc = np.concatenate([acc_r, acc_e], axis=0)

    # Shuffle
    perm = np.random.RandomState(args.seed).permutation(len(all_state))
    all_state = all_state[perm]
    all_action = all_action[perm]
    all_acc = all_acc[perm]

    # Build input/output tensors
    # Input: [cosθ, sinθ, x, x_dot, th_dot, force]
    cos = torch.tensor(np.cos(all_state[:, 2]), dtype=torch.float32)
    sin = torch.tensor(np.sin(all_state[:, 2]), dtype=torch.float32)
    x = torch.tensor(all_state[:, 0], dtype=torch.float32)
    x_dot = torch.tensor(all_state[:, 1], dtype=torch.float32)
    th_dot = torch.tensor(all_state[:, 3], dtype=torch.float32)
    force = torch.tensor(np.where(all_action == 1, FM, -FM), dtype=torch.float32)

    X = torch.stack([cos, sin, x, x_dot, th_dot, force], dim=-1)  # (N, 6)
    Y = torch.tensor(all_acc, dtype=torch.float32)                 # (N, 2)

    n_total = len(X)
    # 80/20 train/val split
    n_train = int(n_total * 0.8)
    train = (X[:n_train], Y[:n_train])
    val = (X[n_train:], Y[n_train:])

    torch.save({'train': train, 'val': val, 'n_total': n_total}, args.save)
    print(f"\nSaved to {args.save}: {n_total} samples "
          f"({n_train} train, {n_total-n_train} val) "
          f"in {time.time()-t0:.1f}s")

    # Statistics
    print(f"\nAcceleration statistics:")
    print(f"  ẍ:   mean={Y[:,0].mean():.2f}, std={Y[:,0].std():.2f}, "
          f"min={Y[:,0].min():.1f}, max={Y[:,0].max():.1f}")
    print(f"  θ̈:   mean={Y[:,1].mean():.2f}, std={Y[:,1].std():.2f}, "
          f"min={Y[:,1].min():.1f}, max={Y[:,1].max():.1f}")
    print(f"\nState statistics:")
    print(f"  θ:    mean={np.rad2deg(all_state[:,2]).mean():.1f}°, "
          f"max_abs={np.rad2deg(abs(all_state[:,2]).max()):.0f}°")
    print(f"  θ̇:    mean={all_state[:,3].mean():.2f}, "
          f"max_abs={abs(all_state[:,3]).max():.1f}")
    print(f"  ẋ:    mean={all_state[:,1].mean():.2f}, "
          f"max_abs={abs(all_state[:,1]).max():.1f}")


if __name__ == '__main__':
    main()
