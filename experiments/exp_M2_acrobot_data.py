#!/usr/bin/env python3
"""Generate Acrobot training data using analytical Lagrangian dynamics.

Physics:
  M(q)q̈ + C(q,q̇)q̇ + ∇U = [0, τ]ᵀ

Uses analytical formulas (verified against env.step() in _check_acrobot_formula.py).
Acrobot env uses RK4 integration which introduces small deviations (~1-2% error),
but the Lagrangian formulas are the ground truth physics.

Output: saved_data/acrobot_lag_data.pt
  train/val: {'train': (X, Y), 'val': (X, Y)}
  X: (N, 8) = [cosθ₁, sinθ₁, cosθ₂, sinθ₂, θ̇₁, θ̇₂, torque]  [all raw scale]
  Y: (N, 2) = [θ̈₁, θ̈₂]  [rad/s²]
"""
import numpy as np, torch, time, argparse

# Acrobot physical parameters
l1 = 1.0; l2 = 1.0
lc1 = 0.5; lc2 = 0.5
m1 = 1.0; m2 = 1.0
I1 = 1.0; I2 = 1.0
g = 9.8
dt = 0.2


def mass_matrix(cos2):
    """M₁₁, M₁₂, M₂₂."""
    M11 = I1 + m1*lc1**2 + m2*(l1**2 + lc2**2 + 2*l1*lc2*cos2) + I2
    M12 = m2*(lc2**2 + l1*lc2*cos2) + I2
    M22 = m2*lc2**2 + I2
    return M11, M12, M22


def coriolis(sin2, thd1, thd2):
    """C₁, C₂ generalized forces."""
    b1 = 2.0 * m2 * l1 * lc2  # = 1.0
    b2 = m2 * l1 * lc2         # = 0.5
    C1 = -b1 * sin2 * thd1 * thd2 - b2 * sin2 * thd2**2
    C2 = 0.5 * b1 * sin2 * thd1**2
    return C1, C2


def gravity(cos1, sin1, cos2, sin2):
    """g₁, g₂ = ∇U."""
    g1 = (m1*lc1 + m2*l1) * g * sin1 + m2*lc2 * g * np.sin(np.arctan2(sin1, cos1) + np.arctan2(sin2, cos2))
    # More efficient: sin(θ₁+θ₂) = sinθ₁cosθ₂ + cosθ₁sinθ₂
    sin12 = sin1 * cos2 + cos1 * sin2
    g1 = (m1*lc1 + m2*l1) * g * sin1 + m2*lc2 * g * sin12
    g2 = m2*lc2 * g * sin12
    return g1, g2


def lagrangian_accel(cos1, sin1, cos2, sin2, thd1, thd2, torque):
    """Compute [θ̈₁, θ̈₂] from Lagrangian dynamics."""
    M11, M12, M22 = mass_matrix(cos2)
    C1, C2 = coriolis(sin2, thd1, thd2)
    g1, g2 = gravity(cos1, sin1, cos2, sin2)

    det = M11 * M22 - M12**2
    rhs1 = -C1 - g1
    rhs2 = torque - C2 - g2

    thdd1 = (M22 * rhs1 - M12 * rhs2) / det
    thdd2 = (-M12 * rhs1 + M11 * rhs2) / det
    return thdd1, thdd2


def semi_implicit_euler(cos1, sin1, cos2, sin2, thd1, thd2, torque, ddt=dt):
    """One-step semi-implicit Euler using Lagrangian dynamics."""
    thdd1, thdd2 = lagrangian_accel(cos1, sin1, cos2, sin2, thd1, thd2, torque)
    theta1 = np.arctan2(sin1, cos1)
    theta2 = np.arctan2(sin2, cos2)
    thd1_n = thd1 + ddt * thdd1
    thd2_n = thd2 + ddt * thdd2
    theta1_n = theta1 + ddt * thd1_n
    theta2_n = theta2 + ddt * thd2_n
    return (np.cos(theta1_n), np.sin(theta1_n),
            np.cos(theta2_n), np.sin(theta2_n), thd1_n, thd2_n)


def generate_random(n_samples, seed=42):
    """Random (state, torque) pairs + exact accelerations."""
    rng = np.random.RandomState(seed)
    th1 = rng.uniform(-np.pi, np.pi, n_samples)
    th2 = rng.uniform(-np.pi, np.pi, n_samples)
    dth1 = rng.uniform(-6.0, 6.0, n_samples)
    dth2 = rng.uniform(-9.0, 9.0, n_samples)
    torque = rng.choice([-1.0, 0.0, 1.0], n_samples)

    cos1, sin1 = np.cos(th1), np.sin(th1)
    cos2, sin2 = np.cos(th2), np.sin(th2)

    thdd1, thdd2 = lagrangian_accel(cos1, sin1, cos2, sin2, dth1, dth2, torque)

    X = np.stack([cos1, sin1, cos2, sin2, dth1, dth2, torque], axis=-1)
    Y = np.stack([thdd1, thdd2], axis=-1)
    return X, Y


def generate_trajectories(n_episodes=500, max_steps=200, seed=123):
    """Random-walk trajectories for coverage of dynamically relevant states."""
    rng = np.random.RandomState(seed)
    all_X, all_Y = [], []

    for ep in range(n_episodes):
        th1 = rng.uniform(-np.pi, np.pi)
        th2 = rng.uniform(-np.pi, np.pi)
        dth1 = rng.uniform(-3.0, 3.0)
        dth2 = rng.uniform(-4.0, 4.0)

        for step in range(max_steps):
            torque = rng.choice([-1.0, 0.0, 1.0])

            cos1, sin1 = np.cos(th1), np.sin(th1)
            cos2, sin2 = np.cos(th2), np.sin(th2)

            thdd1, thdd2 = lagrangian_accel(cos1, sin1, cos2, sin2, dth1, dth2, torque)

            all_X.append([cos1, sin1, cos2, sin2, dth1, dth2, torque])
            all_Y.append([thdd1, thdd2])

            # Step forward
            cos1, sin1, cos2, sin2, dth1, dth2 = semi_implicit_euler(
                cos1, sin1, cos2, sin2, dth1, dth2, torque)

            th1 = np.arctan2(sin1, cos1)
            th2 = np.arctan2(sin2, cos2)

            # Stop if velocities blow up (unstable)
            if abs(dth1) > 10.0 or abs(dth2) > 15.0:
                break

    return np.array(all_X, dtype=np.float32), np.array(all_Y, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-random', type=int, default=80000)
    parser.add_argument('--n-trajectories', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save', type=str, default='saved_data/acrobot_lag_data.pt')
    args = parser.parse_args()

    t0 = time.time()

    # 1. Random state+torque pairs
    print(f"Generating {args.n_random} random samples...")
    Xr, Yr = generate_random(args.n_random, seed=args.seed)
    print(f"  θ̈₁ range: [{Yr[:,0].min():.1f}, {Yr[:,0].max():.1f}]")
    print(f"  θ̈₂ range: [{Yr[:,1].min():.1f}, {Yr[:,1].max():.1f}]")

    # 2. Trajectories for dynamical coverage
    print(f"Generating {args.n_trajectories} trajectory episodes...")
    Xt, Yt = generate_trajectories(args.n_trajectories, seed=args.seed + 1)
    print(f"  {len(Xt)} transitions (avg {len(Xt)/args.n_trajectories:.0f} steps/ep)")

    # 3. Combine and shuffle
    X = np.concatenate([Xr, Xt], axis=0)
    Y = np.concatenate([Yr, Yt], axis=0)
    perm = np.random.RandomState(args.seed).permutation(len(X))
    X, Y = X[perm], Y[perm]

    Xtensor = torch.tensor(X, dtype=torch.float32)
    Ytensor = torch.tensor(Y, dtype=torch.float32)

    n_total = len(Xtensor)
    n_train = int(n_total * 0.8)
    train = (Xtensor[:n_train], Ytensor[:n_train])
    val = (Xtensor[n_train:], Ytensor[n_train:])

    torch.save({'train': train, 'val': val, 'n_total': n_total}, args.save)
    print(f"\nSaved to {args.save}: {n_total} samples "
          f"({n_train} train, {n_total-n_train} val) "
          f"in {time.time()-t0:.1f}s")

    print(f"\nAcceleration stats:")
    print(f"  θ̈₁: mean={Ytensor[:,0].mean():.2f}, std={Ytensor[:,0].std():.1f}")
    print(f"  θ̈₂: mean={Ytensor[:,1].mean():.2f}, std={Ytensor[:,1].std():.1f}")


if __name__ == '__main__':
    main()
