#!/usr/bin/env python3
"""Verify CartPole physics: analytical dynamics vs env.step(), Lagrangian structure."""
import numpy as np, gymnasium as gym, torch

G = 9.8; MC = 1.0; MP = 0.1; L = 0.5; FM = 10.0; DT = 0.02
TOTAL_MASS = MC + MP
POLE_MASS_LEN = MP * L


def step_analytic(state, action):
    """Analytical CartPole step (from gymnasium source). Vectorized."""
    x, x_dot, theta, theta_dot = (state[:, i] for i in range(4))
    force = np.where(action == 1, FM, -FM)
    costheta = np.cos(theta)
    sintheta = np.sin(theta)
    temp = (force + POLE_MASS_LEN * theta_dot**2 * sintheta) / TOTAL_MASS
    theta_acc = (G * sintheta - costheta * temp) / \
                (L * (4.0/3.0 - MP * costheta**2 / TOTAL_MASS))
    x_acc = temp - POLE_MASS_LEN * theta_acc * costheta / TOTAL_MASS
    return np.stack([
        x + x_dot * DT,
        x_dot + x_acc * DT,
        theta + theta_dot * DT,
        theta_dot + theta_acc * DT,
    ], axis=-1), np.stack([x_acc, theta_acc], axis=-1)


def compute_accel_derived(costh, sinth, thd, force):
    """Compute [x_ddot, th_ddot] from derived Lagrangian-based formulas.

    Lagrangian: L = ½(M+m)ẋ² + ½(4/3·mL²)θ̇² + mLẋθ̇cosθ + mgLcosθ
    (Note: 4/3·mL² is pole's moment of inertia about pivot: I_com + mL² = mL²/3 + mL²)

    Mass matrix: M = [[M+m, mLcosθ], [mLcosθ, 4/3·mL²]]
    EL: M(q)q̈ + C(q,q̇)q̇ = [F, 0]ᵀ + [mLθ̇²sinθ, mgLsinθ]ᵀ
    """
    a = MC + MP                    # M+m
    b = MP * L * costh              # mLcosθ
    d_theta = (4.0/3.0) * MP * L**2  # 4/3·mL² (pole inertia about pivot)
    det = a * d_theta - b**2

    # RHS: [F + mLθ̇²sinθ, mgLsinθ]
    rhs0 = force + MP * L * thd**2 * sinth
    rhs1 = MP * G * L * sinth       # mgLsinθ (gravity torque)

    x_ddot = (d_theta * rhs0 - b * rhs1) / det
    th_ddot = (-b * rhs0 + a * rhs1) / det
    return x_ddot, th_ddot


def main():
    torch.manual_seed(42); np.random.seed(42)
    env = gym.make('CartPole-v1')
    env.reset()  # required before env.step() / env.unwrapped.state setter

    # --- Test 1: analytical step vs env.step() ---
    print("=== Test 1: Analytical step vs env.step() ===")
    n_test = 1000
    state = np.zeros((n_test, 4))
    state[:, 0] = np.random.uniform(-2.0, 2.0, n_test)
    state[:, 1] = np.random.uniform(-2.0, 2.0, n_test)
    state[:, 2] = np.random.uniform(-0.2, 0.2, n_test)
    state[:, 3] = np.random.uniform(-2.0, 2.0, n_test)
    actions = np.random.randint(0, 2, n_test)

    # env.step()
    env_states = np.zeros((n_test, 4))
    env_acc = np.zeros((n_test, 2))
    for i in range(n_test):
        env.unwrapped.state = (state[i, 0], state[i, 1], state[i, 2], state[i, 3])
        obs, _, _, _, _ = env.step(int(actions[i]))
        env_states[i] = obs.copy()
        env_acc[i, 0] = (obs[1] - state[i, 1]) / DT  # x_ddot from FD
        env_acc[i, 1] = (obs[3] - state[i, 3]) / DT  # th_ddot from FD

    # Analytical
    ana_states, ana_acc = step_analytic(state, actions)

    # Derived Lagrangian
    costh = np.cos(state[:, 2])
    sinth = np.sin(state[:, 2])
    force = np.where(actions == 1, FM, -FM)
    x_dd, th_dd = compute_accel_derived(costh, sinth, state[:, 3], force)
    derived_acc = np.stack([x_dd, th_dd], axis=-1)

    # Compare
    state_err = np.abs(env_states - ana_states)
    acc_ana_err = np.abs(env_acc - ana_acc)
    acc_derived_err = np.abs(env_acc - derived_acc)

    print(f"  State |env - analytic|: max={state_err.max():.2e}, mean={state_err.mean():.2e}")
    print(f"  Accel |env - analytic|: max={acc_ana_err.max():.2e}, mean={acc_ana_err.mean():.2e}")
    print(f"  Accel |env - lagrangian|: max={acc_derived_err.max():.2e}, mean={acc_derived_err.mean():.2e}")
    print(f"  x_ddot range: env=[{env_acc[:,0].min():.1f}, {env_acc[:,0].max():.1f}], "
          f"lagr=[{x_dd.min():.1f}, {x_dd.max():.1f}]")
    print(f"  th_ddot range: env=[{env_acc[:,1].min():.1f}, {env_acc[:,1].max():.1f}], "
          f"lagr=[{th_dd.min():.1f}, {th_dd.max():.1f}]")

    # --- Test 2: Lagrangian mass matrix check ---
    print("\n=== Test 2: Mass matrix properties ===")
    M_true = np.array([[MC+MP, MP*L], [MP*L, MP*L**2]])  # at θ=0 (upright)
    d_th = (4.0/3.0) * MP * L**2  # 4/3·mL²
    print(f"  M(θ=0) = [[{M_true[0,0]:.3f}, {M_true[0,1]:.3f}], [{M_true[1,0]:.3f}, {M_true[1,1]:.3f}]]")
    eigvals = np.linalg.eigvalsh(M_true)
    print(f"  Eigenvalues: {eigvals} (min={eigvals.min():.4f}, cond={eigvals.max()/eigvals.min():.1f})")
    print(f"  Correct M_θθ = 4/3·mL² = {d_th:.4f} (was: mL² = {MP*L**2:.4f})")

    # At θ=π/2 (horizontal)
    costh_pi2, sinth_pi2 = 0.0, 1.0
    M_pi2 = np.array([[MC+MP, MP*L*costh_pi2], [MP*L*costh_pi2, d_th]])
    print(f"  M(θ=π/2) = [[{M_pi2[0,0]:.3f}, {M_pi2[0,1]:.3f}], [{M_pi2[1,0]:.3f}, {M_pi2[1,1]:.3f}]]")

    # --- Test 3: velocity clipping check ---
    print("\n=== Test 3: Velocity clipping ===")
    state[:, 1] = np.random.uniform(-5.0, 5.0, n_test)  # wider velocity range
    state[:, 3] = np.random.uniform(-6.0, 6.0, n_test)
    env_states2 = np.zeros((n_test, 4))
    for i in range(n_test):
        env.unwrapped.state = (state[i, 0], state[i, 1], state[i, 2], state[i, 3])
        obs, _, _, _, _ = env.step(int(actions[i]))
        env_states2[i] = obs.copy()

    # CartPole doesn't clip velocities like Pendulum! It terminates instead.
    # Check if velocities are preserved in normal range
    vel_diff = np.abs(env_states2[:, 1] - state[:, 1])  # |ẋ_next - ẋ|
    ang_vel_diff = np.abs(env_states2[:, 3] - state[:, 3])  # |θ̇_next - θ̇|
    # Acceleration should be bounded: ẍ ≤ FM/TOTAL_MASS ≈ 9.1, θ̈ ≤ ~30 rad/s²
    max_x_acc = FM / TOTAL_MASS * DT  # ≈ 0.182
    max_th_acc = 30 * DT  # ≈ 0.6
    x_clipped = (vel_diff > max_x_acc * 1.1)
    th_clipped = (ang_vel_diff > max_th_acc * 1.1)
    print(f"  x clipped: {x_clipped.sum()}/{n_test}, th clipped: {th_clipped.sum()}/{n_test}")
    print(f"  |Δẋ| range: [{vel_diff.min():.4f}, {vel_diff.max():.4f}]")
    print(f"  |Δθ̇| range: [{ang_vel_diff.min():.4f}, {ang_vel_diff.max():.4f}]")

    # --- Test 4: Energy conservation check ---
    print("\n=== Test 4: Energy conservation ===")
    # E = ½(M+m)ẋ² + ½(4/3·mL²)θ̇² + mLẋθ̇cosθ + mgLcosθ
    # (with correct rotational inertia)
    d_th_ene = (4.0/3.0) * MP * L**2
    for i in range(5):
        x0, xd0, th0, thd0 = state[i]
        c0 = np.cos(th0)
        E_before = (0.5*(MC+MP)*xd0**2 + 0.5*d_th_ene*thd0**2 +
                     MP*L*xd0*thd0*c0 + MP*G*L*c0)

        # Use analytical step for clean comparison
        sa, acc = step_analytic(state[i:i+1], np.array([actions[i]]))
        x1, xd1, th1, thd1 = sa[0]
        c1 = np.cos(th1)
        E_after = (0.5*(MC+MP)*xd1**2 + 0.5*d_th_ene*thd1**2 +
                    MP*L*xd1*thd1*c1 + MP*G*L*c1)
        force_i = FM if actions[i] == 1 else -FM
        work = force_i * (x1 - x0)
        print(f"  Sample {i}: E_before={E_before:.3f}, E_after={E_after:.3f}, "
              f"ΔE={E_after-E_before:.3f}, work={work:.3f}, "
              f"conservation_err={abs(E_after-E_before-work):.4f}")

    env.close()
    print("\n✅ Diagnostics complete.")


if __name__ == '__main__':
    main()
