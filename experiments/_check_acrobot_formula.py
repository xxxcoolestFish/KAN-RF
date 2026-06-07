#!/usr/bin/env python3
"""Diagnostic: verify Acrobot Lagrangian against env.step().

Acrobot-v1 parameters (from gymnasium):
  LINK_LENGTH_1 = 1.0, LINK_LENGTH_2 = 1.0
  LINK_MASS_1 = 1.0, LINK_MASS_2 = 1.0
  LINK_COM_POS_1 = 0.5, LINK_COM_POS_2 = 0.5
  LINK_MOI = 1.0 (for both links)
  dt = 0.2
  g ≈ 9.8 (check)

State: [cosθ₁, sinθ₁, cosθ₂, sinθ₂, θ̇₁, θ̇₂]  (6D)
Action: {0, 1, 2} → torque on joint 2

The Acrobot is a 2-link pendulum:
  - Joint 1: unactuated (no torque)
  - Joint 2: actuated (torque ∈ {−1, 0, +1} × torque_multiplier?)
  - θ₁, θ₂ measured from vertically-down

Lagrangian derivation:
  Position of COM1: (lc₁·sinθ₁, −lc₁·cosθ₁)
  Position of COM2: (l₁·sinθ₁ + lc₂·sin(θ₁+θ₂), −l₁·cosθ₁ − lc₂·cos(θ₁+θ₂))

  Mass matrix:
    M₁₁ = I₁ + m₁lc₁² + m₂(l₁² + lc₂² + 2l₁lc₂cosθ₂) + I₂
    M₂₂ = m₂lc₂² + I₂
    M₁₂ = M₂₁ = m₂(lc₂² + l₁lc₂cosθ₂) + I₂
"""
import numpy as np, gymnasium as gym, torch

# Acrobot parameters
l1 = 1.0; l2 = 1.0
lc1 = 0.5; lc2 = 0.5
m1 = 1.0; m2 = 1.0
I1 = 1.0; I2 = 1.0
g = 9.8
dt = 0.2


def mass_matrix(cos2, sin2=None):
    """Acrobot mass matrix M(θ₂)."""
    M11 = I1 + m1*lc1**2 + m2*(l1**2 + lc2**2 + 2*l1*lc2*cos2) + I2
    M12 = m2*(lc2**2 + l1*lc2*cos2) + I2
    M22 = m2*lc2**2 + I2
    return np.array([[M11, M12], [M12, M22]])


def coriolis_centripetal(cos2, sin2, thd1, thd2):
    """C(q,q̇)q̇ from the mass matrix derivatives.

    Since M depends only on θ₂:
      Ṁ = ∂M/∂θ₂ · θ̇₂

    C(q,q̇)q̇ = Ṁq̇ − ½ ∂/∂q (q̇ᵀMq̇)
    where ∂/∂q(q̇ᵀMq̇) = [0, q̇ᵀ(∂M/∂θ₂)q̇]ᵀ
    """
    dM11 = -2*m2*l1*lc2*sin2  # ∂M₁₁/∂θ₂
    dM12 = -m2*l1*lc2*sin2     # ∂M₁₂/∂θ₂
    dM22 = 0.0                  # ∂M₂₂/∂θ₂

    # Ṁ = θ̇₂ * dM/dθ₂
    # Ṁq̇:
    Md_qd_1 = thd2 * (dM11 * thd1 + dM12 * thd2)
    Md_qd_2 = thd2 * (dM12 * thd1 + dM22 * thd2)

    # ∂/∂q(q̇ᵀMq̇) = [0, q̇ᵀ∂M/∂θ₂ q̇]
    quad = dM11*thd1**2 + 2*dM12*thd1*thd2 + dM22*thd2**2

    C1 = Md_qd_1 - 0.5 * 0.0       # ∂/∂θ₁ = 0
    C2 = Md_qd_2 - 0.5 * quad

    return np.array([C1, C2])


def gravity_torque(cos1, sin1, cos12, sin12):
    """Gravity term g(q) = ∇U(q)."""
    # U = −(m₁lc₁+m₂l₁)g·cosθ₁ − m₂lc₂g·cos(θ₁+θ₂)
    # ∂U/∂θ₁ = (m₁lc₁+m₂l₁)g·sinθ₁ + m₂lc₂g·sin(θ₁+θ₂)
    # ∂U/∂θ₂ = m₂lc₂g·sin(θ₁+θ₂)
    g1 = (m1*lc1 + m2*l1) * g * sin1 + m2*lc2 * g * sin12
    g2 = m2*lc2 * g * sin12
    return np.array([g1, g2])


def lagrangian_accel(cos1, sin1, cos2, sin2, thd1, thd2, torque):
    """Compute accelerations from Lagrangian dynamics.

    M(q)q̈ + C(q,q̇)q̇ + g(q) = B·τ
    where B = [0, 1]ᵀ (torque only on joint 2)
    → q̈ = M⁻¹([0, τ]ᵀ − C − g)
    """
    M = mass_matrix(cos2)
    C = coriolis_centripetal(cos2, sin2, thd1, thd2)
    g_vec = gravity_torque(cos1, sin1, cos2, sin2)

    rhs = np.array([0.0, torque]) - C - g_vec
    qdd = np.linalg.solve(M, rhs)
    return qdd  # [θ̈₁, θ̈₂]


def env_step_accel(env, theta1, theta2, dtheta1, dtheta2, action):
    """Get accelerations from env.step() via finite differences."""
    env.unwrapped.state = (theta1, theta2, dtheta1, dtheta2)
    obs0 = env.unwrapped._get_ob()
    obs, _, term, _, _ = env.step(int(action))
    if term:
        return None, None
    # Store next state's velocities from env (env.state is set after step)
    dtheta1_next, dtheta2_next = env.unwrapped.state[2], env.unwrapped.state[3]
    thdd1 = (dtheta1_next - dtheta1) / dt
    thdd2 = (dtheta2_next - dtheta2) / dt
    return thdd1, thdd2


def main():
    env = gym.make('Acrobot-v1')
    env.reset()

    # Check torque
    print("=== Acrobot physical check ===")
    # What's the torque multiplier?
    for a in [0, 1, 2]:
        env.unwrapped.state = (0.0, 0.0, 0.0, 0.0)
        obs_before = env.unwrapped._get_ob()
        env.step(a)
        dth1, dth2 = env.unwrapped.state[2], env.unwrapped.state[3]
        print(f"  Action {a}: dθ₁_next={dth1:.4f}, dθ₂_next={dth2:.4f}")

    # The torque values seem to be: the Acrobot uses AVAIL_TORQUE = [-1.0, 0.0, 1.0]
    # Let me check
    print(f"  AVAIL_TORQUE: {env.unwrapped.AVAIL_TORQUE}")
    torque_vals = env.unwrapped.AVAIL_TORQUE

    # --- Test Lagrangian vs env.step() ---
    print("\n=== Lagrangian accel vs env.step() ===")
    np.random.seed(42)
    n_test = 500
    errors = []

    for i in range(n_test):
        th1 = np.random.uniform(-np.pi, np.pi)
        th2 = np.random.uniform(-np.pi, np.pi)
        dth1 = np.random.uniform(-4.0, 4.0)
        dth2 = np.random.uniform(-6.0, 6.0)
        a = np.random.randint(0, 3)

        cos1, sin1 = np.cos(th1), np.sin(th1)
        cos2, sin2 = np.cos(th2), np.sin(th2)
        torque = torque_vals[a]

        # env.step()
        thdd1_env, thdd2_env = env_step_accel(env, th1, th2, dth1, dth2, a)
        if thdd1_env is None:
            continue

        # Lagrangian
        qdd = lagrangian_accel(cos1, sin1, cos2, sin2, dth1, dth2, torque)
        err = np.abs(qdd - np.array([thdd1_env, thdd2_env]))
        errors.append(err)

    errors = np.array(errors)
    print(f"  n_valid: {len(errors)}/{n_test}")
    print(f"  |θ̈₁ - env|: mean={errors[:,0].mean():.4f}, max={errors[:,0].max():.4f}")
    print(f"  |θ̈₂ - env|: mean={errors[:,1].mean():.4f}, max={errors[:,1].max():.4f}")
    print(f"  Overall RMSE: {np.sqrt((errors**2).mean()):.4f}")

    # --- Check mass matrix ---
    print("\n=== Mass matrix check ===")
    for th2_deg in [0, 90, 180]:
        th2 = np.deg2rad(th2_deg)
        M = mass_matrix(np.cos(th2))
        print(f"  θ₂={th2_deg}°: M = [[{M[0,0]:.2f}, {M[0,1]:.2f}], [{M[1,0]:.2f}, {M[1,1]:.2f}]]")
        eig = np.linalg.eigvalsh(M)
        print(f"           cond={eig.max()/eig.min():.1f}")

    # --- Check energy conservation ---
    print("\n=== Energy conservation ===")
    for i in range(5):
        th1 = np.random.uniform(-1.0, 1.0)
        th2 = np.random.uniform(-1.0, 1.0)
        dth1 = np.random.uniform(-2.0, 2.0)
        dth2 = np.random.uniform(-2.0, 2.0)

        cos1, sin1 = np.cos(th1), np.sin(th1)
        cos2, sin2 = np.cos(th2), np.sin(th2)
        M = mass_matrix(cos2)
        qd = np.array([dth1, dth2])

        # Kinetic energy
        T = 0.5 * qd @ M @ qd
        # Potential energy
        U = -(m1*lc1 + m2*l1)*g*cos1 - m2*lc2*g*np.cos(th1+th2)
        E_before = T + U

        # Step forward
        a = np.random.randint(0, 3)
        thdd1, thdd2 = env_step_accel(env, th1, th2, dth1, dth2, a)
        if thdd1 is None:
            continue

        dth1_n = dth1 + dt*thdd1
        dth2_n = dth2 + dt*thdd2
        th1_n = th1 + dt*dth1_n
        th2_n = th2 + dt*dth2_n

        M_n = mass_matrix(np.cos(th2_n))
        qd_n = np.array([dth1_n, dth2_n])
        T_n = 0.5 * qd_n @ M_n @ qd_n
        U_n = -(m1*lc1 + m2*l1)*g*np.cos(th1_n) - m2*lc2*g*np.cos(th1_n+th2_n)
        E_after = T_n + U_n

        work = torque_vals[a] * (th2_n - th2)  # torque × angular displacement
        print(f"  Sample {i}: ΔE={E_after-E_before:.4f}, work={work:.4f}, "
              f"err={abs(E_after-E_before-work):.4f}")

    env.close()
    print("\n✅ Acrobot Lagrangian diagnostic complete.")


if __name__ == '__main__':
    main()
