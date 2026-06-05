"""Strategy Layer v2: energy-gap-based deviation + velocity field guidance.

Replaces the geometric intermediate_target with a physics-grounded gap:
  - gap = Delta_E (scalar energy deficit) + grad_E (energy gradient direction)
  - desired velocity field: v_des = kappa * Delta_E * grad_E

Model-free. Generalizes to any system with an energy-like scalar function.
"""
import numpy as np

G = 10.0
E_DES = G
PI_2 = np.pi / 2


def compute_gap(s):
    """Compute multi-dimensional gap between s and goal s* = [0,1,0].

    Returns dict with:
      delta_E: scalar energy deficit (E_des - E). >0 means need energy.
      grad_E:  (3,) energy gradient [dE/dcos, dE/dsin, dE/dthd]
      near_upright: bool, whether pendulum is near the top
      d_pos: angular deviation from upright
      E, e_kin: energy and kinetic energy (diagnostic)
    """
    cos_th, sin_th, thd = s

    E = 0.5 * thd * thd + G * sin_th
    delta_E = E_DES - E

    # Gradient of E w.r.t. state [cos, sin, thd]
    # E = 0.5*thd^2 + G*sin, so:
    #   dE/dcos = 0,  dE/dsin = G,  dE/dthd = thd
    grad_E = np.array([0.0, G, thd], dtype=np.float32)

    angle = np.arctan2(sin_th, cos_th)
    d_pos = angle - PI_2
    d_pos = (d_pos + np.pi) % (2 * np.pi) - np.pi

    e_kin = 0.5 * thd * thd
    near_upright = abs(cos_th) < 0.5 and sin_th > 0 and abs(thd) < 3.0

    return {
        'delta_E': delta_E,
        'grad_E': grad_E,
        'E': E,
        'e_kin': e_kin,
        'd_pos': d_pos,
        'near_upright': near_upright,
        'angle': angle,
        'cos': cos_th,
        'sin': sin_th,
        'thd': thd,
    }


def desired_velocity(gap, mode=None):
    """Compute desired velocity field v_des on state space.

    v_des(s) = kappa * delta_E * grad_E(s)

    This is the direction and rate at which the state should change.
    For swing-up: v_des primarily drives theta_dot in the current direction.
    For stabilize: v_des points toward (0,1,0) with damping.
    """
    delta_E = gap['delta_E']
    grad_E = gap['grad_E']

    if mode == 'stabilize' or (gap['near_upright'] and abs(gap['d_pos']) < 0.3):
        # Near upright: target (0,1,0) with damping
        cos_th, sin_th, thd = gap['cos'], gap['sin'], gap['thd']
        v_des = np.array([
            (0.0 - cos_th) * 2.0,     # drive cos → 0
            (1.0 - sin_th) * 2.0,     # drive sin → 1
            -thd * 1.5,               # damp velocity
        ], dtype=np.float32)
        # Light clamp to prevent numerical blowup, not to restrict control
        max_v = 0.5
        norm = np.linalg.norm(v_des)
        if norm > max_v:
            v_des *= max_v / norm
    else:
        # Swing-up / brake: energy-guided velocity field.
        # v_des = kappa * delta_E * grad_E
        # No clamp — let the physics determine magnitude. When delta_E is
        # large, v_des asks for a large change; the optimizer will hit the
        # physical limit (max torque) and the remaining gap carries over
        # to the next step.
        kappa = 0.005
        v_des = kappa * delta_E * grad_E

    return v_des.astype(np.float32)


def strategy_mode(gap):
    """Determine control mode from gap. (Same logic as v1, kept for diagnostics.)"""
    if not gap['near_upright']:
        if gap['delta_E'] > 2.0:
            return 'swing_up'
        elif gap['delta_E'] < -1.0:
            return 'brake'
        else:
            return 'swing_up'
    if abs(gap['d_pos']) < 0.15 and gap['e_kin'] < 0.5:
        return 'stabilize'
    elif gap['e_kin'] > 1.0:
        return 'brake'
    else:
        return 'stabilize'
