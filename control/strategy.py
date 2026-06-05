"""Strategy Layer: deviation metric + strategy mapping for Pendulum-v1.

Model-free, physics-informed. Determines WHAT the system should do next
without any forward prediction through the KAN world model.
"""
import numpy as np


G = 10.0          # gravity (Pendulum-v1)
E_DES = G         # desired energy at upright (sin=1, thetadot=0)
PI_2 = np.pi / 2


def deviation(s, s_goal=None):
    """Compute deviation vector from current state to goal.

    Args:
        s: (3,) numpy array [cosθ, sinθ, θ̇]
    Returns:
        dict with: d_pos (angular error), e_kin (kinetic energy),
                   delta_E (energy deficit), near_upright (bool)
    """
    cos_th, sin_th, thd = s

    # Angular distance from upright
    angle = np.arctan2(sin_th, cos_th)
    d_pos = angle - PI_2
    # Normalize to [-π, π]
    d_pos = (d_pos + np.pi) % (2 * np.pi) - np.pi

    # Kinetic energy
    e_kin = 0.5 * thd * thd

    # Energy deficit: E_des - E_current
    E = e_kin + G * sin_th
    delta_E = E_DES - E

    # Near upright: cos close to 0, sin close to +1 (NOT -1 = bottom!)
    near_upright = abs(cos_th) < 0.5 and sin_th > 0 and abs(thd) < 3.0

    return {
        'd_pos': d_pos, 'e_kin': e_kin, 'delta_E': delta_E,
        'E': E, 'near_upright': near_upright,
        'angle': angle, 'cos': cos_th, 'sin': sin_th, 'thd': thd
    }


def strategy_mode(dev):
    """Map deviation to strategy mode.

    Thresholds tuned for Pendulum-v1 dynamics.
    """
    if not dev['near_upright']:
        if dev['delta_E'] > 2.0:
            return 'swing_up'   # E far below E_des: need energy
        elif dev['delta_E'] < -1.0:
            return 'brake'       # E far above E_des: too much energy, will overshoot
        else:
            return 'swing_up'    # default: pump energy

    # Near upright
    if abs(dev['d_pos']) < 0.15 and dev['e_kin'] < 0.5:
        return 'stabilize'       # close enough, try to hold
    elif dev['e_kin'] > 1.0:
        return 'brake'           # at upright but moving fast
    else:
        return 'stabilize'       # near upright, slow → stabilize


def intermediate_target(s, mode, step_size=0.15):
    """Generate intermediate target s_mid based on strategy mode.

    s_mid encodes a physics-guided *direction* for the Execution Layer.
    Each mode targets the right physical quantity:
      swing_up:  pump |θ̇| in current direction → add energy
      brake:     reduce |θ̇| → remove energy
      stabilize: small corrections toward (0,1,0)

    Key constraint: s_mid must be achievable by f_KAN(s, a) in one step.
    Max |Δθ̇| ≈ 0.1 rad/s from torque (a_max=2, dt=0.05, m=l=1).
    """
    cos_th, sin_th, thd = s
    sign_v = 1.0 if thd >= 0 else -1.0
    delta_v = 0.08  # achievable θ̇ change in one step

    if mode == 'swing_up':
        # PUMP velocity only. Don't force position change — let the
        # pumped velocity drive the natural swing dynamics.
        # Physical justification: d(sin)/dt = cos*θ̇. At the bottom
        # (cos≈-1, sin≈0) we need θ̇<0 to increase sin, but the
        # geometric push toward (0,1) doesn't know this sign.
        # Solution: only ask for |θ̇| increase, trust physics for position.
        thd_mid = thd + sign_v * delta_v
        cos_mid = cos_th
        sin_mid = sin_th

    elif mode == 'brake':
        # BRAKE: reduce |θ̇|. Don't force position — gravity decides
        # where the pendulum goes next.
        if abs(thd) < delta_v:
            thd_mid = 0.0
        else:
            thd_mid = thd - sign_v * delta_v
        cos_mid = cos_th
        sin_mid = sin_th

    elif mode == 'stabilize':
        # STABILIZE: near upright, we CAN target (0,1,0) because the
        # single-step reachable region is centered near here.
        # Use damped velocity target for soft landing.
        step = min(step_size * 0.5, 0.08)
        d_cos = 0.0 - cos_th
        d_sin = 1.0 - sin_th
        dist = np.sqrt(d_cos**2 + d_sin**2)
        if dist > step:
            d_cos = d_cos / dist * step
            d_sin = d_sin / dist * step
        cos_mid = cos_th + d_cos
        sin_mid = sin_th + d_sin
        norm = np.sqrt(cos_mid**2 + sin_mid**2)
        cos_mid /= norm
        sin_mid /= norm
        thd_mid = thd * 0.5

    return np.array([cos_mid, sin_mid, thd_mid], dtype=np.float32)
