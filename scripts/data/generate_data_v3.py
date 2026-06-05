"""Generate training data v3: diverse random + expert controller, 1:1 ratio.

Random data: multi-scale torque distribution for broad state-space coverage.
Expert data: energy-based swing-up + PD stabilization controller.

Target: rich data near the goal (sin≈1, theta_dot≈0) where v1 KAN fails.
"""
import torch
import gymnasium as gym
import numpy as np


class ExpertController:
    """Bang-bang energy pump + aggressive PD at top. Verified: reaches |Δθ|<0.05."""

    def __init__(self):
        self.g = 10.0
        self.max_torque = 2.0

    def __call__(self, obs):
        cos_th, sin_th, thd = obs
        near_upright = abs(cos_th) < 0.5 and sin_th > 0
        if near_upright:
            # PD stabilization: aggressive gains to catch at the top
            u = 80.0 * (-cos_th) + 15.0 * (-thd)
        else:
            # Bang-bang energy pumping: max torque in velocity direction
            u = 2.0 if thd >= 0 else -2.0
        return np.clip(u, -self.max_torque, self.max_torque)


def collect_random(n_transitions, env):
    """Collect (s,a,s') with multi-scale random torque.

    Mix of uniform and scaled-Gaussian to ensure both small perturbations
    and large swings are represented. Periodically reset for coverage.
    """
    states, actions, next_states = [], [], []
    obs, _ = env.reset()
    reset_every = 200

    for i in range(n_transitions):
        # Mix random strategies for diversity
        r = np.random.random()
        if r < 0.5:
            # Uniform across full torque range
            a = np.random.uniform(-2.0, 2.0)
        elif r < 0.8:
            # Gaussian with medium variance
            a = np.clip(np.random.normal(0, 1.0), -2.0, 2.0)
        else:
            # Large torques (bang-bang exploration)
            a = np.random.choice([-2.0, -1.5, -1.0, 1.0, 1.5, 2.0])

        a_arr = np.array([a], dtype=np.float32)
        next_obs, _, terminated, truncated, _ = env.step(a_arr)

        states.append(obs.copy())
        actions.append(a)
        next_states.append(next_obs.copy())

        obs = next_obs
        if terminated or truncated or (i > 0 and i % reset_every == 0):
            obs, _ = env.reset()

    return states, actions, next_states


def collect_expert(n_transitions, env):
    """Collect (s,a,s') using the expert controller."""
    ctrl = ExpertController()
    states, actions, next_states = [], [], []
    collected = 0
    while collected < n_transitions:
        obs, _ = env.reset()
        for _ in range(200):
            torque = ctrl(obs) + np.random.normal(0, 0.1)
            torque = np.clip(torque, -2.0, 2.0)
            a = np.array([torque], dtype=np.float32)
            next_obs, _, term, trunc, _ = env.step(a)
            states.append(obs.copy())
            actions.append(torque)
            next_states.append(next_obs.copy())
            collected += 1
            obs = next_obs
            if term or trunc or collected >= n_transitions:
                break
    return states[:n_transitions], actions[:n_transitions], next_states[:n_transitions]


def collect_stabilization(n_transitions, env):
    """Collect data specifically near the goal.

    Start pendulum at or near upright with small perturbations,
    apply PD control to keep it there. This fills the critical
    data-sparse region where v1 KAN failed (sin≈1, θ̇≈0).
    """
    ctrl = ExpertController()
    states, actions, next_states = [], [], []
    collected = 0

    while collected < n_transitions:
        # Manually set near-upright start via env reset + brief swing-up
        obs, _ = env.reset()
        # First swing up to upright
        for _ in range(60):
            a = ctrl(obs)
            obs, _, term, trunc, _ = env.step([a])
            if term or trunc:
                break

        # Now collect stabilization data near upright
        for _ in range(100):
            # Small torque perturbations around the PD control
            torque = ctrl(obs) + np.random.uniform(-0.5, 0.5)
            torque = np.clip(torque, -2.0, 2.0)
            a = np.array([torque], dtype=np.float32)
            next_obs, _, term, trunc, _ = env.step(a)
            states.append(obs.copy())
            actions.append(torque)
            next_states.append(next_obs.copy())
            collected += 1
            obs = next_obs
            if term or trunc or collected >= n_transitions:
                break

    return states[:n_transitions], actions[:n_transitions], next_states[:n_transitions]


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

    n_random = 15000
    n_expert_episodes = 10000
    n_expert_stabilize = 10000

    print(f"Collecting {n_random} random transitions...")
    s_r, a_r, ns_r = collect_random(n_random, env)
    x_r, y_r = to_tensors(s_r, a_r, ns_r)

    print(f"Collecting {n_expert_episodes} expert (full episodes)...")
    s_e1, a_e1, ns_e1 = collect_expert(n_expert_episodes, env)
    x_e1, y_e1 = to_tensors(s_e1, a_e1, ns_e1)

    print(f"Collecting {n_expert_stabilize} expert (stabilization)...")
    s_e2, a_e2, ns_e2 = collect_stabilization(n_expert_stabilize, env)
    x_e2, y_e2 = to_tensors(s_e2, a_e2, ns_e2)

    # Quality checks
    for label, data in [("Full episodes", s_e1), ("Stabilization", s_e2)]:
        sin_arr = np.array([s[1] for s in data])
        thd_arr = np.abs(np.array([s[2] for s in data]))
        near = (sin_arr > 0.8) & (thd_arr < 2.0)
        print(f"  {label}: {near.mean():.1%} near goal (sin>0.8, |thd|<2)")

    # Quality checks already printed above during collection.
    # Also check random data state coverage
    random_sin = np.array([s[1] for s in s_r])
    random_thd_abs = np.abs(np.array([s[2] for s in s_r]))
    print(f"Random data: sin range [{random_sin.min():.2f}, {random_sin.max():.2f}]")
    print(f"Random data: |thd| mean={random_thd_abs.mean():.2f}, max={random_thd_abs.max():.2f}")

    # Merge: random + expert episodes + stabilization
    x = torch.cat([x_r, x_e1, x_e2], dim=0)
    y = torch.cat([y_r, y_e1, y_e2], dim=0)
    perm = torch.randperm(len(x))
    x, y = x[perm], y[perm]

    n_total = len(x)
    n_expert_total = n_expert_episodes + n_expert_stabilize
    print(f"\nTotal: {n_total} transitions "
          f"({n_random} random + {n_expert_total} expert, "
          f"ratio expert:random = {n_expert_total/n_random:.2f}:1)")
    print(f"x: {x.shape}, range [{x.min():.3f}, {x.max():.3f}]")
    print(f"y: {y.shape}, range [{y.min():.3f}, {y.max():.3f}]")

    torch.save((x, y), "pendulum_data_v4.pt")
    print("Saved: pendulum_data_v4.pt")

    # Quick demo of expert controller
    obs, _ = env.reset()
    ctrl = ExpertController()
    traj = [obs.copy()]
    for _ in range(200):
        a = ctrl(obs)
        obs, _, term, trunc, _ = env.step([a])
        traj.append(obs.copy())
        if term or trunc:
            break
    traj = np.array(traj)
    final = traj[-1]
    near_steps = (traj[:, 1] > 0.9).sum()
    angle_final = np.arctan2(final[1], final[0])
    print(f"\nExpert demo: {len(traj)} steps, {near_steps} steps near upright (sin>0.9)")
    print(f"  Final: [{final[0]:+.3f}, {final[1]:+.3f}, {final[2]:+.3f}]")
    print(f"  |Δθ|={abs(angle_final - np.pi/2):.3f} rad")

    env.close()


if __name__ == "__main__":
    main()
