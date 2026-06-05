"""MountainCar multi-scale data via analytical dynamics.

State (2D): [position, velocity] normalized to [-1, 1]
  position /= 0.6, velocity /= 0.07
Action one-hot: (3,) for {0, 1, 2}
k ∈ {1, 2, 4}  (discrete-time env, no dt)

Dynamics:
  v' = v + (a-1)*0.001 + cos(3*pos)*(-0.0025)
  v' = clip(v', -0.07, 0.07)
  p' = p + v'
  p' = clip(p', -1.2, 0.6)
"""
import torch, numpy as np, time, sys, os
K_VALS = [1, 2, 4]
N_SAMPLES = 10000
FORCE = 0.001
GRAVITY = 0.0025
MAX_SPEED = 0.07


def step_mountaincar(state, action):
    """Single step.  state: (N, 2) [pos, vel].  action: (N,) {0,1,2}."""
    pos, vel = state[:, 0], state[:, 1]
    vel = vel + (action.float() - 1) * FORCE + np.cos(3 * pos.numpy()) * (-GRAVITY)
    vel = torch.tensor(vel).clamp(-MAX_SPEED, MAX_SPEED)
    pos = (pos + vel).clamp(-1.2, 0.6)
    return torch.stack([pos, vel.to(torch.float32)], dim=-1)


def main():
    torch.manual_seed(42); np.random.seed(42); t0 = time.time()

    n = N_SAMPLES
    pos = torch.rand(n) * 1.8 - 1.2       # [-1.2, 0.6]
    vel = torch.rand(n) * 0.14 - 0.07     # [-0.07, 0.07]
    actions = torch.randint(0, 3, (n,))

    xs, ys = [], []
    for k in K_VALS:
        a_oh = torch.zeros(n, 3)
        a_oh[torch.arange(n), actions] = 1.0
        kn = torch.full((n, 1), k / 4.0)

        p_norm = pos / 0.6
        v_norm = vel / 0.07

        # Simulate k steps
        state = torch.stack([pos, vel], dim=-1)
        for _ in range(k):
            state = step_mountaincar(state, actions)

        y_norm = torch.stack([state[:, 0] / 0.6, state[:, 1] / 0.07], dim=-1)
        x_k = torch.cat([p_norm.unsqueeze(1), v_norm.unsqueeze(1), a_oh, kn], dim=-1)
        xs.append(x_k.float())
        ys.append(y_norm.float())

        # Diagnostic
        dpos = (state[:, 0] - pos).abs()
        print(f'  k={k:2d}: n={n}, |delta_pos| mean={dpos.mean():.4f} max={dpos.max():.4f}')

    x_ms = torch.cat(xs, dim=0); y_ms = torch.cat(ys, dim=0)
    perm = torch.randperm(len(x_ms)); x_ms, y_ms = x_ms[perm], y_ms[perm]
    out_dir = os.path.dirname(os.path.abspath(__file__))
    torch.save((x_ms, y_ms), os.path.join(out_dir, 'mountaincar_data_ms.pt'))
    print(f'\nSaved mountaincar_data_ms.pt: {x_ms.shape[0]} samples, '
          f'{x_ms.shape[1]}d -> {y_ms.shape[1]}d, {time.time()-t0:.1f}s')


if __name__ == '__main__':
    main()
