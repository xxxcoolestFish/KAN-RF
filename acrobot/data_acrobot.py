"""Generate multi-scale Acrobot training data.  Uses env.step() for correct RK4.

State (6D): [cosθ1, sinθ1, cosθ2, sinθ2, dθ1/6, dθ2/8]
Action one-hot: (3,) for {0, 1, 2}
k ∈ {1, 2, 4, 8}  (dt=0.2s, k=8 = 1.6s lookahead)
"""
import numpy as np, gymnasium as gym, torch, time, sys, os
K_VALS = [1, 2, 4, 8]
N_SAMPLES = 5000


def main():
    torch.manual_seed(42); np.random.seed(42); t0 = time.time()
    env = gym.make('Acrobot-v1')
    env.reset()  # required before env.step() + env.unwrapped.state

    # Sample random physics states (4D) then convert to 6D obs
    n = N_SAMPLES
    theta1 = np.random.uniform(-np.pi, np.pi, n)
    theta2 = np.random.uniform(-np.pi, np.pi, n)
    dtheta1 = np.random.uniform(-6.0, 6.0, n)
    dtheta2 = np.random.uniform(-8.0, 8.0, n)
    actions = np.random.randint(0, 3, n)

    xs, ys = [], []
    for k in K_VALS:
        x_k = np.zeros((n, 6 + 3 + 1))  # state(6) + action(3) + k_norm(1)
        y_k = np.zeros((n, 6))
        for i in range(n):
            env.unwrapped.state = (theta1[i], theta2[i], dtheta1[i], dtheta2[i])
            obs0 = env.unwrapped._get_ob()
            a = int(actions[i])

            for step in range(k):
                obs, _, term, _, _ = env.step(a)
                if term: break

            # Normalize state
            sn = np.array([obs0[0], obs0[1], obs0[2], obs0[3],
                           obs0[4]/6.0, obs0[5]/8.0], dtype=np.float32)
            yn = np.array([obs[0], obs[1], obs[2], obs[3],
                           obs[4]/6.0, obs[5]/8.0], dtype=np.float32)
            a_oh = np.zeros(3, dtype=np.float32); a_oh[a] = 1.0

            x_k[i, :6] = sn
            x_k[i, 6:9] = a_oh
            x_k[i, 9] = k / 8.0
            y_k[i] = yn

        xs.append(torch.tensor(x_k, dtype=torch.float32))
        ys.append(torch.tensor(y_k, dtype=torch.float32))

        # Diagnostic: angle changes
        dcos1 = np.abs(y_k[:, 0] - x_k[:, 0])
        dtheta1 = (np.arctan2(y_k[:, 1], y_k[:, 0]) -
                    np.arctan2(x_k[:, 1], x_k[:, 0]))
        dtheta1 = np.abs((dtheta1 + np.pi) % (2*np.pi) - np.pi)
        print(f'  k={k:2d}: n={n}, |delta_cosθ1| mean={dcos1.mean():.4f} '
              f'max={dcos1.max():.4f}, |delta_angle| max={np.rad2deg(dtheta1.max()):.0f}deg')

    env.close()

    x_ms = torch.cat(xs, dim=0); y_ms = torch.cat(ys, dim=0)
    perm = torch.randperm(len(x_ms)); x_ms, y_ms = x_ms[perm], y_ms[perm]
    torch.save((x_ms, y_ms), 'acrobot_data_ms.pt')
    print(f'\nSaved acrobot_data_ms.pt: {x_ms.shape[0]} samples, '
          f'{x_ms.shape[1]}d -> {y_ms.shape[1]}d, {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
