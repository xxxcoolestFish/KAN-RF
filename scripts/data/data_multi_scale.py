"""Generate multi-scale training data for the KAN world model.

For each (s, a) in pendulum_data_v4.pt, simulate forward k steps
(k ∈ {1, 2, 4, 8, 16}) with constant torque through analytical dynamics.

Output: pendulum_data_ms.pt — (x_ms, y) where
  x_ms: (N*5, 5)  = [cos, sin, thd/8, a/2, k/16]
  y:    (N*5, 3)  = [cos', sin', thd'/8]  (k steps later)
"""
import torch, time

K_VALUES = [1, 2, 4, 8, 16]
DT = 0.05
G = 10.0
MAX_SPEED = 8.0


def simulate_k_steps(s_norm, a_norm, k):
    """Simulate pendulum k steps with constant torque.  Fully vectorized.

    Args:
        s_norm: (N, 3) normalized state [cos, sin, thd/8]
        a_norm: (N,)  normalized action torque/2
        k:      int, number of timesteps to simulate

    Returns:
        s_next_norm: (N, 3) normalized state after k steps
    """
    thd = s_norm[:, 2] * 8.0             # denormalize theta_dot
    theta = torch.atan2(s_norm[:, 1], s_norm[:, 0])
    torque = a_norm * 2.0                 # denormalize to [-2, 2]

    for _ in range(k):
        thd = thd + (15.0 * torch.sin(theta) + 3.0 * torque) * DT
        thd = thd.clamp(-MAX_SPEED, MAX_SPEED)
        theta = theta + thd * DT

    return torch.stack([torch.cos(theta), torch.sin(theta), thd / 8.0], dim=-1)


def main():
    t0 = time.time()

    # Load source data — use all 35k (s,a) pairs
    x_src, y_src = torch.load("pendulum_data_v4.pt", weights_only=True)
    s_norm = x_src[:, :3]                 # (N, 3)
    a_norm = x_src[:, 3]                  # (N,)

    xs, ys = [], []
    for k in K_VALUES:
        y_k = simulate_k_steps(s_norm, a_norm, k)
        k_norm = torch.full((len(s_norm), 1), k / 16.0)
        x_k = torch.cat([s_norm, a_norm.unsqueeze(1), k_norm], dim=-1)  # (N, 5)
        xs.append(x_k)
        ys.append(y_k)
        print(f"  k={k:2d}: {len(x_k)} samples, y angle range "
              f"[{torch.atan2(y_k[:,1], y_k[:,0]).min():.2f}, "
              f"{torch.atan2(y_k[:,1], y_k[:,0]).max():.2f}] rad")

    x_ms = torch.cat(xs, dim=0)           # (N*5, 5)
    y_ms = torch.cat(ys, dim=0)           # (N*5, 3)

    # Shuffle to mix k values within batches
    perm = torch.randperm(len(x_ms))
    x_ms, y_ms = x_ms[perm], y_ms[perm]

    torch.save((x_ms, y_ms), "pendulum_data_ms.pt")
    print(f"\nSaved pendulum_data_ms.pt: {x_ms.shape[0]} samples, "
          f"{x_ms.shape[1]}d → {y_ms.shape[1]}d")
    print(f"Time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
