"""Oracle MPC for Acrobot: use the real simulator as WM.

If this fails, the MPC approach itself is insufficient.
If this succeeds, our WM needs more accuracy in multi-step rollouts.
"""
import torch, numpy as np, time, gymnasium as gym

MAX_V1, MAX_V2 = 6.0, 8.0


def tip_height(s_norm):
    """s_norm: [cosθ1, sinθ1, cosθ2, sinθ2, vel1_norm, vel2_norm]"""
    cos_th1, sin_th1 = s_norm[0], s_norm[1]
    cos_th2, sin_th2 = s_norm[2], s_norm[3]
    cos_th12 = cos_th1 * cos_th2 - sin_th1 * sin_th2
    return cos_th1 + cos_th12


def oracle_step(s_norm, action):
    """Use the REAL simulator to predict next normalized state."""
    cos_th1, sin_th1, cos_th2, sin_th2 = s_norm[0:4]
    theta1 = np.arctan2(sin_th1, cos_th1)
    theta2 = np.arctan2(sin_th2, cos_th2)
    dtheta1 = s_norm[4] * MAX_V1
    dtheta2 = s_norm[5] * MAX_V2

    # Need a separate env for oracle — create once, reuse
    pass  # handled differently


def evaluate_oracle_mpc(method='1-step', depth=3, horizon=6, n_samples=200,
                        cost='height', n_trials=10):
    """MPC using the actual simulator as the perfect WM."""
    env = gym.make('Acrobot-v1')
    successes = 0
    all_steps = []
    t0 = time.time()

    for trial in range(n_trials):
        seed = 42 + trial * 100
        obs, _ = env.reset(seed=seed)
        ok = False

        for step in range(500):
            s_n = np.array([obs[0], obs[1], obs[2], obs[3],
                           obs[4]/MAX_V1, obs[5]/MAX_V2], dtype=np.float32)

            if method == '1-step':
                # Try each action, use real sim
                best_action = 0
                best_cost = float('inf')
                for a in range(3):
                    # Save state
                    theta1, theta2, dth1, dth2 = env.unwrapped.state
                    obs_a, _, _, _, _ = env.step(a)
                    sn_a = np.array([obs_a[0], obs_a[1], obs_a[2], obs_a[3],
                                    obs_a[4]/MAX_V1, obs_a[5]/MAX_V2], dtype=np.float32)
                    # Restore state
                    env.unwrapped.state = (theta1, theta2, dth1, dth2)
                    c = -tip_height(sn_a) if cost == 'height' else -tip_height(sn_a)
                    if c < best_cost:
                        best_cost = c
                        best_action = a
                action = best_action

            elif method == 'shooting':
                best_action = 0
                best_cost = float('inf')
                for _ in range(n_samples):
                    seq = np.random.randint(0, 3, horizon)
                    # Rollout using real sim
                    theta1, theta2, dth1, dth2 = env.unwrapped.state
                    s_cur = np.array([np.cos(theta1), np.sin(theta1),
                                     np.cos(theta2), np.sin(theta2),
                                     dth1/MAX_V1, dth2/MAX_V2], dtype=np.float32)
                    for a in seq:
                        env.unwrapped.state = (
                            np.arctan2(s_cur[1], s_cur[0]),
                            np.arctan2(s_cur[3], s_cur[2]),
                            s_cur[4] * MAX_V1,
                            s_cur[5] * MAX_V2)
                        obs_r, _, _, _, _ = env.step(a)
                        s_cur = np.array([obs_r[0], obs_r[1], obs_r[2], obs_r[3],
                                         obs_r[4]/MAX_V1, obs_r[5]/MAX_V2], dtype=np.float32)
                    # Restore
                    env.unwrapped.state = (theta1, theta2, dth1, dth2)
                    c = -tip_height(s_cur)
                    if c < best_cost:
                        best_cost = c
                        best_action = seq[0]
                action = best_action

            elif method == 'exhaustive':
                best_action = 0
                best_cost = float('inf')
                n_seqs = 3 ** depth
                for seq_idx in range(n_seqs):
                    seq = []
                    tmp = seq_idx
                    for _ in range(depth):
                        seq.append(tmp % 3)
                        tmp //= 3
                    theta1, theta2, dth1, dth2 = env.unwrapped.state
                    s_cur = np.array([np.cos(theta1), np.sin(theta1),
                                     np.cos(theta2), np.sin(theta2),
                                     dth1/MAX_V1, dth2/MAX_V2], dtype=np.float32)
                    for a in seq:
                        env.unwrapped.state = (
                            np.arctan2(s_cur[1], s_cur[0]),
                            np.arctan2(s_cur[3], s_cur[2]),
                            s_cur[4] * MAX_V1,
                            s_cur[5] * MAX_V2)
                        obs_r, _, _, _, _ = env.step(a)
                        s_cur = np.array([obs_r[0], obs_r[1], obs_r[2], obs_r[3],
                                         obs_r[4]/MAX_V1, obs_r[5]/MAX_V2], dtype=np.float32)
                    env.unwrapped.state = (theta1, theta2, dth1, dth2)
                    c = -tip_height(s_cur)
                    if c < best_cost:
                        best_cost = c
                        best_action = seq[0]
                action = best_action

            obs, _, term, trunc, _ = env.step(action)
            if term:
                successes += 1
                all_steps.append(step + 1)
                ok = True
                break

        if not ok:
            all_steps.append(500)
        elapsed = time.time() - t0
        print(f"  [{trial+1:2d}] {'✓' if ok else '✗'} steps={all_steps[-1]}  ({elapsed:.0f}s)")

    env.close()
    return successes, all_steps


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='1-step',
                        choices=['1-step', 'shooting', 'exhaustive'])
    parser.add_argument('--cost', type=str, default='height')
    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--horizon', type=int, default=6)
    parser.add_argument('--n_samples', type=int, default=200)
    parser.add_argument('--trials', type=int, default=3)
    args = parser.parse_args()

    print("=" * 70)
    print(f"ORACLE MPC (real sim as WM): {args.method} (cost={args.cost})")
    print("=" * 70)

    print(f"\nRunning {args.trials} trials...")
    successes, all_steps = evaluate_oracle_mpc(
        method=args.method, depth=args.depth,
        horizon=args.horizon, n_samples=args.n_samples,
        cost=args.cost, n_trials=args.trials)

    print(f"\n{'='*70}")
    print(f"ORACLE RESULT: {successes}/{args.trials} ({successes*100/args.trials:.0f}%)")
    print(f"  Mean steps: {np.mean(all_steps):.0f}")
    print("=" * 70)


if __name__ == '__main__':
    main()
