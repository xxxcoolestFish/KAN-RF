"""ProtoKAN vs KAN: L-BFGS full-batch accuracy comparison on Acrobot WM.

Uses fresh generated single-scale data (no k_norm), full-batch L-BFGS training.
This is the definitive comparison — same conditions, same optimizer.
"""
import torch, torch.nn as nn, numpy as np, time, sys, os
import gymnasium as gym
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kanrf import KAN, ProtoKAN


def generate_single_scale_data(n_states=5000, device='cpu'):
    """Generate single-scale (k=1) Acrobot data. No Jacobian labels."""
    env = gym.make('Acrobot-v1')
    env.reset()
    max_v1, max_v2 = 6.0, 8.0
    xs, ys = [], []

    for _ in range(n_states):
        theta1 = np.random.uniform(-np.pi, np.pi)
        theta2 = np.random.uniform(-np.pi, np.pi)
        dtheta1 = np.random.uniform(-max_v1, max_v1)
        dtheta2 = np.random.uniform(-max_v2, max_v2)
        env.unwrapped.state = (theta1, theta2, dtheta1, dtheta2)
        s0 = env.unwrapped._get_ob()

        for a in range(3):
            env.unwrapped.state = (theta1, theta2, dtheta1, dtheta2)
            obs, _, term, trunc, _ = env.step(a)
            s_norm = np.array([s0[0], s0[1], s0[2], s0[3],
                              s0[4]/max_v1, s0[5]/max_v2], dtype=np.float32)
            s_next_norm = np.array([obs[0], obs[1], obs[2], obs[3],
                                    obs[4]/max_v1, obs[5]/max_v2], dtype=np.float32)
            a_oh = np.zeros(3, dtype=np.float32); a_oh[a] = 1.0
            x = np.concatenate([s_norm, a_oh])
            xs.append(x); ys.append(s_next_norm)

    env.close()
    X = torch.tensor(np.array(xs), dtype=torch.float32).to(device)
    Y = torch.tensor(np.array(ys), dtype=torch.float32).to(device)
    return X, Y


def train_lbfgs(model, X_tr, Y_tr, X_val, Y_val, max_iter=200, label='model'):
    """Train with L-BFGS on full training set."""
    mse_fn = nn.MSELoss()
    best_val = float('inf')
    best_state = None
    t0 = time.time()

    def closure():
        model.train()
        optimizer.zero_grad()
        loss = mse_fn(model(X_tr), Y_tr)
        loss.backward()
        return loss

    optimizer = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=20,
                                  history_size=50, line_search_fn='strong_wolfe')

    for step in range(1, max_iter + 1):
        loss = optimizer.step(closure)
        model.eval()
        with torch.no_grad():
            val_mse = mse_fn(model(X_val), Y_val).item()
        if val_mse < best_val:
            best_val = val_mse
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if step % 25 == 0 or step == 1:
            print(f"  [{label}] L-BFGS {step:3d}/{max_iter}  "
                  f"val_mse={val_mse:.6f}  best={best_val:.6f}  {time.time()-t0:.0f}s")

    if best_state:
        model.load_state_dict(best_state)
    return best_val


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_states', type=int, default=5000,
                        help='Number of unique states (x3 for data points)')
    parser.add_argument('--hidden', type=int, default=32)
    parser.add_argument('--n_prototypes', type=int, default=16)
    parser.add_argument('--lbfgs_iters', type=int, default=200)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--skip_data_gen', action='store_true', default=False)
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(42); np.random.seed(42)

    print("=" * 70)
    print("ProtoKAN vs KAN: L-BFGS Full-Batch Acrobot WM")
    print("=" * 70)

    # Generate or load data
    data_file = f'/tmp/acrobot_ss_{args.n_states}.pt'
    if args.skip_data_gen and os.path.exists(data_file):
        X, Y = torch.load(data_file, weights_only=True, map_location=device)
        print(f"Loaded {len(X)} cached samples")
    else:
        print(f"Generating {args.n_states} states ({args.n_states*3} data points)...")
        X, Y = generate_single_scale_data(args.n_states, device)
        torch.save((X, Y), data_file)
        print(f"  Cached to {data_file}")

    print(f"  X: {X.shape}, Y: {Y.shape}")

    # Train/val split
    n_train = int(len(X) * 0.85)
    X_tr, Y_tr = X[:n_train], Y[:n_train]
    X_val, Y_val = X[n_train:], Y[n_train:]

    layer_dims = [9, args.hidden, 6]
    results = {}

    # ── ProtoKAN ──
    print(f"\n--- ProtoKAN [9,{args.hidden},6] N={args.n_prototypes} ---")
    torch.manual_seed(42); np.random.seed(42)
    proto = ProtoKAN(layer_dims, n_prototypes=args.n_prototypes).to(device)
    print(f"  Params: {count_params(proto)}")
    results['ProtoKAN'] = train_lbfgs(proto, X_tr, Y_tr, X_val, Y_val,
                                      max_iter=args.lbfgs_iters, label='ProtoKAN')

    # ── KAN ──
    print(f"\n--- KAN [9,{args.hidden},6] grid=5 order=3 ---")
    torch.manual_seed(42); np.random.seed(42)
    kan = KAN(layer_dims, grid_size=5, spline_order=3).to(device)
    print(f"  Params: {count_params(kan)}")
    results['KAN'] = train_lbfgs(kan, X_tr, Y_tr, X_val, Y_val,
                                 max_iter=args.lbfgs_iters, label='KAN')

    # ── MLP ──
    print(f"\n--- MLP [9,{args.hidden},{args.hidden},6] ---")
    torch.manual_seed(42); np.random.seed(42)
    mlp = nn.Sequential(
        nn.Linear(9, args.hidden), nn.SiLU(),
        nn.Linear(args.hidden, args.hidden), nn.SiLU(),
        nn.Linear(args.hidden, 6)
    ).to(device)
    print(f"  Params: {count_params(mlp)}")
    results['MLP'] = train_lbfgs(mlp, X_tr, Y_tr, X_val, Y_val,
                                 max_iter=args.lbfgs_iters, label='MLP')

    # ── Report ──
    print(f"\n{'='*70}")
    print("FINAL RESULTS (L-BFGS, full batch)")
    print("=" * 70)
    for name, mse in sorted(results.items(), key=lambda x: x[1]):
        print(f"  {name:20s}  val_mse={mse:.6f}")

    kan_mse = results['KAN']
    proto_mse = results['ProtoKAN']
    mlp_mse = results['MLP']
    best = min(kan_mse, proto_mse, mlp_mse)
    print(f"\n  Best: {best:.6f}")
    if proto_mse < kan_mse:
        print(f"  ProtoKAN / KAN: {kan_mse/proto_mse:.2f}x improvement")
    else:
        print(f"  KAN / ProtoKAN: {proto_mse/kan_mse:.2f}x improvement")
    print(f"  Gap to MLP: {best/mlp_mse:.2f}x" if best > mlp_mse else
          f"  Better than MLP by: {mlp_mse/best:.2f}x")


if __name__ == '__main__':
    main()
