"""ProtoKAN vs KAN accuracy benchmark on Acrobot World Model.

Uses the same data, same optimizer (Adam), same training settings.
Only difference: the network architecture (ProtoKAN vs KAN).

Also compares with MLP baseline.
"""
import torch, torch.nn as nn, numpy as np, time, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kanrf import KAN, ProtoKAN


def load_data(device='cpu'):
    """Load Acrobot CWS data, extract k=1 (single-scale) subset.

    Returns X (N,9), Y (N,6) for single-scale WM training.
    """
    x_data, y_data = torch.load('acrobot/acrobot_data_ms.pt',
                                weights_only=True, map_location=device)
    x_data, y_data = x_data.float(), y_data.float()

    # Filter for k=1 scale only
    mask = x_data[:, 9] == 1.0
    X = x_data[mask][:, :9]  # (state 6 + action 3)
    Y = y_data[mask]  # next state 6
    print(f"Data: {X.shape[0]} samples (k=1), input={X.shape[1]}, output={Y.shape[1]}")
    return X, Y


def make_mlp(layer_dims):
    """Build MLP with same layer dimensions."""
    layers = []
    for i in range(len(layer_dims) - 1):
        layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
        if i < len(layer_dims) - 2:
            layers.append(nn.SiLU())
    return nn.Sequential(*layers)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def train_model(model, X_tr, Y_tr, X_val, Y_val, epochs=600,
                lr=1e-2, batch_size=1024, label='model', lr_step=150,
                lr_gamma=0.5, n_batches_per_epoch=1):
    """Train a WM model with MSE loss. Returns best_val_mse, history."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=lr_step, gamma=lr_gamma)
    mse_fn = nn.MSELoss()
    best_val = float('inf')
    best_state = None
    n_train = len(X_tr)
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        for _ in range(n_batches_per_epoch):
            idx = torch.randperm(n_train)[:batch_size]
            xb, yb = X_tr[idx], Y_tr[idx]
            opt.zero_grad()
            loss = mse_fn(model(xb), yb)
            loss.backward()
            opt.step()
        scheduler.step()

        if epoch % 100 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                val_mse = mse_fn(model(X_val), Y_val).item()
            if val_mse < best_val:
                best_val = val_mse
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

            lr_now = scheduler.get_last_lr()[0]
            elapsed = time.time() - t0
            print(f"  [{label}] Epoch {epoch:3d}  lr={lr_now:.4f}  "
                  f"val_mse={val_mse:.6f}  best={best_val:.6f}  {elapsed:.0f}s")

    if best_state:
        model.load_state_dict(best_state)
    return best_val


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=600)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--device', type=str, default='cpu')
    # ProtoKAN settings
    parser.add_argument('--n_prototypes', type=int, default=16)
    # Sweep mode
    parser.add_argument('--sweep', action='store_true', default=False)
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(42); np.random.seed(42)

    print("=" * 70)
    print("ProtoKAN vs KAN vs MLP — Acrobot World Model Accuracy")
    print("=" * 70)

    # Load data
    X, Y = load_data(device)
    n_train = int(len(X) * 0.85)
    X_tr, Y_tr = X[:n_train], Y[:n_train]
    X_val, Y_val = X[n_train:], Y[n_train:]

    # Common settings
    layer_dims = [9, 24, 6]  # small model for quick comparison
    epochs = args.epochs
    lr = args.lr

    if args.sweep:
        # Sweep over proto counts
        print(f"\n{'='*70}")
        print("SWEEP: ProtoKAN with different prototype counts")
        print("=" * 70)
        results = {}
        for n_proto in [8, 12, 16, 24, 32]:
            print(f"\n--- ProtoKAN N={n_proto} ---")
            torch.manual_seed(42); np.random.seed(42)
            model = ProtoKAN(layer_dims, n_prototypes=n_proto).to(device)
            n_params = count_params(model)
            print(f"  Params: {n_params}")
            best_mse = train_model(model, X_tr, Y_tr, X_val, Y_val,
                                   epochs=epochs, lr=lr,
                                   label=f'ProtoKAN-N{n_proto}')
            results[f'ProtoKAN-N{n_proto}'] = best_mse

        print(f"\n{'='*70}")
        print("SWEEP RESULTS:")
        print("=" * 70)
        for name, mse in sorted(results.items(), key=lambda x: x[1]):
            print(f"  {name:20s}  val_mse={mse:.6f}")
    else:
        # Single comparison
        results = {}

        # 1. ProtoKAN
        print(f"\n--- ProtoKAN N={args.n_prototypes} ---")
        torch.manual_seed(42); np.random.seed(42)
        proto = ProtoKAN(layer_dims, n_prototypes=args.n_prototypes).to(device)
        print(f"  Params: {count_params(proto)}")
        results['ProtoKAN'] = train_model(proto, X_tr, Y_tr, X_val, Y_val,
                                          epochs=epochs, lr=lr, label='ProtoKAN')

        # 2. KAN (B-spline)
        print(f"\n--- KAN (B-spline) ---")
        torch.manual_seed(42); np.random.seed(42)
        kan = KAN(layer_dims, grid_size=5, spline_order=3).to(device)
        print(f"  Params: {count_params(kan)}")
        results['KAN'] = train_model(kan, X_tr, Y_tr, X_val, Y_val,
                                     epochs=epochs, lr=lr, label='KAN')

        # 3. MLP
        print(f"\n--- MLP ---")
        torch.manual_seed(42); np.random.seed(42)
        mlp = make_mlp(layer_dims).to(device)
        print(f"  Params: {count_params(mlp)}")
        results['MLP'] = train_model(mlp, X_tr, Y_tr, X_val, Y_val,
                                     epochs=epochs, lr=lr, label='MLP')

        print(f"\n{'='*70}")
        print("RESULTS:")
        print("=" * 70)
        for name, mse in sorted(results.items(), key=lambda x: x[1]):
            print(f"  {name:20s}  val_mse={mse:.6f}")

        # Improvement ratios
        kan_mse = results['KAN']
        proto_mse = results['ProtoKAN']
        mlp_mse = results['MLP']
        print(f"\n  ProtoKAN vs KAN:   {kan_mse/proto_mse:.2f}x better" if proto_mse < kan_mse
              else f"  KAN vs ProtoKAN:   {proto_mse/kan_mse:.2f}x better")
        print(f"  MLP vs best:        {min(proto_mse, kan_mse)/mlp_mse:.2f}x gap to MLP")


if __name__ == '__main__':
    main()
