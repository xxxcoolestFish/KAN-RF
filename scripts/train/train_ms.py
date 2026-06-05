"""Train multi-scale KAN world model: f(s, a, k) → s_{t+k·dt}.

Architecture: [5, 16, 3] — slightly larger hidden layer than single-scale [4,12,3]
since the model must represent 5 different dynamics regimes (k ∈ {1,2,4,8,16}).

Loss: MSE + λ·‖Δ²c‖² (MOPS P-spline for smooth derivatives).

Usage:
  python train_ms.py              # default λ=0.1
  python train_ms.py --lam 0.01   # lighter regularization
"""
import torch, argparse, time
from kanrf import KAN


def p_spline_penalty(model):
    """Σ ‖Δ²c‖² across all spline weights."""
    total = 0.0
    for layer in model.layers:
        c = layer.spline_weight          # (out, in, n_basis)
        d2 = c[:, :, :-2] - 2 * c[:, :, 1:-1] + c[:, :, 2:]
        total += (d2 ** 2).mean()
    return total


def main(lam=0.1):
    torch.manual_seed(42)

    x_all, y_all = torch.load("pendulum_data_ms.pt", weights_only=True)
    # Subsample evenly per k value: 5k per k = 25k total (enough for 1152 params)
    n_per_k = 5000
    idxs = []
    for k_val in [1, 2, 4, 8, 16]:
        mask = (x_all[:, 4] * 16).round() == k_val
        k_idx = torch.where(mask)[0]
        chosen = k_idx[torch.randperm(len(k_idx))[:n_per_k]]
        idxs.append(chosen)
    idx = torch.cat(idxs)
    # Further split into train/val
    idx = idx[torch.randperm(len(idx))]
    n_train = int(len(idx) * 0.85)
    x_train, y_train = x_all[idx[:n_train]], y_all[idx[:n_train]]
    x_val, y_val = x_all[idx[n_train:]], y_all[idx[n_train:]]

    model = KAN([5, 16, 3], grid_size=5, spline_order=3)
    print(f"Multi-scale KAN: [5, 16, 3]  |  params={sum(p.numel() for p in model.parameters())}  |  λ={lam}")
    print(f"Train: {len(x_train)}, Val: {len(x_val)}")

    # NOTE: torch.compile hangs on KAN's einsum operations.
    # Full-batch CPU training is fast enough for this model size.

    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=600, gamma=0.5)
    mse_fn = torch.nn.MSELoss()

    t_start = time.time()
    for epoch in range(1, 1201):
        model.train()
        opt.zero_grad()
        pred = model(x_train)
        loss = mse_fn(pred, y_train)
        if lam > 0:
            loss = loss + lam * p_spline_penalty(model)
        loss.backward()
        opt.step()
        scheduler.step()

        if epoch % 200 == 0:
            model.eval()
            with torch.no_grad():
                val_mse = mse_fn(model(x_val), y_val).item()
            elapsed = time.time() - t_start
            eta = elapsed / epoch * (1200 - epoch)
            print(f"Epoch {epoch:4d}  lr={scheduler.get_last_lr()[0]:.4f}  "
                  f"train={loss.item():.6f}  val_mse={val_mse:.6f}  "
                  f"[{elapsed:.0f}s ETA {eta:.0f}s]")

    model.eval()
    with torch.no_grad():
        val_mse = mse_fn(model(x_val), y_val).item()
        # Per-k breakdown
        for k_val in [1, 2, 4, 8, 16]:
            mask = (x_val[:, 4] * 16).round() == k_val
            if mask.sum() > 0:
                k_mse = mse_fn(model(x_val[mask]), y_val[mask]).item()
                print(f"  k={k_val:2d}: val_mse={k_mse:.8f}  (n={mask.sum().item()})")

    print(f"Final val MSE: {val_mse:.8f}  |  time: {time.time()-t_start:.0f}s")

    fname = f"kan_ms.pt"
    torch.save(model.state_dict(), fname)
    print(f"Saved: {fname}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lam', type=float, default=0.1)
    args = parser.parse_args()
    main(lam=args.lam)
