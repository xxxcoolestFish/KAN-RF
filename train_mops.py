"""Train KAN with P-spline (control-point curvature) regularization.

Adds ||Δ²c||² penalty to standard MSE loss. This directly penalizes
the first-derivative energy of each B-spline edge function, encouraging
smooth derivatives without requiring knowledge of the true Jacobian.

Usage:
  python train_mops.py --lam 0.1
  python train_mops.py --lam 0.01
  python train_mops.py --lam 1.0
  python train_mops.py --lam 0     # baseline (standard MSE)
"""
import torch, argparse
from kanrf import KAN, p_spline_penalty


def main(lam=0.1):
    torch.manual_seed(42)
    device = torch.device('cpu')

    # Load data
    x, y = torch.load("pendulum_data_v4.pt", weights_only=True)
    n_train = int(len(x) * 0.85)
    x_train, y_train = x[:n_train], y[:n_train]
    x_val, y_val = x[n_train:], y[n_train:]

    model = KAN([4, 12, 3], grid_size=5, spline_order=3)
    print(f"P-spline λ={lam}  |  params: {sum(p.numel() for p in model.parameters())}")

    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=600, gamma=0.5)
    mse_fn = torch.nn.MSELoss()

    for epoch in range(1, 2401):
        model.train()
        opt.zero_grad()
        pred = model(x_train)
        loss = mse_fn(pred, y_train)
        if lam > 0:
            loss = loss + lam * p_spline_penalty(model)
        loss.backward()
        opt.step()
        scheduler.step()

        if epoch % 400 == 0:
            model.eval()
            with torch.no_grad():
                val_mse = mse_fn(model(x_val), y_val).item()
            print(f"Epoch {epoch:4d}  lr={scheduler.get_last_lr()[0]:.4f}  "
                  f"train={loss.item():.6f}  val_mse={val_mse:.6f}")

    # Final eval
    model.eval()
    with torch.no_grad():
        val_pred = model(x_val)
        val_mse = mse_fn(val_pred, y_val).item()
    print(f"Final val MSE: {val_mse:.8f}")

    fname = f"kan_mops_lam{lam}.pt"
    torch.save(model.state_dict(), fname)
    print(f"Saved: {fname}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lam', type=float, default=0.1, help='P-spline weight')
    args = parser.parse_args()
    main(lam=args.lam)
