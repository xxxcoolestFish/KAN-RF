"""Train KAN with unit-circle constraint on cos/sin outputs.

The model outputs [cos, sin, thd/8]. cos² + sin² should equal 1.
Standard training doesn't enforce this, causing divergence in multi-step rollouts.

Adds: L = MSE + λ_norm * (cos² + sin² - 1)²
"""
import torch, argparse
from kanrf import KAN


def main(lam_norm=10.0):
    torch.manual_seed(42)

    x, y = torch.load("pendulum_data_v4.pt", weights_only=True)
    n_train = int(len(x) * 0.85)
    x_train, y_train = x[:n_train], y[:n_train]
    x_val, y_val = x[n_train:], y[n_train:]

    model = KAN([4, 12, 3], grid_size=5, spline_order=3)
    print(f"Norm-constrained KAN, λ_norm={lam_norm}  |  params={sum(p.numel() for p in model.parameters())}")

    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=600, gamma=0.5)
    mse_fn = torch.nn.MSELoss()

    for epoch in range(1, 2401):
        model.train()
        opt.zero_grad()
        pred = model(x_train)
        loss_mse = mse_fn(pred, y_train)
        # Unit-circle penalty: cos² + sin² should equal 1
        norm_dev = (pred[:, 0]**2 + pred[:, 1]**2 - 1.0)
        loss_norm = lam_norm * (norm_dev ** 2).mean()
        loss = loss_mse + loss_norm
        loss.backward()
        opt.step()
        scheduler.step()

        if epoch % 400 == 0:
            model.eval()
            with torch.no_grad():
                val_pred = model(x_val)
                val_norms = (val_pred[:, 0]**2 + val_pred[:, 1]**2).sqrt()
                norm_mean_err = (val_norms - 1.0).abs().mean().item()
                print(f"Epoch {epoch:4d}  lr={scheduler.get_last_lr()[0]:.4f}  "
                      f"mse={loss_mse.item():.6f}  norm_loss={loss_norm.item():.6f}  "
                      f"val_norm_err={norm_mean_err:.6f}")

    model.eval()
    with torch.no_grad():
        val_pred = model(x_val)
        val_mse = mse_fn(val_pred, y_val).item()
        val_norms = (val_pred[:, 0]**2 + val_pred[:, 1]**2).sqrt()
        print(f"Final val MSE: {val_mse:.8f}  mean|norm-1|: {(val_norms-1.0).abs().mean().item():.6f}")

    fname = "kan_norm_constrained.pt"
    torch.save(model.state_dict(), fname)
    print(f"Saved: {fname}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lam-norm', type=float, default=10.0)
    args = parser.parse_args()
    main(lam_norm=args.lam_norm)
