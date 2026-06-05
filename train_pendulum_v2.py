"""Retrain KAN world model on mixed data (controller + random)."""
import torch
from kanrf import KAN


def main():
    torch.manual_seed(42)

    x, y = torch.load("pendulum_data_mixed.pt", weights_only=True)
    n_train = int(len(x) * 0.85)
    x_train, y_train = x[:n_train], y[:n_train]
    x_val, y_val = x[n_train:], y[n_train:]

    model = KAN([4, 12, 3], grid_size=5, spline_order=3)
    print(f"Params: {sum(p.numel() for p in model.parameters())}")
    print(f"Train: {len(x_train)}, Val: {len(x_val)}")

    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=600, gamma=0.5)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(1, 2401):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(x_train), y_train)
        loss.backward()
        opt.step()
        scheduler.step()

        if epoch % 400 == 0:
            model.eval()
            with torch.no_grad():
                pred = model(x_val)
                vloss = loss_fn(pred, y_val)
                dim_mse = ((pred - y_val) ** 2).mean(dim=0)
            print(f"Epoch {epoch:4d}  lr={scheduler.get_last_lr()[0]:.4f}  "
                  f"train={loss.item():.8f}  val={vloss.item():.8f}  "
                  f"dim=[{dim_mse[0]:.6f},{dim_mse[1]:.6f},{dim_mse[2]:.6f}]")

    model.eval()
    with torch.no_grad():
        pred = model(x_val)
        final_mse = loss_fn(pred, y_val).item()
        dim_mse = ((pred - y_val) ** 2).mean(dim=0)
    print(f"\nFinal val MSE: {final_mse:.8f}")
    print(f"Per-dim: cosθ={dim_mse[0]:.8f} sinθ={dim_mse[1]:.8f} θ̇={dim_mse[2]:.8f}")
    print(f"θ̇ MSE (raw): {dim_mse[2].item() * 64:.6f}")

    torch.save(model.state_dict(), "kan_pendulum_model_v2.pt")
    print("Saved: kan_pendulum_model_v2.pt")


if __name__ == "__main__":
    main()
