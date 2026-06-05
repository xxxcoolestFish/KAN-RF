"""Phase 1: Train KAN world model on Pendulum dynamics."""
import torch
from kanrf import KAN


def main():
    torch.manual_seed(42)

    # Load data
    x, y = torch.load("pendulum_data.pt", weights_only=True)

    # Train/val split
    n_train = int(len(x) * 0.85)
    x_train, y_train = x[:n_train], y[:n_train]
    x_val, y_val = x[n_train:], y[n_train:]

    # KAN: input=[cosθ, sinθ, θ̇, torque] → output=[cosθ', sinθ', θ̇']
    model = KAN([4, 12, 3], grid_size=5, spline_order=3)
    print(f"Params: {sum(p.numel() for p in model.parameters())}")

    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=600, gamma=0.5)
    loss_fn = torch.nn.MSELoss()

    n_epochs = 2400
    for epoch in range(1, n_epochs + 1):
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
                # Per-dim MSE (denormalized: cosθ, sinθ, θ̇*8)
                dim_mse = ((pred - y_val) ** 2).mean(dim=0)
            print(f"Epoch {epoch:4d}  lr={scheduler.get_last_lr()[0]:.4f}  "
                  f"train={loss.item():.8f}  val={vloss.item():.8f}  "
                  f"dim_mse=[{dim_mse[0]:.6f},{dim_mse[1]:.6f},{dim_mse[2]:.6f}]")

    model.eval()
    with torch.no_grad():
        val_pred = model(x_val)
        final_mse = loss_fn(val_pred, y_val).item()
        dim_mse = ((val_pred - y_val) ** 2).mean(dim=0)

    print(f"\nFinal val MSE: {final_mse:.8f}")
    print(f"Per-dim MSE (norm): cosθ={dim_mse[0]:.8f} sinθ={dim_mse[1]:.8f} θ̇={dim_mse[2]:.8f}")
    # Denormalized theta_dot MSE
    print(f"θ̇ MSE (raw): {dim_mse[2].item() * 64:.6f}")  # scale^2 = 8^2 = 64

    torch.save(model.state_dict(), "kan_pendulum_model.pt")
    print("\nSaved: kan_pendulum_model.pt")


if __name__ == "__main__":
    main()
