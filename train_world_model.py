"""Phase 1: Train KAN world model to predict s_{t+1} from (s_t, a_t)."""
import torch
from kanrf import KAN
from env import PointMass, generate_data


def main():
    torch.manual_seed(42)

    # --- Env ---
    env = PointMass(nonlinear=False)

    # --- Data ---
    x_train, y_train = generate_data(env, n_samples=2000)
    x_val, y_val = generate_data(env, n_samples=500)

    # --- Model ---
    model = KAN([4, 5, 2], grid_size=5, spline_order=3)
    print(f"Model params: {sum(p.numel() for p in model.parameters())}")

    # --- Train ---
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=500, gamma=0.5)
    loss_fn = torch.nn.MSELoss()

    n_epochs = 2000
    for epoch in range(1, n_epochs + 1):
        model.train()
        opt.zero_grad()
        pred = model(x_train)
        loss = loss_fn(pred, y_train)
        loss.backward()
        opt.step()
        scheduler.step()

        if epoch % 400 == 0:
            model.eval()
            with torch.no_grad():
                val_pred = model(x_val)
                val_loss = loss_fn(val_pred, y_val)
            print(f"Epoch {epoch:4d}  lr={scheduler.get_last_lr()[0]:.4f}  train_loss={loss.item():.8f}  val_loss={val_loss.item():.8f}")

    # --- Final eval ---
    model.eval()
    with torch.no_grad():
        val_pred = model(x_val)
        val_mse = loss_fn(val_pred, y_val).item()
        dim_mse = ((val_pred - y_val) ** 2).mean(dim=0)

    print(f"\nFinal val MSE: {val_mse:.10f}")
    print(f"Per-dim MSE:   [{dim_mse[0]:.10f}, {dim_mse[1]:.10f}]")

    # --- Save ---
    torch.save(model.state_dict(), "kan_world_model.pt")
    print("\nSaved: kan_world_model.pt")


if __name__ == "__main__":
    main()
