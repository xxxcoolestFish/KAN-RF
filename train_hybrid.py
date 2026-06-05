"""Train KAN with Hybrid loss: P-spline (MOPS) + Controllability-Weighted Sobolev (CWS).

L = MSE + λ·||Δ²c||² + ν·||w ⊙ (∂f/∂a - J_true)||²

Usage:
  python train_hybrid.py --lam 0.1 --nu 0.1
"""
import torch, argparse
from kanrf import KAN
from kanrf import p_spline_penalty
from kanrf import true_jacobian, jacobian_loss


def main(lam=0.1, nu=0.1):
    torch.manual_seed(42)
    device = torch.device('cpu')

    x, y = torch.load("pendulum_data_v4.pt", weights_only=True)
    n_train = int(len(x) * 0.85)
    x_train, y_train = x[:n_train], y[:n_train]
    x_val, y_val = x[n_train:], y[n_train:]

    model = KAN([4, 12, 3], grid_size=5, spline_order=3)
    w = torch.tensor([1.0, 1.0, 3.0], device=device)
    print(f"Hybrid λ={lam} ν={nu}  |  params: {sum(p.numel() for p in model.parameters())}")

    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=600, gamma=0.5)
    mse_fn = torch.nn.MSELoss()
    batch_size = 2048

    for epoch in range(1, 2401):
        model.train()
        perm = torch.randperm(len(x_train))
        total_mse, total_jac, total_rough = 0.0, 0.0, 0.0
        n_batches = 0

        for start in range(0, len(x_train), batch_size):
            idx = perm[start:start + batch_size]
            s_b = x_train[idx, :3]
            a_b = x_train[idx, 3:4]
            y_b = y_train[idx]

            pred = model(torch.cat([s_b, a_b], dim=-1))
            loss = mse_fn(pred, y_b)
            if lam > 0:
                loss = loss + lam * p_spline_penalty(model)
            if nu > 0:
                loss_jac = jacobian_loss(model, s_b, a_b, y_b, w)
                loss = loss + nu * loss_jac
                total_jac += loss_jac.item()

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_mse += mse_fn(pred, y_b).item()
            total_rough += p_spline_penalty(model).item()
            n_batches += 1

        scheduler.step()

        if epoch % 400 == 0:
            model.eval()
            with torch.no_grad():
                val_mse = mse_fn(model(x_val), y_val).item()
            print(f"Epoch {epoch:4d}  lr={scheduler.get_last_lr()[0]:.4f}  "
                  f"mse={total_mse/n_batches:.6f}  jac={total_jac/n_batches:.6f}  "
                  f"rough={total_rough/n_batches:.4f}  val_mse={val_mse:.6f}")

    model.eval()
    with torch.no_grad():
        val_mse = mse_fn(model(x_val), y_val).item()
    print(f"Final val MSE: {val_mse:.8f}")

    fname = f"kan_hybrid_lam{lam}_nu{nu}.pt"
    torch.save(model.state_dict(), fname)
    print(f"Saved: {fname}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lam', type=float, default=0.1, help='P-spline weight')
    parser.add_argument('--nu', type=float, default=0.1, help='Sobolev (Jacobian) weight')
    args = parser.parse_args()
    main(lam=args.lam, nu=args.nu)
