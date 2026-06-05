"""Combine Hybrid (MOPS + CWS) with unit-circle norm constraint.

L = MSE + λ·||Δ²c||² + ν·||w⊙(∂f/∂a - J_true)||² + λ_norm·(cos²+sin²-1)²
"""
import torch, argparse
from kanrf import KAN
from kanrf import p_spline_penalty
from kanrf import jacobian_loss, true_jacobian


def main(lam=0.1, nu=0.1, lam_norm=1.0):
    torch.manual_seed(42)
    device = torch.device('cpu')

    x, y = torch.load("pendulum_data_v4.pt", weights_only=True)
    n_train = int(len(x) * 0.85)
    x_train, y_train = x[:n_train], y[:n_train]
    x_val, y_val = x[n_train:], y[n_train:]

    model = KAN([4, 12, 3], grid_size=5, spline_order=3)
    w = torch.tensor([1.0, 1.0, 3.0], device=device)
    mse_fn = torch.nn.MSELoss()
    batch_size = 2048

    print(f"Full constrained: λ={lam} ν={nu} λ_norm={lam_norm}  params={sum(p.numel() for p in model.parameters())}")

    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=600, gamma=0.5)

    for epoch in range(1, 2401):
        model.train()
        perm = torch.randperm(len(x_train))
        total = {'mse': 0, 'ps': 0, 'jac': 0, 'norm': 0}
        nb = 0

        for start in range(0, len(x_train), batch_size):
            idx = perm[start:start + batch_size]
            s_b, a_b, y_b = x_train[idx, :3], x_train[idx, 3:4], y_train[idx]

            pred = model(torch.cat([s_b, a_b], dim=-1))
            loss = mse_fn(pred, y_b)

            # MOPS: control-point smoothness
            if lam > 0:
                loss_ps = lam * p_spline_penalty(model)
                loss = loss + loss_ps
                total['ps'] += loss_ps.item()

            # CWS: Jacobian matching
            if nu > 0:
                loss_jac = nu * jacobian_loss(model, s_b, a_b, y_b, w)
                loss = loss + loss_jac
                total['jac'] += loss_jac.item()

            # Norm: unit-circle constraint
            if lam_norm > 0:
                norm_dev = (pred[:, 0]**2 + pred[:, 1]**2 - 1.0)
                loss_norm = lam_norm * (norm_dev ** 2).mean()
                loss = loss + loss_norm
                total['norm'] += loss_norm.item()

            opt.zero_grad()
            loss.backward()
            opt.step()
            total['mse'] += mse_fn(pred, y_b).item()
            nb += 1

        scheduler.step()

        if epoch % 400 == 0:
            model.eval()
            with torch.no_grad():
                val_pred = model(x_val)
                val_mse = mse_fn(val_pred, y_val).item()
                val_norms = (val_pred[:, 0]**2 + val_pred[:, 1]**2).sqrt()
                norm_err = (val_norms - 1.0).abs().mean().item()
            print(f"Epoch {epoch:4d}  lr={scheduler.get_last_lr()[0]:.4f}  "
                  f"mse={total['mse']/nb:.6f}  ps={total['ps']/nb:.4f}  "
                  f"jac={total['jac']/nb:.4f}  norm={total['norm']/nb:.4f}  "
                  f"val_mse={val_mse:.6f}  val_norm={norm_err:.4f}")

    model.eval()
    with torch.no_grad():
        val_pred = model(x_val)
        val_mse = mse_fn(val_pred, y_val).item()
    print(f"Final val MSE: {val_mse:.8f}")
    fname = "kan_full_constrained.pt"
    torch.save(model.state_dict(), fname)
    print(f"Saved: {fname}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lam', type=float, default=0.1)
    parser.add_argument('--nu', type=float, default=0.1)
    parser.add_argument('--lam-norm', type=float, default=1.0)
    args = parser.parse_args()
    main(lam=args.lam, nu=args.nu, lam_norm=args.lam_norm)
