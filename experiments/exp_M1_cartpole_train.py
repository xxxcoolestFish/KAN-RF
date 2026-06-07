#!/usr/bin/env python3
"""Train CartPole Lagrangian Network.

Learns mass matrix parameters (mp, mc, length) and potential U(θ)
by fitting predicted accelerations [x_ddot, th_ddot] to finite-difference
accelerations from trajectory data.
"""
import argparse, sys, os, torch
from torch.utils.data import TensorDataset, DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from lagrangian._cartpole import CartPoleLagNet


def load_data(data_path, device='cpu'):
    """Load saved training data.

    Expected format (from exp_M1_cartpole_data.py):
      {'train': (X, Y), 'val': (X, Y)}
      X: (N, 6) = [cosθ, sinθ, x, x_dot, th_dot, force]
      Y: (N, 2) = [x_ddot, th_ddot]
    """
    data = torch.load(data_path, map_location=device, weights_only=False)
    train_X, train_Y = data['train']
    val_X, val_Y = data['val']
    return (train_X[:, 0], train_X[:, 1], train_X[:, 3], train_X[:, 4], train_X[:, 5],
            train_Y[:, 0], train_Y[:, 1],
            val_X[:, 0], val_X[:, 1], val_X[:, 3], val_X[:, 4], val_X[:, 5],
            val_Y[:, 0], val_Y[:, 1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='saved_data/cartpole_lag_data.pt')
    parser.add_argument('--hidden', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch', type=int, default=512)
    parser.add_argument('--lr', type=float, default=5e-3)
    parser.add_argument('--save', type=str, default='saved_models/cartpole_lagrangian.pt')
    args = parser.parse_args()

    device = torch.device('cpu')

    result = load_data(args.data, device)
    (cos_tr, sin_tr, xd_tr, thd_tr, f_tr, xdd_tr, thdd_tr,
     cos_va, sin_va, xd_va, thd_va, f_va, xdd_va, thdd_va) = result

    n_train = len(cos_tr)
    n_val = len(cos_va)
    print(f"Train: {n_train}, Val: {n_val}")

    # DataLoaders
    ds_tr = TensorDataset(cos_tr, sin_tr, xd_tr, thd_tr, f_tr, xdd_tr, thdd_tr)
    ds_va = TensorDataset(cos_va, sin_va, xd_va, thd_va, f_va, xdd_va, thdd_va)
    train_ldr = DataLoader(ds_tr, batch_size=args.batch, shuffle=True)
    val_ldr = DataLoader(ds_va, batch_size=args.batch, shuffle=False)

    model = CartPoleLagNet(hidden=args.hidden).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"CartPoleLagNet: hidden={args.hidden}, params={n_params}")
    print(f"  Init: mp={model.mp.item():.4f}, mc={model.mc.item():.4f}, "
          f"len={model.length.item():.4f}, I={model.I_theta.item():.4f}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_val = float('inf')

    for epoch in range(args.epochs):
        # -- Train --
        model.train()
        train_losses = []
        for batch in train_ldr:
            c, s, xd, td, frc, xdd_t, thdd_t = [b.to(device) for b in batch]

            # Enable grad on angle inputs for autograd
            c = c.detach().requires_grad_(True)
            s = s.detach().requires_grad_(True)

            xdd_p, thdd_p = model(c, s, xd, td, frc)

            loss = torch.mean((xdd_p - xdd_t) ** 2 + (thdd_p - thdd_t) ** 2)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            train_losses.append(loss.item())

        # -- Val --
        model.eval()
        val_losses = []
        for batch in val_ldr:
            c, s, xd, td, frc, xdd_t, thdd_t = [b.to(device) for b in batch]
            c.requires_grad_(True)
            s.requires_grad_(True)

            xdd_p, thdd_p = model(c, s, xd, td, frc)
            vloss = torch.mean((xdd_p - xdd_t) ** 2 + (thdd_p - thdd_t) ** 2)
            val_losses.append(vloss.item())

        avg_t = sum(train_losses) / len(train_losses)
        avg_v = sum(val_losses) / len(val_losses)

        if epoch % 50 == 0 or epoch < 5:
            print(f"Epoch {epoch:4d} | Train={avg_t:.4f} | Val={avg_v:.4f} | "
                  f"mp={model.mp.item():.3f} mc={model.mc.item():.3f} "
                  f"L={model.length.item():.3f} I={model.I_theta.item():.4f}")

        if avg_v < best_val:
            best_val = avg_v
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_loss': avg_v,
                'mp': model.mp.item(),
                'mc': model.mc.item(),
                'length': model.length.item(),
                'I': model.I_theta.item(),
            }, args.save)

    # -- Final analysis --
    print(f"\n=== Final ===")
    print(f"Best val loss: {best_val:.6f}")
    print(f"  mp = {model.mp.item():.4f}  (true: 0.1)")
    print(f"  mc = {model.mc.item():.4f}  (true: 1.0)")
    print(f"  L  = {model.length.item():.4f}  (true: 0.5)")
    print(f"  I  = {model.I_theta.item():.4f}  (true: 0.0333)")

    # Check U(θ) values
    model.eval()
    with torch.no_grad():
        angles = torch.linspace(-torch.pi, torch.pi, 100)
        ca = torch.cos(angles).unsqueeze(1)
        sa = torch.sin(angles).unsqueeze(1)
        U_vals = model.U_net(torch.cat([ca, sa], dim=1)).squeeze()

        top_idx = (angles.abs() < 0.1)
        bot_idx = (angles.abs() > 3.0)
        mid_idx = ((angles - torch.pi/2).abs() < 0.1)
        print(f"  U(θ=0 upright): {U_vals[top_idx].mean().item():.3f}  "
              f"(true: mgL = 0.49)")
        print(f"  U(θ=π/2 horiz): {U_vals[mid_idx].mean().item():.3f}  "
              f"(true: 0)")
        print(f"  U(θ=π hang):   {U_vals[bot_idx].mean().item():.3f}  "
              f"(true: -0.49)")

        # Check dU/dθ
        c_test = torch.tensor([1.0, 0.0, -1.0], requires_grad=True)  # cos(0, π/2, π)
        s_test = torch.tensor([0.0, 1.0, 0.0], requires_grad=True)
        U_prime = model.dU_dtheta(c_test, s_test)
        print(f"  U'(θ=0): {U_prime[0].item():.3f}  (true: 0)")
        print(f"  U'(θ=π/2): {U_prime[1].item():.3f}  (true: mgL = 0.49)")
        print(f"  U'(θ=π): {U_prime[2].item():.3f}  (true: 0)")

    print(f"\nSaved to {args.save}")


if __name__ == '__main__':
    main()
