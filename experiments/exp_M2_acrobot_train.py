#!/usr/bin/env python3
"""Train Acrobot Lagrangian Network.

Predicts [θ̈₁, θ̈₂] from (cosθ₁, sinθ₁, cosθ₂, sinθ₂, θ̇₁, θ̇₂, torque).
Target accelerations are computed analytically from Lagrangian dynamics.
"""
import argparse, sys, os, torch
from torch.utils.data import TensorDataset, DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from lagrangian._acrobot import AcrobotLagNet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='saved_data/acrobot_lag_data.pt')
    parser.add_argument('--hidden', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch', type=int, default=512)
    parser.add_argument('--lr', type=float, default=5e-3)
    parser.add_argument('--save', type=str, default='saved_models/acrobot_lagrangian.pt')
    args = parser.parse_args()

    device = torch.device('cpu')

    data = torch.load(args.data, map_location=device, weights_only=False)
    train_X, train_Y = data['train']
    val_X, val_Y = data['val']

    n_train = len(train_X)
    n_val = len(val_X)
    print(f"Train: {n_train}, Val: {n_val}")
    print(f"train_Y θ̈₁ std: {train_Y[:,0].std():.2f}, θ̈₂ std: {train_Y[:,1].std():.2f}")

    ds_tr = TensorDataset(train_X, train_Y)
    ds_va = TensorDataset(val_X, val_Y)
    train_ldr = DataLoader(ds_tr, batch_size=args.batch, shuffle=True)
    val_ldr = DataLoader(ds_va, batch_size=args.batch, shuffle=False)

    model = AcrobotLagNet(hidden=args.hidden).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    diag = model.true_params()
    print(f"AcrobotLagNet: hidden={args.hidden}, params={n_params}")
    print(f"  Init: a₁={diag['a1']:.2f}, b₁={diag['b1']:.2f}, "
          f"a₂={diag['a2']:.2f}, b₂={diag['b2']:.2f}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)
    best_val = float('inf')

    for epoch in range(args.epochs):
        # -- Train --
        model.train()
        train_losses = []
        for batch in train_ldr:
            Xb, Yb = [b.to(device) for b in batch]
            cos1, sin1, cos2, sin2 = Xb[:, 0], Xb[:, 1], Xb[:, 2], Xb[:, 3]
            thd1, thd2, torque = Xb[:, 4], Xb[:, 5], Xb[:, 6]

            # Enable grads on angle inputs for autograd
            cos1 = cos1.detach().requires_grad_(True)
            sin1 = sin1.detach().requires_grad_(True)
            cos2 = cos2.detach().requires_grad_(True)
            sin2 = sin2.detach().requires_grad_(True)

            thdd1_p, thdd2_p = model(cos1, sin1, cos2, sin2, thd1, thd2, torque)
            thdd1_t, thdd2_t = Yb[:, 0], Yb[:, 1]

            loss = torch.mean((thdd1_p - thdd1_t)**2 + (thdd2_p - thdd2_t)**2)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()

        # -- Val --
        model.eval()
        val_losses = []
        for batch in val_ldr:
            Xb, Yb = [b.to(device) for b in batch]
            cos1, sin1, cos2, sin2 = Xb[:, 0], Xb[:, 1], Xb[:, 2], Xb[:, 3]
            thd1, thd2, torque = Xb[:, 4], Xb[:, 5], Xb[:, 6]
            cos1.requires_grad_(True); sin1.requires_grad_(True)
            cos2.requires_grad_(True); sin2.requires_grad_(True)

            thdd1_p, thdd2_p = model(cos1, sin1, cos2, sin2, thd1, thd2, torque)
            thdd1_t, thdd2_t = Yb[:, 0], Yb[:, 1]
            vloss = torch.mean((thdd1_p - thdd1_t)**2 + (thdd2_p - thdd2_t)**2)
            val_losses.append(vloss.item())

        avg_t = sum(train_losses) / len(train_losses)
        avg_v = sum(val_losses) / len(val_losses)

        if epoch % 50 == 0 or epoch < 5:
            diag = model.true_params()
            print(f"Epoch {epoch:4d} | Train={avg_t:.4f} | Val={avg_v:.4f} | "
                  f"a₁={diag['a1']:.2f} b₁={diag['b1']:.2f} "
                  f"a₂={diag['a2']:.2f} b₂={diag['b2']:.2f}")

        if avg_v < best_val:
            best_val = avg_v
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_loss': avg_v,
                **model.true_params(),
            }, args.save)

    # -- Final --
    diag = model.true_params()
    print(f"\n=== Final ===")
    print(f"Best val loss: {best_val:.6f}")
    print(f"  a₁ = {diag['a1']:.4f}  (true: 3.5)")
    print(f"  b₁ = {diag['b1']:.4f}  (true: 1.0)")
    print(f"  a₂ = {diag['a2']:.4f}  (true: 1.25)")
    print(f"  b₂ = {diag['b2']:.4f}  (true: 0.5)")
    print(f"\nSaved to {args.save}")


if __name__ == '__main__':
    main()
