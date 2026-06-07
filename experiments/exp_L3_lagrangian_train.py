#!/usr/bin/env python3
"""
Exp L v3: Lagrangian NN Training - Long Train, No LR Decay
"""
import argparse, sys, os, math, torch
from torch.utils.data import TensorDataset, DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from lagrangian_v2 import LagNet


def load_data(data_path, device='cpu'):
    data = torch.load(data_path, map_location=device)
    sa = data[0]; ns = data[1]
    return (sa[:,0], sa[:,1], sa[:,2]*8.0, sa[:,3]*2.0,
            ns[:,0], ns[:,1], ns[:,2]*8.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='saved_data/pendulum_data_v5_swingup.pt')
    parser.add_argument('--hidden', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch', type=int, default=512)
    parser.add_argument('--lr', type=float, default=5e-3)
    parser.add_argument('--save', type=str, default='saved_models/lagrangian_v3.pt')
    args = parser.parse_args()
    
    device = torch.device('cpu')
    cos_t, sin_t, thd_t, u_t, cos_n, sin_n, thd_n = load_data(args.data, device)
    
    n = len(cos_t)
    n_val = int(n * 0.2)
    idx = torch.randperm(n)
    train_idx, val_idx = idx[n_val:], idx[:n_val]
    
    def mk_loader(ix):
        ds = TensorDataset(cos_t[ix], sin_t[ix], thd_t[ix], u_t[ix], cos_n[ix], sin_n[ix], thd_n[ix])
        return DataLoader(ds, batch_size=args.batch, shuffle=True)
    
    train_ldr = mk_loader(train_idx)
    val_ldr = mk_loader(val_idx)
    
    model = LagNet(hidden=args.hidden).to(device)
    print(f"LagNet v3: hidden={args.hidden}, params={sum(p.numel() for p in model.parameters()):,}, init I={model.I.item():.4f}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_val = float('inf')
    
    for epoch in range(args.epochs):
        # Train
        model.train()
        train_losses = []
        for batch in train_ldr:
            c, s, thd, u, cn, sn, thdn = [b.to(device) for b in batch]
            c = c.detach().requires_grad_(True)
            s = s.detach().requires_grad_(True)
            
            target = (thdn - thd) / 0.05
            pred = model(c, s, thd, u)
            loss = torch.mean((pred - target) ** 2)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            train_losses.append(loss.item())
        
        # Val
        model.eval()
        val_losses = []
        for batch in val_ldr:
            c, s, thd, u, cn, sn, thdn = [b.to(device) for b in batch]
            c.requires_grad_(True); s.requires_grad_(True)
            target = (thdn - thd) / 0.05
            pred = model(c, s, thd, u)
            vloss = torch.mean((pred - target) ** 2)
            val_losses.append(vloss.item())
        
        avg_t = sum(train_losses)/len(train_losses)
        avg_v = sum(val_losses)/len(val_losses)
        
        if epoch % 50 == 0 or epoch < 5:
            print(f"Epoch {epoch:4d} | Train={avg_t:.4f} | Val={avg_v:.4f} | I={model.I.item():.4f}")
        
        if avg_v < best_val:
            best_val = avg_v
            torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch, 'val_loss': avg_v, 'I': model.I.item()}, args.save)
    
    # Final analysis
    print(f"\n=== Final ===")
    print(f"Best val: {best_val:.4f}, I={model.I.item():.4f}")
    
    # Check U values
    model.eval()
    with torch.no_grad():
        angles = torch.linspace(-torch.pi, torch.pi, 100)
        ca = torch.cos(angles).unsqueeze(1)
        sa = torch.sin(angles).unsqueeze(1)
        U_vals = model.net(torch.cat([ca, sa], dim=1)).squeeze()
        print(f"U(θ=0): {U_vals[50].item():.3f}, U(θ=π/2): {U_vals[25].item():.3f}, U(θ=π): {U_vals[0].item():.3f}")
        print(f"Target: U(0)≈−5, U(π/2)≈0, U(π)≈5")
    
    print(f"\nSaved to {args.save}")


if __name__ == '__main__':
    main()
