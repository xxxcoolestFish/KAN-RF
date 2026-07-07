"""Continual learning: ProtoKAN vs KAN WM adaptation after physics change.

The core question: when Pendulum gravity changes (10→15), can ProtoKAN WM
adapt quickly via online learning without forgetting old knowledge?

This is the ORIGINAL goal of using KAN — B-spline local support enables
fast, local adaptation. We test if ProtoKAN's Gaussian kernel preserves this.
"""
import torch, torch.nn as nn, numpy as np, time, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kanrf import ProtoKAN, KAN


def pendulum_step(s, a_norm, g=10.0):
    """Pendulum dynamics: s=[cos,sin,thd/8], a_norm ∈ [-1,1] → torque ∈ [-2,2]."""
    th = torch.atan2(s[:, 1], s[:, 0])
    thd = s[:, 2] * 8.0
    u = a_norm.squeeze(-1) * 2.0  # ensure (B,) shape
    DT = 0.05
    thd_new = thd + (g * torch.sin(th) + u) * DT
    th_new = th + thd_new * DT
    return torch.stack([torch.cos(th_new), torch.sin(th_new), thd_new / 8.0], dim=-1)


def generate_data(g=10.0, n=5000, device='cpu'):
    """Generate (s,a,s') triplets with given gravity."""
    xs, ys = [], []
    for _ in range(n):
        th = np.random.uniform(-np.pi, np.pi)
        thd = np.random.uniform(-8.0, 8.0)
        a = np.random.uniform(-1.0, 1.0)
        s = torch.tensor([[np.cos(th), np.sin(th), thd / 8.0]], dtype=torch.float32)
        a_t = torch.tensor([[a]], dtype=torch.float32)
        s_next = pendulum_step(s, a_t, g=g)
        xs.append(torch.cat([s, a_t], dim=-1))
        ys.append(s_next)
    return (torch.cat(xs, dim=0).to(device),
            torch.cat(ys, dim=0).to(device))


def train_wm_lbfgs(X, Y, wm_type='protokan', n_proto=16, n_iters=100, device='cpu',
                   init_sigma=-1.5):
    """Train WM with L-BFGS."""
    n_tr = int(len(X) * 0.85)
    X_tr, Y_tr = X[:n_tr], Y[:n_tr]
    X_val, Y_val = X[n_tr:], Y[n_tr:]

    if wm_type == 'protokan':
        wm = ProtoKAN([4, 8, 3], n_prototypes=n_proto).to(device)
        # Initialize with smaller sigma for better locality
        for layer in wm.layers:
            layer.log_sigma.data.fill_(init_sigma)
    else:
        wm = KAN([4, 8, 3], grid_size=5, spline_order=3).to(device)

    mse_fn = nn.MSELoss()
    best_val = float('inf')
    best_state = None

    def closure():
        opt.zero_grad()
        loss = mse_fn(wm(X_tr), Y_tr)
        loss.backward()
        return loss

    opt = torch.optim.LBFGS(wm.parameters(), lr=1.0, max_iter=20,
                            history_size=50, line_search_fn='strong_wolfe')
    for step in range(1, n_iters + 1):
        opt.step(closure)
        with torch.no_grad():
            val = mse_fn(wm(X_val), Y_val).item()
        if val < best_val:
            best_val = val
            best_state = {k: v.clone() for k, v in wm.state_dict().items()}
    wm.load_state_dict(best_state)
    wm.eval()
    return wm, best_val


def continual_adapt(wm, X_old, Y_old, X_new, Y_new, n_steps=30, lr=1e-3,
                    device='cpu', label=''):
    """Online adaptation: learn new physics while tracking old knowledge."""
    mse_fn = nn.MSELoss()
    opt = torch.optim.Adam(wm.parameters(), lr=lr)

    new_errors = []
    old_errors = []

    for step in range(1, n_steps + 1):
        # One step on new data
        idx = np.random.choice(len(X_new), min(64, len(X_new)), replace=False)
        xb, yb = X_new[idx], Y_new[idx]
        wm.train(); opt.zero_grad()
        loss = mse_fn(wm(xb), yb)
        loss.backward()
        opt.step()
        wm.eval()

        # Track errors
        with torch.no_grad():
            new_err = mse_fn(wm(X_new), Y_new).item()
            old_err = mse_fn(wm(X_old), Y_old).item()
        new_errors.append(new_err)
        old_errors.append(old_err)

        if step % 10 == 0 or step == 1:
            print(f"  [{label}] step {step:2d}  new_mse={new_err:.6f}  old_mse={old_err:.6f}")

    return new_errors, old_errors


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--init_sigma', type=float, default=-1.5)
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(42); np.random.seed(42)

    print("=" * 70)
    print("Continual Learning: ProtoKAN vs KAN after Gravity Change (10 → 15)")
    print("=" * 70)

    # ── 1. Generate data ──
    print("\n[1] Generating data...")
    X_old, Y_old = generate_data(g=10.0, n=3000, device=device)
    X_new, Y_new = generate_data(g=15.0, n=500, device=device)  # few samples for adaptation
    print(f"  Old (g=10): {X_old.shape[0]} samples")
    print(f"  New (g=15): {X_new.shape[0]} samples")

    # ── 2. Train ProtoKAN WM on old physics ──
    print("\n[2] Training ProtoKAN WM on g=10...")
    t0 = time.time()
    proto_wm, proto_val = train_wm_lbfgs(X_old, Y_old, 'protokan', 16, 100,
                                          device, args.init_sigma)
    sigma_val = [torch.exp(l.log_sigma).item() for l in proto_wm.layers]
    print(f"  val_mse={proto_val:.6f}  sigma={sigma_val}  time={time.time()-t0:.0f}s")

    # ── 3. Train KAN WM on old physics ──
    print("\n[3] Training KAN WM on g=10...")
    t0 = time.time()
    kan_wm, kan_val = train_wm_lbfgs(X_old, Y_old, 'kan', 16, 100, device)
    print(f"  val_mse={kan_val:.6f}  time={time.time()-t0:.0f}s")

    # ── 4. Check initial error on NEW physics ──
    print("\n[4] Errors BEFORE adaptation:")
    mse_fn = nn.MSELoss()
    with torch.no_grad():
        proto_new = mse_fn(proto_wm(X_new), Y_new).item()
        kan_new = mse_fn(kan_wm(X_new), Y_new).item()
        proto_old_ref = mse_fn(proto_wm(X_old[:500]), Y_old[:500]).item()
        kan_old_ref = mse_fn(kan_wm(X_old[:500]), Y_old[:500]).item()
    print(f"  ProtoKAN: new_mse={proto_new:.6f}  old_mse={proto_old_ref:.6f}")
    print(f"  KAN:      new_mse={kan_new:.6f}  old_mse={kan_old_ref:.6f}")

    # ── 5. Continual adaptation ──
    print("\n[5] ProtoKAN adaptation (online, g=15)...")
    proto_new_errs, proto_old_errs = continual_adapt(
        proto_wm, X_old[:1000], Y_old[:1000], X_new, Y_new,
        n_steps=30, lr=1e-3, device=device, label='ProtoKAN')

    print("\n[6] KAN adaptation (online, g=15)...")
    kan_new_errs, kan_old_errs = continual_adapt(
        kan_wm, X_old[:1000], Y_old[:1000], X_new, Y_new,
        n_steps=30, lr=1e-3, device=device, label='KAN')

    # ── 6. Report ──
    print("\n" + "=" * 70)
    print("CONTINUAL LEARNING RESULTS")
    print("=" * 70)

    proto_new_start = proto_new_errs[0]
    proto_new_end = proto_new_errs[-1]
    proto_old_start = proto_old_errs[0]
    proto_old_end = proto_old_errs[-1]

    kan_new_start = kan_new_errs[0]
    kan_new_end = kan_new_errs[-1]
    kan_old_start = kan_old_errs[0]
    kan_old_end = kan_old_errs[-1]

    print(f"  {'':20s}  {'ProtoKAN':>12s}  {'KAN':>12s}")
    print(f"  {'New MSE start':20s}  {proto_new_start:12.6f}  {kan_new_start:12.6f}")
    print(f"  {'New MSE end':20s}  {proto_new_end:12.6f}  {kan_new_end:12.6f}")
    print(f"  {'New MSE recovery':20s}  {proto_new_start/proto_new_end:11.2f}x  "
          f"{kan_new_start/kan_new_end:11.2f}x")
    print(f"  {'Old MSE start':20s}  {proto_old_start:12.6f}  {kan_old_start:12.6f}")
    print(f"  {'Old MSE end':20s}  {proto_old_end:12.6f}  {kan_old_end:12.6f}")
    print(f"  {'Forgetting ratio':20s}  {proto_old_end/proto_old_start:11.4f}  "
          f"{kan_old_end/kan_old_start:11.4f}")
    print(f"  (ratio > 1 = forgetting, ratio ≈ 1 = preserved)")


if __name__ == '__main__':
    main()
