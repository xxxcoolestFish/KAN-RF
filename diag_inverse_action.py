"""Diagnostic: inverse action accuracy of KAN world model.

Tests: given (s, s'_true), can gradient descent through frozen KAN recover a_true?

The pipeline uses this inverse optimization for decision-making, so its accuracy
directly bounds control quality — regardless of forward prediction MSE.
"""

import sys, time, argparse, os
import torch
import numpy as np
from kanrf import KAN

G = 10.0


def load_model(path, device):
    ckpt = torch.load(path, weights_only=True)
    layer_dims = [4]
    for key in sorted(ckpt.keys()):
        if 'base_weight' in key:
            layer_dims.append(ckpt[key].shape[0])
    model = KAN(layer_dims, grid_size=5, spline_order=3)
    model.load_state_dict(ckpt)
    model.to(device)
    model.eval()
    return model


def inverse_optimize(model, s_norm, s_target_norm, n_iters=50, lr=0.05, device=None):
    """Find a* = argmin ||f(s,a) - s_target||^2 via Adam gradient descent.

    Returns:
        a_star: optimized action (raw, in [-2, 2])
        s_pred: model-predicted next state from a*
        trace: list of (iter, loss, a) during optimization
    """
    a_n = torch.zeros(1, 1, device=device)
    torch.nn.init.uniform_(a_n, a=-0.3, b=0.3)
    a_n.requires_grad_(True)
    opt = torch.optim.Adam([a_n], lr=lr)
    trace = []

    for i in range(n_iters):
        opt.zero_grad()
        x = torch.cat([s_norm, a_n], dim=-1)
        s_pred = model(x)
        loss = ((s_pred - s_target_norm) ** 2).sum()
        loss.backward()
        opt.step()
        with torch.no_grad():
            a_n.clamp_(-1.0, 1.0)
        if i % 10 == 0 or i == n_iters - 1:
            trace.append((i, loss.item(), a_n.detach().item() * 2.0))

    with torch.no_grad():
        s_pred = model(torch.cat([s_norm, a_n], dim=-1))
    a_star = a_n.detach().item() * 2.0
    return a_star, s_pred, trace


def run_diagnostics(model, x_test, y_test, device, n_samples=2000, n_iters=50):
    """Main diagnostic: compare forward vs inverse accuracy."""
    model.eval()
    n = min(n_samples, len(x_test))
    idx = torch.randperm(len(x_test))[:n]
    x, y = x_test[idx].to(device), y_test[idx].to(device)

    results = {
        'forward_err': [],       # ||f(s,a_true) - s'_true||  (L2 in normed space)
        'inverse_action_err': [], # |a* - a_true|              (raw torque)
        'inverse_state_err': [],  # ||f(s,a*) - s'_true||      (L2 in normed space)
        'opt_loss_final': [],    # final optimization loss
        'a_true': [],
        'a_star': [],
        's_sin': [],             # sin(theta) of current state
        's_thd': [],             # theta_dot of current state
    }

    for i in range(n):
        s_norm = x[i:i+1, :3]        # (1, 3)
        a_true_norm = x[i:i+1, 3:4]  # (1, 1)
        s_target = y[i:i+1]          # (1, 3)
        a_true_raw = a_true_norm.item() * 2.0

        with torch.no_grad():
            s_pred_forward = model(torch.cat([s_norm, a_true_norm], dim=-1))
            fwd_err = (s_pred_forward - s_target).norm().item()

        a_star, s_pred_inv, trace = inverse_optimize(
            model, s_norm, s_target, n_iters=n_iters, lr=0.05, device=device)

        inv_state_err = (s_pred_inv - s_target).norm().item()
        inv_action_err = abs(a_star - a_true_raw)

        results['forward_err'].append(fwd_err)
        results['inverse_action_err'].append(inv_action_err)
        results['inverse_state_err'].append(inv_state_err)
        results['opt_loss_final'].append(trace[-1][1])
        results['a_true'].append(a_true_raw)
        results['a_star'].append(a_star)

        # Raw state (denormalize)
        results['s_sin'].append(s_norm[0, 1].item())
        results['s_thd'].append(s_norm[0, 2].item() * 8.0)

    return {k: np.array(v) for k, v in results.items()}


def print_report(r):
    """Print diagnostic report."""
    fwd = r['forward_err']
    inv_a = r['inverse_action_err']
    inv_s = r['inverse_state_err']

    print(f"\n{'='*70}")
    print(f"INVERSE ACTION DIAGNOSTIC  (n={len(fwd)} samples)")
    print(f"{'='*70}")

    print(f"\n  Forward prediction (s,a_true):")
    print(f"    ||f(s,a_true) - s'_true||  mean={fwd.mean():.4f}  "
          f"median={np.median(fwd):.4f}  max={fwd.max():.4f}")

    print(f"\n  Inverse optimization (s, s'_target → a*):")
    print(f"    |a* - a_true| (raw torque)  mean={inv_a.mean():.4f}  "
          f"median={np.median(inv_a):.4f}  max={inv_a.max():.4f}")
    print(f"    ||f(s,a*) - s'_true||        mean={inv_s.mean():.4f}  "
          f"median={np.median(inv_s):.4f}  max={inv_s.max():.4f}")

    # Key ratio
    ratio = inv_a.mean() / (fwd.mean() + 1e-8)
    print(f"\n  Inverse/Forward error ratio: {ratio:.1f}x")

    # By region
    sin = r['s_sin']
    thd_abs = np.abs(r['s_thd'])
    bottom = sin < -0.3
    mid = (sin >= -0.3) & (sin <= 0.5)
    top = sin > 0.5

    print(f"\n  By region:")
    for label, mask in [('Bottom (sin<-0.3)', bottom), ('Mid (-0.3<sin<0.5)', mid),
                         ('Top (sin>0.5)', top)]:
        if mask.sum() > 0:
            print(f"    {label:>22s}: n={mask.sum():4d}  "
                  f"|a*-a|={inv_a[mask].mean():.3f}  "
                  f"fwd={fwd[mask].mean():.4f}  inv_s={inv_s[mask].mean():.4f}")

    # By action magnitude
    a_abs = np.abs(r['a_true'])
    small_a = a_abs < 0.5
    med_a = (a_abs >= 0.5) & (a_abs <= 1.5)
    large_a = a_abs > 1.5
    print(f"\n  By |a_true|:")
    for label, mask in [('|a|<0.5', small_a), ('0.5<|a|<1.5', med_a),
                         ('|a|>1.5', large_a)]:
        if mask.sum() > 0:
            print(f"    {label:>15s}: n={mask.sum():4d}  "
                  f"|a*-a|={inv_a[mask].mean():.3f}  "
                  f"a_err/a_true={inv_a[mask].mean()/(a_abs[mask].mean()+1e-8):.2f}")

    # Scatter stats
    corr = np.corrcoef(r['a_true'], r['a_star'])[0, 1]
    print(f"\n  Correlation(a_true, a_star): {corr:.4f}")

    print(f"\n  Optimization loss: mean={r['opt_loss_final'].mean():.6f}  "
          f"max={r['opt_loss_final'].max():.6f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='kan_pendulum_model_v6.pt')
    parser.add_argument('--data', type=str, default='pendulum_data_v4.pt')
    parser.add_argument('--n-samples', type=int, default=2000)
    parser.add_argument('--n-iters', type=int, default=50)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--plot', action='store_true', default=True)
    args = parser.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)

    device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"Model: {args.model}  |  Data: {args.data}")

    model = load_model(args.model, device)
    print(f"  Layers: {[4] + [l.out_dim for l in model.layers]}  "
          f"Params: {sum(p.numel() for p in model.parameters())}")

    data = torch.load(args.data, weights_only=True)
    x_test, y_test = data if isinstance(data, tuple) else (data[0], data[1])

    # Use last 20% as test split (not seen during training)
    n_test = len(x_test) // 5
    x_test, y_test = x_test[-n_test:], y_test[-n_test:]
    print(f"  Test samples: {len(x_test)} (last 20% of data)")

    t0 = time.time()
    r = run_diagnostics(model, x_test, y_test, device,
                        n_samples=args.n_samples, n_iters=args.n_iters)
    print(f"  Time: {time.time()-t0:.0f}s")

    print_report(r)

    if args.plot:
        _plot(r, args)


def _plot(r, args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    a_true, a_star = r['a_true'], r['a_star']
    inv_a_err = r['inverse_action_err']
    fwd_err = r['forward_err']
    inv_s_err = r['inverse_state_err']
    sin = r['s_sin']

    # 1. a* vs a_true scatter
    ax = axes[0, 0]
    ax.scatter(a_true, a_star, c=sin, cmap='coolwarm', s=3, alpha=0.5)
    ax.plot([-2, 2], [-2, 2], 'k--', linewidth=1)
    ax.set_xlabel('a_true'); ax.set_ylabel('a*')
    ax.set_title(f'a* vs a_true  (r={np.corrcoef(a_true, a_star)[0,1]:.3f})')
    ax.set_xlim(-2.2, 2.2); ax.set_ylim(-2.2, 2.2)

    # 2. |a* - a_true| histogram
    ax = axes[0, 1]
    ax.hist(inv_a_err, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
    ax.axvline(inv_a_err.mean(), color='red', linestyle='--', label=f'mean={inv_a_err.mean():.3f}')
    ax.set_xlabel('|a* - a_true|'); ax.set_ylabel('count')
    ax.set_title('Inverse Action Error Distribution')
    ax.legend()

    # 3. forward vs inverse state error
    ax = axes[0, 2]
    ax.scatter(fwd_err, inv_s_err, c=sin, cmap='coolwarm', s=3, alpha=0.5)
    lim = max(fwd_err.max(), inv_s_err.max()) * 1.1
    ax.plot([0, lim], [0, lim], 'k--', linewidth=1)
    ax.set_xlabel('Forward err ||f(s,a_true)-s\'||'); ax.set_ylabel('Inverse err ||f(s,a*)-s\'||')
    ax.set_title('Forward vs Inverse State Error')

    # 4. action error vs sin(theta)
    ax = axes[1, 0]
    ax.scatter(sin, inv_a_err, c=np.abs(r['s_thd']), cmap='viridis', s=3, alpha=0.5)
    ax.set_xlabel('sin(theta)'); ax.set_ylabel('|a* - a_true|')
    ax.set_title('Action Error vs State (color=|thd|)')
    ax.axvline(-0.3, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(0.5, color='gray', linestyle=':', alpha=0.5)

    # 5. action error vs |a_true|
    ax = axes[1, 1]
    ax.scatter(np.abs(a_true), inv_a_err, c=sin, cmap='coolwarm', s=3, alpha=0.5)
    ax.set_xlabel('|a_true|'); ax.set_ylabel('|a* - a_true|')
    ax.set_title('Action Error vs True Action Magnitude')

    # 6. action error vs forward error
    ax = axes[1, 2]
    ax.scatter(fwd_err, inv_a_err, c=sin, cmap='coolwarm', s=3, alpha=0.5)
    ax.set_xlabel('Forward error'); ax.set_ylabel('|a* - a_true|')
    ax.set_title('Action Error vs Forward Prediction Error')

    plt.tight_layout()
    out = 'diag_inverse_action.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\n  Plot saved: {out}")


if __name__ == "__main__":
    main()
