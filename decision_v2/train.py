"""Train TinyDecisionNet using inverse optimization labels.

Supports: base [4,12,3], hybrid [4,12,3], multi-scale [5,16,3] with --k.

Usage:
  python train.py                                    # base model (default)
  python train.py --model hybrid                     # hybrid model
  python train.py --model ms --k 8                   # multi-scale k=8
"""
import torch, numpy as np, time, os, sys, argparse
from kanrf import KAN
from decision_v2.core import FeatureComputer, TinyDecisionNet


K_VALUES = [1, 2, 4, 8, 16]


def generate_labels(wm_ms, fc, n_samples, device):
    """Try all k ∈ {1,2,4,8,16}, pick best (a, k) per state.  Features from single-scale fc."""
    torch.manual_seed(42)
    np.random.seed(42)
    s_target = torch.tensor([[0., 1., 0.]], device=device)

    theta = np.random.uniform(-np.pi, np.pi, n_samples)
    cos_th, sin_th = np.cos(theta), np.sin(theta)
    thd = np.random.uniform(-8.0, 8.0, n_samples)
    states_raw = np.stack([cos_th, sin_th, thd / 8.0], axis=-1).astype(np.float32)

    all_features, all_actions, all_k = [], [], []

    def inverse_opt_for_k(s, k_val):
        """Find best a for a given k."""
        kn = torch.tensor([[k_val / 16.0]], device=device)
        best_loss, best_a = float('inf'), None
        for _ in range(2):
            a = torch.empty(1, 1, device=device)
            torch.nn.init.uniform_(a, -1, 1)
            a.requires_grad_(True)
            opt = torch.optim.Adam([a], lr=0.05)
            for __ in range(200):
                opt.zero_grad()
                pred = wm_ms(torch.cat([s, a, kn.expand(1, -1)], dim=-1))[:, :3]
                loss = ((pred - s_target) ** 2).sum()
                loss.backward()
                opt.step()
                with torch.no_grad():
                    a.clamp_(-1.0, 1.0)
            with torch.no_grad():
                final = ((wm_ms(torch.cat([s, a, kn.expand(1, -1)], dim=-1))[:, :3]
                         - s_target) ** 2).sum().item()
            if final < best_loss:
                best_loss, best_a = final, a.detach().clone()
        return best_a.item(), best_loss

    for i in range(n_samples):
        s = torch.tensor(states_raw[i:i+1], device=device)

        # Try all k, pick best
        best_k, best_a, best_loss = None, None, float('inf')
        for k_val in K_VALUES:
            a, loss = inverse_opt_for_k(s, k_val)
            if loss < best_loss:
                best_loss, best_a, best_k = loss, a, k_val

        feat = fc.compute_features(s, s_target)
        feat_cpu = {k: v.cpu().squeeze(0) for k, v in feat.items()}
        feat_cpu['s_raw'] = s.cpu().squeeze(0)
        all_features.append(feat_cpu)
        all_actions.append(best_a)
        all_k.append(best_k)

        if (i + 1) % 100 == 0:
            ctrls = [f["ctrl"].item() for f in all_features[-100:]]
            k_counts = {kv: all_k[-100:].count(kv) for kv in K_VALUES}
            print(f'  [{i+1:4d}/{n_samples}]  a∈[{min(all_actions):.2f},{max(all_actions):.2f}]  '
                  f'k_dist={k_counts}')

    X = {
        'a_init': torch.tensor([[f['a_init'].item()] for f in all_features]),
        'gap':    torch.stack([f['gap'] for f in all_features]),
        'align':  torch.tensor([[f['align'].item()] for f in all_features]),
        'ctrl':   torch.tensor([[f['ctrl'].item()] for f in all_features]),
        'trust':  torch.tensor([[f['trust'].item()] for f in all_features]),
        's':      torch.stack([f['s_raw'] for f in all_features]),
    }
    Y = torch.tensor([[a, k / 16.0] for a, k in zip(all_actions, all_k)],
                     dtype=torch.float32)
    return X, Y


def train_decision_net(dn, X, Y, epochs, device):
    """Behavior cloning on (features, (a*, k*)) pairs."""
    feat = {
        'a_init': X['a_init'].to(device), 'gap': X['gap'].to(device),
        'align':  X['align'].to(device), 'ctrl': X['ctrl'].to(device),
        'trust':  X['trust'].to(device),
    }
    s_raw, Y = X['s'].to(device), Y.to(device)

    n_train = int(len(Y) * 0.8)
    idx = torch.randperm(len(Y))
    tr_idx, va_idx = idx[:n_train], idx[n_train:]
    feat_tr = {k: v[tr_idx] for k, v in feat.items()}
    feat_va = {k: v[va_idx] for k, v in feat.items()}
    s_tr, s_va = s_raw[tr_idx], s_raw[va_idx]
    y_tr, y_va = Y[tr_idx], Y[va_idx]

    opt = torch.optim.Adam(dn.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=400, gamma=0.5)
    mse_fn = torch.nn.MSELoss()

    for ep in range(1, epochs + 1):
        dn.train()
        opt.zero_grad()
        pred = dn(feat_tr, s_tr)
        loss = mse_fn(pred[:, 0:1], y_tr[:, 0:1]) + 0.5 * mse_fn(pred[:, 1:2], y_tr[:, 1:2])
        loss.backward()
        opt.step()
        scheduler.step()
        if ep % 200 == 0:
            dn.eval()
            with torch.no_grad():
                pred_v = dn(feat_va, s_va)
                val_a = mse_fn(pred_v[:, 0:1], y_va[:, 0:1]).item()
                val_k = mse_fn(pred_v[:, 1:2], y_va[:, 1:2]).item()
            print(f'  Epoch {ep:4d}  train={loss.item():.4f}  '
                  f'val_a={val_a:.4f}  val_k={val_k:.4f}')

    dn.eval()
    with torch.no_grad():
        pred_v = dn(feat_va, s_va)
        final_a = mse_fn(pred_v[:, 0:1], y_va[:, 0:1]).item()
        final_k = mse_fn(pred_v[:, 1:2], y_va[:, 1:2]).item()
    print(f'Final val_a: {final_a:.4f}  val_k: {final_k:.4f}')
    return dn


def main(n_samples=500, epochs=1000, device='mps', with_k=False):
    if device == 'mps' and not torch.backends.mps.is_available():
        device = 'cpu'
    tag = 'with_k' if with_k else 'base'
    print(f'Device: {device}  Mode: {tag}')

    root = os.path.dirname(os.path.dirname(__file__))

    # Feature computer: always single-scale [4,12,3] — features describe the STATE
    # at k=1 (ctrl small → need larger k, etc.)
    wm_feat = KAN([4, 12, 3], grid_size=5, spline_order=3)
    wm_feat.load_state_dict(torch.load(
        os.path.join(root, 'kan_pendulum_model_v4.pt'), weights_only=True))
    wm_feat.eval().to(device)
    for p in wm_feat.parameters():
        p.requires_grad = False
    fc = FeatureComputer(wm_feat, device=device)

    if with_k:
        # Label generation: multi-scale model tries all k
        wm_ms = KAN([5, 16, 3], grid_size=5, spline_order=3)
        wm_ms.load_state_dict(torch.load(
            os.path.join(root, 'kan_ms.pt'), weights_only=True))
        wm_ms.eval().to(device)
        for p in wm_ms.parameters():
            p.requires_grad = False

        print(f'Generating {n_samples} samples (trying all k)...')
        t0 = time.time()
        X, Y = generate_labels(wm_ms, fc, n_samples, device)
    else:
        # Original: single-scale labels
        print(f'Generating {n_samples} samples...')
        t0 = time.time()
        X, Y = generate_labels_single(wm_feat, fc, n_samples, device)

    print(f'Time: {time.time()-t0:.0f}s  |  {len(Y)} samples')
    print(f'  Y shape: {list(Y.shape)}  a∈[{Y[:,0].min():.2f},{Y[:,0].max():.2f}]')

    if with_k:
        k_vals = (Y[:, 1] * 16).round().int().tolist()
        k_dist = {kv: k_vals.count(kv) for kv in sorted(set(k_vals))}
        print(f'  k distribution: {k_dist}')

    print(f'\nTraining TinyDecisionNet ({epochs} epochs)...')
    dn = TinyDecisionNet(hidden=32, output_k=with_k).to(device)
    dn = train_decision_net(dn, X, Y, epochs, device)

    out_path = os.path.join(os.path.dirname(__file__), f'tiny_dn_{tag}.pt')
    torch.save(dn.state_dict(), out_path)
    print(f'Saved: {out_path}')


def generate_labels_single(wm, fc, n_samples, device):
    """Original: single k=1 labels (for baseline comparison)."""
    torch.manual_seed(42)
    np.random.seed(42)
    s_target = torch.tensor([[0., 1., 0.]], device=device)
    theta = np.random.uniform(-np.pi, np.pi, n_samples)
    cos_th, sin_th = np.cos(theta), np.sin(theta)
    thd = np.random.uniform(-8.0, 8.0, n_samples)
    states_raw = np.stack([cos_th, sin_th, thd / 8.0], axis=-1).astype(np.float32)

    all_features, all_actions = [], []
    for i in range(n_samples):
        s = torch.tensor(states_raw[i:i+1], device=device)
        best_loss, best_a = float('inf'), None
        for _ in range(2):
            a = torch.empty(1, 1, device=device)
            torch.nn.init.uniform_(a, -1, 1)
            a.requires_grad_(True)
            opt = torch.optim.Adam([a], lr=0.05)
            for __ in range(200):
                opt.zero_grad()
                loss = ((wm(torch.cat([s, a], dim=-1))[:, :3] - s_target) ** 2).sum()
                loss.backward()
                opt.step()
                with torch.no_grad():
                    a.clamp_(-1.0, 1.0)
            with torch.no_grad():
                final = ((wm(torch.cat([s, a], dim=-1))[:, :3]
                         - s_target) ** 2).sum().item()
            if final < best_loss:
                best_loss, best_a = final, a.detach().clone()
        feat = fc.compute_features(s, s_target)
        feat_cpu = {k: v.cpu().squeeze(0) for k, v in feat.items()}
        feat_cpu['s_raw'] = s.cpu().squeeze(0)
        all_features.append(feat_cpu)
        all_actions.append(best_a.squeeze().item())
        if (i + 1) % 100 == 0:
            print(f'  [{i+1:4d}/{n_samples}]')

    X = {
        'a_init': torch.tensor([[f['a_init'].item()] for f in all_features]),
        'gap':    torch.stack([f['gap'] for f in all_features]),
        'align':  torch.tensor([[f['align'].item()] for f in all_features]),
        'ctrl':   torch.tensor([[f['ctrl'].item()] for f in all_features]),
        'trust':  torch.tensor([[f['trust'].item()] for f in all_features]),
        's':      torch.stack([f['s_raw'] for f in all_features]),
    }
    Y = torch.tensor([[a] for a in all_actions], dtype=torch.float32)
    return X, Y


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-samples', type=int, default=500)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--device', type=str, default='mps')
    parser.add_argument('--with-k', action='store_true', default=False)
    args = parser.parse_args()
    main(n_samples=args.n_samples, epochs=args.epochs, device=args.device,
         with_k=args.with_k)
