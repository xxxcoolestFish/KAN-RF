"""Offline pretraining of V(s) via synthetic trajectories from frozen world model.

Batch-parallel trajectory generation on GPU (MPS/CUDA).
Frozen KAN world model → synthetic rollouts → MC regression on V(s).

Supports both single-scale [4,12,3] and multi-scale [5,16,3] world models.
With multi-scale + fixed k, each step covers k*dt, reducing autoregressive
accumulation error by ~k×.
"""
import torch, numpy as np, time, os, sys
from kanrf import KAN
from wm_v_core import MLPValue


def sample_states(n, device='cpu'):
    theta = np.random.uniform(-np.pi, np.pi, n)
    states = np.stack([np.cos(theta), np.sin(theta),
                       np.random.uniform(-8.0, 8.0, n) / 8.0], axis=-1)
    return torch.from_numpy(states.astype(np.float32)).to(device)


def batch_inverse_optimize(model, s_batch, target, k_norm=None, n_iters=30):
    """Batched inverse optimization: find a* for each state in parallel.

    Args:
        s_batch: (B, 3) batch of states
        target:  (1, 3) target state
        k_norm:  (1, 1) or None — fixed timescale to feed to multi-scale model
    Returns:
        a: (B, 1) optimized actions, detached
    """
    B = s_batch.shape[0]
    a = torch.empty(B, 1, device=s_batch.device)
    torch.nn.init.uniform_(a, -0.3, 0.3)
    a.requires_grad_(True)
    opt = torch.optim.Adam([a], lr=0.05)
    tgt = target.expand(B, -1)

    for _ in range(n_iters):
        opt.zero_grad()
        if k_norm is not None:
            kv = k_norm.expand(B, -1)
            sp = model(torch.cat([s_batch, a, kv], dim=-1))[:, :3]
        else:
            sp = model(torch.cat([s_batch, a], dim=-1))[:, :3]
        loss = ((sp - tgt) ** 2).sum()
        loss.backward()
        opt.step()
        with torch.no_grad():
            a.clamp_(-1.0, 1.0)

    return a.detach()


def generate_trajectories(model, s0_all, target, gamma=0.97,
                          max_steps=60, success_thresh=0.2, batch_size=256,
                          k_norm=None):
    """Generate synthetic trajectories in batch-parallel fashion."""
    device = s0_all.device
    X_list, Y_list = [], []

    # Adjust max_steps for large k: fewer steps needed for same time horizon
    # but keep it as-is since the success condition triggers early anyway

    for batch_start in range(0, len(s0_all), batch_size):
        batch_end = min(batch_start + batch_size, len(s0_all))
        B = batch_end - batch_start

        s = s0_all[batch_start:batch_end].clone()
        active = torch.ones(B, dtype=torch.bool, device=device)
        traj_s = [[] for _ in range(B)]
        traj_r = [[] for _ in range(B)]

        for step in range(max_steps):
            if not active.any():
                break

            a = batch_inverse_optimize(model, s, target, k_norm=k_norm, n_iters=30)
            if k_norm is not None:
                x = torch.cat([s, a, k_norm.expand(B, -1)], dim=-1)
            else:
                x = torch.cat([s, a], dim=-1)

            with torch.no_grad():
                s_next = model(x)[:, :3]
            # Unit-circle projection
            norm_cs = torch.sqrt(s_next[:, 0]**2 + s_next[:, 1]**2 + 1e-8)
            s_next[:, 0] = s_next[:, 0] / norm_cs
            s_next[:, 1] = s_next[:, 1] / norm_cs
            s_next[:, 2] = s_next[:, 2].clamp(-1.0, 1.0)
            r = -((s_next - target.expand(B, -1)) ** 2).sum(dim=-1)

            active_idx = torch.where(active)[0]
            for i in active_idx:
                i_int = i.item()
                traj_s[i_int].append(s[i_int].cpu())
                traj_r[i_int].append(r[i_int].item())

            dist = (s_next - target.expand(B, -1)).norm(dim=-1)
            reached = dist < success_thresh
            if reached.any():
                reached_idx = torch.where(reached & active)[0]
                for i in reached_idx:
                    i_int = i.item()
                    traj_s[i_int].append(s_next[i_int].cpu())
                    traj_r[i_int].append(0.0)
                active = active & ~reached

            s = s_next

        for i in range(B):
            G = 0.0
            for t in reversed(range(len(traj_r[i]))):
                G = traj_r[i][t] + gamma * G
                X_list.append(traj_s[i][t])
                Y_list.append(G)

        if (batch_end) % 500 == 0 or batch_start == 0:
            print(f'  [{batch_end:5d}/{len(s0_all)}]  '
                  f'collected={len(X_list)} samples')

    X = torch.stack(X_list)
    Y = torch.tensor(Y_list, dtype=torch.float32).unsqueeze(1)
    return X, Y


def load_world_model(model_path, multi_scale=False):
    """Load KAN world model.  Infers architecture from state_dict."""
    ckpt = torch.load(model_path, weights_only=True)
    # Infer layer dims from saved weights
    # First layer's base_weight has shape (out_dim, in_dim)
    dims = [ckpt['layers.0.base_weight'].shape[1]]  # input dim
    i = 0
    while f'layers.{i}.base_weight' in ckpt:
        bw = ckpt[f'layers.{i}.base_weight']
        dims.append(bw.shape[0])  # output dim
        i += 1
    model = KAN(dims, grid_size=5, spline_order=3)
    model.load_state_dict(ckpt)
    return model.eval()


def main(n_states=5000, gamma=0.97, lr=1e-3, epochs=500,
         device='mps', model_path=None, k_val=None):
    torch.manual_seed(42)
    np.random.seed(42)
    t0 = time.time()

    if device == 'mps' and not torch.backends.mps.is_available():
        print('MPS not available, falling back to CPU')
        device = 'cpu'
    print(f'Device: {device}')

    # Model path resolution
    root = os.path.dirname(os.path.dirname(__file__))
    if model_path is None:
        if k_val is not None:
            model_path = os.path.join(root, 'kan_ms.pt')
        else:
            model_path = os.path.join(root, 'kan_pendulum_model_v4.pt')

    wm = load_world_model(model_path).to(device)
    for p in wm.parameters():
        p.requires_grad = False
    layer_dims = [wm.layers[0].in_dim] + [l.out_dim for l in wm.layers]
    print(f'World model: {layer_dims}  params={sum(p.numel() for p in wm.parameters())}')

    # Multi-scale: fixed k_norm = k/16, appended as extra input dim
    k_norm = None
    if k_val is not None:
        k_norm = torch.tensor([[k_val / 16.0]], device=device)
        print(f'Multi-scale: k={k_val} fixed  (k_norm={k_val/16:.3f})')

    target = torch.tensor([[0., 1., 0.]], device=device)

    # Phase 1: Generate trajectories
    print(f'\n=== Phase 1: Trajectory generation ({n_states} states) ===')
    s0 = sample_states(n_states, device=device)
    X, Y = generate_trajectories(wm, s0, target, gamma=gamma,
                                 max_steps=60, success_thresh=0.2,
                                 batch_size=256, k_norm=k_norm)
    print(f'Total: {len(X)} (state, G) pairs')
    print(f'  G range: [{Y.min().item():.3f}, {Y.max().item():.3f}]')
    print(f'  Time: {time.time()-t0:.0f}s')

    # Phase 2: Train V(s)
    print(f'\n=== Phase 2: V(s) regression ({epochs} epochs) ===')
    n_train = int(len(X) * 0.85)
    idx = torch.randperm(len(X))
    X_tr, Y_tr = X[idx[:n_train]].to(device), Y[idx[:n_train]].to(device)
    X_va, Y_va = X[idx[n_train:]].to(device), Y[idx[n_train:]].to(device)

    V = MLPValue(3).to(device)
    opt = torch.optim.Adam(V.parameters(), lr=lr)

    for ep in range(1, epochs + 1):
        V.train()
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(V(X_tr), Y_tr)
        loss.backward()
        opt.step()

        if ep % 100 == 0:
            V.eval()
            with torch.no_grad():
                val_loss = torch.nn.functional.mse_loss(V(X_va), Y_va).item()
                ts = sample_states(100, device=device)
                vv = V(ts)
            print(f'  Epoch {ep:4d}  train={loss.item():.5f}  val={val_loss:.5f}  '
                  f'V_range=[{vv.min():.3f}, {vv.max():.3f}]')

    V.eval()
    with torch.no_grad():
        val_loss = torch.nn.functional.mse_loss(V(X_va), Y_va).item()
    print(f'Final val_loss: {val_loss:.5f}  |  total time: {time.time()-t0:.0f}s')

    # Save
    tag = f'_k{k_val}' if k_val else ''
    out_path = os.path.join(os.path.dirname(__file__), f'v_pretrained{tag}.pt')
    torch.save(V.state_dict(), out_path)
    print(f'Saved: {out_path}')

    # Sanity check
    test_states = torch.tensor([
        [-1., 0., 0.],       # bottom, v=0
        [0., 1., 0.],        # upright
        [-0.5, 0.866, 0.],   # mid-up
        [-1., 0., 0.5],      # bottom, v>0
        [0.5, 0.866, 0.],    # near upright
    ], device=device)
    with torch.no_grad():
        v_test = V(test_states)
    print('\nSanity check:')
    for lbl, v in zip(['bottom v=0', 'upright', 'mid-up', 'bottom v>0', 'near upright'],
                       v_test.squeeze(1).tolist()):
        print(f'  {lbl:15s}: V={v:.4f}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-states', type=int, default=5000)
    parser.add_argument('--gamma', type=float, default=0.97)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--device', type=str, default='mps')
    parser.add_argument('--k', type=int, default=None,
                        help='Fixed k for multi-scale model (1,2,4,8,16)')
    args = parser.parse_args()
    main(n_states=args.n_states, gamma=args.gamma, lr=args.lr,
         epochs=args.epochs, device=args.device, k_val=args.k)
