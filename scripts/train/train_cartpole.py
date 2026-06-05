"""Train multi-scale CartPole world model.

f(s_t, a_onehot, k) -> s_{t+k}   where s ∈ R^4, a ∈ {0,1}

Architecture: [7, 20, 4] (state=4 + action_onehot=2 + k_norm=1 = 7D)
"""
import torch, argparse, time
from kanrf import KAN


def p_spline_penalty(model):
    total = 0.0
    for layer in model.layers:
        c = layer.spline_weight
        d2 = c[:, :, :-2] - 2 * c[:, :, 1:-1] + c[:, :, 2:]
        total += (d2 ** 2).mean()
    return total


def main(lam=0.1):
    torch.manual_seed(42)

    x, y = torch.load('cartpole_data_ms.pt', weights_only=True)
    # Subsample for speed: CartPole is simpler than Pendulum
    n_total = min(len(x), 50000)
    idx = torch.randperm(len(x))[:n_total]
    x, y = x[idx], y[idx]

    n_train = int(len(x) * 0.85)
    x_train, y_train = x[:n_train], y[:n_train]
    x_val, y_val = x[n_train:], y[n_train:]

    model = KAN([7, 20, 4], grid_size=5, spline_order=3)
    print(f'CartPole world model: [7,20,4]  '
          f'params={sum(p.numel() for p in model.parameters())}  lam={lam}')
    print(f'Train: {n_train}, Val: {len(x_val)}')

    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=500, gamma=0.5)
    mse_fn = torch.nn.MSELoss()
    t_start = time.time()

    for epoch in range(1, 1201):
        model.train()
        opt.zero_grad()
        pred = model(x_train)
        loss = mse_fn(pred, y_train)
        if lam > 0:
            loss = loss + lam * p_spline_penalty(model)
        loss.backward()
        opt.step()
        scheduler.step()

        if epoch % 200 == 0:
            model.eval()
            with torch.no_grad():
                val_mse = mse_fn(model(x_val), y_val).item()
            elapsed = time.time() - t_start
            eta = elapsed / epoch * (1200 - epoch)
            print(f'Epoch {epoch:4d}  lr={scheduler.get_last_lr()[0]:.4f}  '
                  f'train={loss.item():.6f}  val={val_mse:.6f}  '
                  f'[{elapsed:.0f}s ETA {eta:.0f}s]')

    model.eval()
    with torch.no_grad():
        val_mse = mse_fn(model(x_val), y_val).item()
        for k_val in [1, 2, 4, 8, 16]:
            mask = (x_val[:, -1] * 16).round() == k_val
            if mask.sum() > 0:
                k_mse = mse_fn(model(x_val[mask]), y_val[mask]).item()
                print(f'  k={k_val:2d}: val_mse={k_mse:.6f} (n={mask.sum().item()})')

    print(f'Final val MSE: {val_mse:.6f}  |  {time.time()-t_start:.0f}s')
    torch.save(model.state_dict(), 'kan_cartpole.pt')
    print('Saved: kan_cartpole.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lam', type=float, default=0.1)
    args = parser.parse_args()
    main(lam=args.lam)
