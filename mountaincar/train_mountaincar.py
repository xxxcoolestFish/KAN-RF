"""Train MountainCar multi-scale world model: KAN([6, 16, 2]) with MOPS."""
import torch, argparse, time, sys, os
from kanrf import KAN


def p_spline_penalty(model):
    total = 0.0
    for layer in model.layers:
        c = layer.spline_weight
        d2 = c[:, :, :-2] - 2 * c[:, :, 1:-1] + c[:, :, 2:]
        total += (d2 ** 2).mean()
    return total


def main(lam=0.1, device_str='mps'):
    torch.manual_seed(42)
    device = torch.device(device_str) if torch.backends.mps.is_available() \
        else torch.device('cpu')
    print(f'Device: {device}')

    x, y = torch.load('mountaincar_data_ms.pt', weights_only=True)
    x, y = x.float().to(device), y.float().to(device)
    n_train = int(len(x) * 0.85)
    x_train, y_train = x[:n_train], y[:n_train]
    x_val, y_val = x[n_train:], y[n_train:]

    model = KAN([6, 16, 2], grid_size=5, spline_order=3).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'MountainCar WM: [6,16,2]  params={n_params}  lam={lam}')
    print(f'Train: {len(x_train)}, Val: {len(x_val)}')

    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=400, gamma=0.5)
    mse_fn = torch.nn.MSELoss()
    t_start = time.time()

    for epoch in range(1, 801):
        model.train(); opt.zero_grad()
        loss = mse_fn(model(x_train), y_train)
        if lam > 0: loss = loss + lam * p_spline_penalty(model)
        loss.backward(); opt.step(); scheduler.step()

        if epoch % 200 == 0:
            model.eval()
            with torch.no_grad():
                val_mse = mse_fn(model(x_val), y_val).item()
            print(f'Epoch {epoch:3d}  lr={scheduler.get_last_lr()[0]:.4f}  '
                  f'train={loss.item():.6f}  val={val_mse:.6f}  '
                  f'[{time.time()-t_start:.0f}s]')

    model.eval()
    with torch.no_grad():
        val_mse = mse_fn(model(x_val), y_val).item()
        print(f'\nFinal val MSE: {val_mse:.6f}')
        for k_val in [1, 2, 4]:
            mask = (x_val[:, 5] * 4).round() == k_val
            if mask.sum() > 0:
                k_mse = mse_fn(model(x_val[mask]), y_val[mask]).item()
                perr = (model(x_val[mask]) - y_val[mask]).norm(dim=-1)
                print(f'  k={k_val}: mse={k_mse:.6f}  '
                      f'mean_err={perr.mean():.4f}  p90={perr.quantile(0.9):.4f}  '
                      f'p99={perr.quantile(0.99):.4f}  (n={mask.sum().item()})')

    torch.save(model.state_dict(), 'mountaincar_wm.pt')
    print(f'Saved: mountaincar_wm.pt  |  {time.time()-t_start:.0f}s')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lam', type=float, default=0.1)
    parser.add_argument('--device', type=str, default='mps')
    args = parser.parse_args()
    main(lam=args.lam, device_str=args.device)
