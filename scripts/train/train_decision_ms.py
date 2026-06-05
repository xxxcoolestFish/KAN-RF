"""Train multi-scale decision network: (s, s_target) → (a, k_cont).

KAN [6, 12, 2] — outputs continuous action and timescale.
Loss: MSE(a) + α * MSE(k)
"""
import torch, argparse, time
from kanrf import KAN


def main(alpha=0.5, lr=1e-2, n_epochs=2000, device='cpu'):
    torch.manual_seed(42)

    data = torch.load('decision_data_ms.pt', weights_only=True)
    s = data['s_norm'].to(device)
    s_tgt = data['s_target_norm'].to(device)
    a_label = data['a_norm'].to(device)
    k_label = data['k_norm_cont'].to(device)

    n_train = int(len(s) * 0.8)
    idx = torch.randperm(len(s))
    s_tr, s_tgt_tr, a_tr, k_tr = (t[idx[:n_train]] for t in [s, s_tgt, a_label, k_label])
    s_va, s_tgt_va, a_va, k_va = (t[idx[n_train:]] for t in [s, s_tgt, a_label, k_label])

    model = KAN([6, 12, 2], grid_size=5, spline_order=3).to(device)
    print(f"Decision KAN: [6, 12, 2]  params={sum(p.numel() for p in model.parameters())}")
    print(f"Train: {n_train}, Val: {len(s)-n_train}")

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=600, gamma=0.5)
    mse_fn = torch.nn.MSELoss()

    t_start = time.time()
    for epoch in range(1, n_epochs + 1):
        model.train()
        opt.zero_grad()
        out = model(torch.cat([s_tr, s_tgt_tr], dim=-1))
        loss = mse_fn(out[:, 0:1], a_tr) + alpha * mse_fn(out[:, 1:2], k_tr)
        loss.backward()
        opt.step()
        scheduler.step()

        if epoch % 400 == 0:
            model.eval()
            with torch.no_grad():
                out_v = model(torch.cat([s_va, s_tgt_va], dim=-1))
                loss_v = mse_fn(out_v[:, 0:1], a_va) + alpha * mse_fn(out_v[:, 1:2], k_va)
                a_rmse = mse_fn(out_v[:, 0:1], a_va).sqrt().item()
                k_rmse = mse_fn(out_v[:, 1:2], k_va).sqrt().item()
            elapsed = time.time() - t_start
            eta = elapsed / epoch * (n_epochs - epoch)
            print(f"Epoch {epoch:4d}  lr={scheduler.get_last_lr()[0]:.4f}  "
                  f"train={loss.item():.4f}  val={loss_v.item():.4f}  "
                  f"a_rmse={a_rmse:.4f}  k_rmse={k_rmse:.4f}  "
                  f"[{elapsed:.0f}s ETA {eta:.0f}s]")

    torch.save(model.state_dict(), 'kan_decision_ms.pt')
    print(f"Saved: kan_decision_ms.pt  |  time: {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    main(alpha=args.alpha, lr=args.lr, n_epochs=args.epochs, device=args.device)
