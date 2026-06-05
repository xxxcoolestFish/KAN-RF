"""Train CartPole decision network: (state_4d) -> (action_logit, k_cont).

Generates labels by trying all (action, k) pairs through the world model,
picking the one that predicts the most balanced state.

Decision net: KAN([4, 12, 2]) — same as Pendulum but input is 4D (no target).
"""
import torch, numpy as np, argparse
from kanrf import KAN

K_VALS = [1, 2, 4, 8, 16]


def load_wm(path='kan_cartpole.pt'):
    model = KAN([7, 20, 4], grid_size=5, spline_order=3)
    model.load_state_dict(torch.load(path, weights_only=True))
    return model.eval()


def score_state(s_norm):
    return abs(s_norm[:, 2]) * 0.5 + abs(s_norm[:, 0]) * 0.2


def generate_labels(wm, n_samples=2000):
    torch.manual_seed(42)
    states = torch.rand(n_samples, 4) * 2 - 1  # normalized [-1, 1]
    a_oh = torch.zeros(1, 2)
    samples = []

    for i in range(n_samples):
        sn = states[i:i+1]
        best_a, best_k, best_score = 0, 1, float('inf')
        for k in K_VALS:
            kn = torch.tensor([[k/16.0]])
            for a in [0, 1]:
                a_oh.zero_()
                a_oh[0, a] = 1.0
                x = torch.cat([sn, a_oh, kn], dim=-1)
                with torch.no_grad():
                    pred = wm(x)
                s = score_state(pred).item()
                if s < best_score:
                    best_score, best_a, best_k = s, a, k

        samples.append({
            'state': sn.squeeze(0).numpy().copy(),
            'action': float(best_a),
            'k_cont': best_k / 16.0,
        })

    out = {
        'state': torch.tensor(np.array([s['state'] for s in samples])),
        'action': torch.tensor([[s['action']] for s in samples]),
        'k_cont': torch.tensor([[s['k_cont']] for s in samples]),
    }
    torch.save(out, 'cartpole_dn_data.pt')
    print(f'Generated {len(samples)} labels')

    # Distribution
    for kv in K_VALS:
        c = sum(1 for s in samples if round(s['k_cont']*16) == kv)
        print(f'  k={kv:2d}: {c}/{n_samples} ({c/n_samples*100:.0f}%)')
    return out


def train(data_path='cartpole_dn_data.pt', n_epochs=1500, lr=1e-2):
    data = torch.load(data_path, weights_only=True)
    s, a_lbl, k_lbl = data['state'], data['action'], data['k_cont']

    n_train = int(len(s) * 0.85)
    idx = torch.randperm(len(s))
    s_tr, a_tr, k_tr = s[idx[:n_train]], a_lbl[idx[:n_train]], k_lbl[idx[:n_train]]
    s_va, a_va, k_va = s[idx[n_train:]], a_lbl[idx[n_train:]], k_lbl[idx[n_train:]]
    print(f'Train: {n_train}, Val: {len(s)-n_train}')

    model = KAN([4, 12, 2], grid_size=5, spline_order=3)
    print(f'CartPole Decision Net [4,12,2]: {sum(p.numel() for p in model.parameters())} params')

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=500, gamma=0.5)
    bce = torch.nn.BCEWithLogitsLoss()
    mse = torch.nn.MSELoss()

    for epoch in range(1, n_epochs + 1):
        model.train()
        opt.zero_grad()
        out = model(s_tr)
        loss = bce(out[:, 0:1], a_tr) + 0.5 * mse(out[:, 1:2], k_tr)
        loss.backward()
        opt.step()
        scheduler.step()

        if epoch % 300 == 0:
            model.eval()
            with torch.no_grad():
                out_v = model(s_va)
                lv = bce(out_v[:, 0:1], a_va) + 0.5 * mse(out_v[:, 1:2], k_va)
                a_acc = ((torch.sigmoid(out_v[:, 0:1]) > 0.5).float() == a_va).float().mean().item()
                k_rmse = mse(out_v[:, 1:2], k_va).sqrt().item()
            print(f'Epoch {epoch:4d}  lr={scheduler.get_last_lr()[0]:.4f}  '
                  f'train={loss.item():.4f}  val={lv.item():.4f}  '
                  f'a_acc={a_acc:.3f}  k_rmse={k_rmse:.3f}')

    torch.save(model.state_dict(), 'kan_cartpole_dn_v2.pt')
    print('Saved: kan_cartpole_dn_v2.pt')
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gen-labels', action='store_true')
    parser.add_argument('--n-samples', type=int, default=2000)
    args = parser.parse_args()

    if args.gen_labels:
        wm = load_wm()
        generate_labels(wm, n_samples=args.n_samples)
    else:
        train()
