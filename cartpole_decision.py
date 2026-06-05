"""CartPole decision network + label generation.

Decision net: (state_4d) -> (action_logit, k_cont)
  - action_logit: BCEWithLogits with action label
  - k_cont: MSE with k/16 label

Labels generated via MPC through frozen world model:
  try all (action, k) pairs, pick the one that predicts the most balanced state.
"""
import torch, numpy as np, time, argparse
from kanrf import KAN

K_VALS = [1, 2, 4, 8]


def load_wm(path='kan_cartpole.pt'):
    model = KAN([7, 20, 4], grid_size=5, spline_order=3)
    model.load_state_dict(torch.load(path, weights_only=True))
    return model.eval()


def score_state(s_norm):
    """Lower score = more balanced (pole_angle=0, cart_pos=0, velocities=0)."""
    # Penalize pole angle (most important) and cart position
    return abs(s_norm[:, 2]) * 0.5 + abs(s_norm[:, 0]) * 0.2 \
        + abs(s_norm[:, 3]) * 0.2 + abs(s_norm[:, 1]) * 0.1


def generate_labels(wm, n_samples=1000):
    """Generate (state, best_action, best_k) labels via MPC through world model."""
    torch.manual_seed(42)
    states_norm = torch.rand(n_samples, 4) * 2 - 1
    samples = []
    a_onehot = torch.zeros(1, 2)

    for i in range(n_samples):
        sn = states_norm[i:i+1]
        best_a, best_k, best_score = None, None, float('inf')

        for a in [0, 1]:
            a_onehot.zero_()
            a_onehot[0, a] = 1.0
            for k in K_VALS:
                kn = torch.tensor([[k/16.0]])
                x = torch.cat([sn, a_onehot, kn], dim=-1)
                with torch.no_grad():
                    pred = wm(x)
                score = score_state(pred).item()
                if score < best_score:
                    best_score, best_a, best_k = score, a, k

        samples.append({
            'state': sn.squeeze(0).numpy().copy(),
            'action': float(best_a),
            'k_cont': best_k / 16.0,
        })

    out = {
        'state': torch.tensor([s['state'] for s in samples]),
        'action': torch.tensor([[s['action']] for s in samples]),
        'k_cont': torch.tensor([[s['k_cont']] for s in samples]),
    }
    torch.save(out, 'cartpole_decision_data.pt')
    print(f'Generated {len(samples)} labels, saved to cartpole_decision_data.pt')

    # Distribution check
    actions = [s['action'] for s in samples]
    print(f'Action distribution: left={actions.count(0)} right={actions.count(1)}')
    ks = [s['k_cont']*16 for s in samples]
    for kv in K_VALS:
        count = sum(1 for k in ks if round(k) == kv)
        print(f'  k={kv}: {count}/{n_samples} ({count/n_samples*100:.0f}%)')
    return out


class CartPoleDecisionNet(torch.nn.Module):
    """KAN decision network for CartPole."""
    def __init__(self):
        super().__init__()
        self.kan = KAN([4, 12, 2], grid_size=5, spline_order=3)

    def forward(self, s_norm):
        out = self.kan(s_norm)
        return out[:, 0:1], out[:, 1:2]  # (action_logit, k_cont)


def train(model, data_path='cartpole_decision_data.pt', n_epochs=1500, lr=1e-2):
    data = torch.load(data_path, weights_only=True)
    s = data['state']
    a_label = data['action']
    k_label = data['k_cont']

    n_train = int(len(s) * 0.8)
    idx = torch.randperm(len(s))
    s_tr, a_tr, k_tr = s[idx[:n_train]], a_label[idx[:n_train]], k_label[idx[:n_train]]
    s_va, a_va, k_va = s[idx[n_train:]], a_label[idx[n_train:]], k_label[idx[n_train:]]
    print(f'Train: {n_train}, Val: {len(s)-n_train}')

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=500, gamma=0.5)
    bce = torch.nn.BCEWithLogitsLoss()
    mse = torch.nn.MSELoss()

    for epoch in range(1, n_epochs + 1):
        model.train()
        opt.zero_grad()
        a_logit, k_pred = model(s_tr)
        loss = bce(a_logit, a_tr) + 0.5 * mse(k_pred, k_tr)
        loss.backward()
        opt.step()
        scheduler.step()

        if epoch % 300 == 0:
            model.eval()
            with torch.no_grad():
                a_logit_v, k_pred_v = model(s_va)
                loss_v = bce(a_logit_v, a_va) + 0.5 * mse(k_pred_v, k_va)
                a_acc = ((torch.sigmoid(a_logit_v) > 0.5).float() == a_va).float().mean().item()
            print(f'Epoch {epoch:4d}  lr={scheduler.get_last_lr()[0]:.4f}  '
                  f'train={loss.item():.4f}  val={loss_v.item():.4f}  a_acc={a_acc:.3f}')

    torch.save(model.state_dict(), 'kan_cartpole_dn.pt')
    print('Saved: kan_cartpole_dn.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gen-labels', action='store_true', help='Generate labels using world model')
    parser.add_argument('--n-samples', type=int, default=1000)
    args = parser.parse_args()

    if args.gen_labels:
        wm = load_wm()
        generate_labels(wm, n_samples=args.n_samples)
    else:
        model = CartPoleDecisionNet()
        print(f'CartPole Decision Net: [4,12,2]  params={sum(p.numel() for p in model.parameters())}')
        train(model)
