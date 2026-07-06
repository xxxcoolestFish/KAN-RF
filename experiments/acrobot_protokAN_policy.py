"""Acrobot: ProtoKAN WM → KAN Policy via frozen WM gradient.

Critical test: Can ProtoKAN WM provide accurate enough gradients
to train a KAN policy for Acrobot? (KAN WM: 0/10)
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, time, sys, os, gymnasium as gym

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kanrf import ProtoKAN, KAN

MAX_V1 = 6.0; MAX_V2 = 8.0
S_TARGET = torch.tensor([[1.0, 0.0, 1.0, 0.0, 0.0, 0.0]])  # both upright


class AcrobotPolicy(nn.Module):
    """KAN Policy for discrete 3-action output."""
    def __init__(self, state_dim=6, hidden=12):
        super().__init__()
        from kanrf import KANLayer
        self.layer1 = KANLayer(state_dim, hidden, grid_size=5, spline_order=3)
        self.layer2 = KANLayer(hidden, hidden, grid_size=5, spline_order=3)
        self.out = nn.Linear(hidden, 3)

    def forward(self, s, return_activations=False):
        x = s
        Bs, Es = [], []
        for layer in [self.layer1, self.layer2]:
            if return_activations:
                x, B, E = layer(x, return_activations=True)
                Bs.append(B); Es.append(E)
            else:
                x = layer(x)
        logits = self.out(x)
        probs = F.softmax(logits, dim=-1)
        if return_activations:
            return probs, Bs, Es
        return probs


def generate_data(n_states=5000, device='cpu'):
    """Generate single-scale Acrobot training data."""
    env = gym.make('Acrobot-v1')
    env.reset()
    max_v1, max_v2 = 6.0, 8.0
    xs, ys = [], []

    for _ in range(n_states):
        theta1 = np.random.uniform(-np.pi, np.pi)
        theta2 = np.random.uniform(-np.pi, np.pi)
        dtheta1 = np.random.uniform(-max_v1, max_v1)
        dtheta2 = np.random.uniform(-max_v2, max_v2)
        env.unwrapped.state = (theta1, theta2, dtheta1, dtheta2)
        s0 = env.unwrapped._get_ob()

        for a in range(3):
            env.unwrapped.state = (theta1, theta2, dtheta1, dtheta2)
            obs, _, term, trunc, _ = env.step(a)
            s_norm = np.array([s0[0], s0[1], s0[2], s0[3],
                              s0[4]/max_v1, s0[5]/max_v2], dtype=np.float32)
            sn = np.array([obs[0], obs[1], obs[2], obs[3],
                          obs[4]/max_v1, obs[5]/max_v2], dtype=np.float32)
            a_oh = np.zeros(3, dtype=np.float32); a_oh[a] = 1.0
            xs.append(np.concatenate([s_norm, a_oh]))
            ys.append(sn)

    env.close()
    return (torch.tensor(np.array(xs), dtype=torch.float32).to(device),
            torch.tensor(np.array(ys), dtype=torch.float32).to(device))


def train_wm(X, Y, n_proto=16, n_lbfgs=150, device='cpu'):
    """Train ProtoKAN WM with L-BFGS."""
    n_train = int(len(X) * 0.85)
    X_tr, Y_tr = X[:n_train], Y[:n_train]
    X_val, Y_val = X[n_train:], Y[n_train:]

    wm = ProtoKAN([9, 32, 6], n_prototypes=n_proto).to(device)
    n_params = sum(p.numel() for p in wm.parameters())
    print(f"ProtoKAN WM [9,32,6] N={n_proto}: {n_params} params")

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

    for step in range(1, n_lbfgs + 1):
        opt.step(closure)
        with torch.no_grad():
            val = mse_fn(wm(X_val), Y_val).item()
        if val < best_val:
            best_val = val
            best_state = {k: v.clone() for k, v in wm.state_dict().items()}
        if step % 30 == 0 or step == 1:
            print(f"  WM L-BFGS {step:3d}/{n_lbfgs}  val_mse={val:.6f}  best={best_val:.6f}")

    wm.load_state_dict(best_state)
    wm.eval()
    print(f"  WM trained: val_mse={best_val:.6f}")
    return wm


def train_policy(wm, s_states, epochs=300, batch_size=256, device='cpu'):
    """Train KAN Policy using frozen ProtoKAN WM as gradient teacher."""
    policy = AcrobotPolicy(state_dim=6, hidden=12).to(device)
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"KAN Policy [6,12,12,3]: {n_params} params")

    opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
    N = len(s_states)
    n_batches = 60
    wm.eval()

    for ep in range(1, epochs + 1):
        total_loss = 0
        for _ in range(n_batches):
            idx = torch.randint(0, N, (batch_size,), device=device)
            s_b = s_states[idx]

            policy.train(); opt.zero_grad()
            ap = policy(s_b)
            wm_in = torch.cat([s_b, ap], dim=-1)
            s_pred = wm(wm_in)

            w = torch.tensor([5., 5., 5., 5., 1., 1.], device=device)
            loss = ((s_pred - S_TARGET.to(device)).pow(2) * w).mean()
            loss = loss - 0.01 * (ap * (ap + 1e-8).log()).sum(-1).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
            opt.step()
            policy.eval()
            total_loss += loss.item()

        if ep % 60 == 0:
            print(f"  Policy epoch {ep:3d}  loss={total_loss/n_batches:.4f}")

    return policy


def evaluate(policy, n_trials=10, device='cpu'):
    """Test policy on Acrobot-v1."""
    env = gym.make('Acrobot-v1')
    successes = 0; all_steps = []

    for trial in range(n_trials):
        seed = 42 + trial * 100
        obs, _ = env.reset(seed=seed)
        ok = False
        for step in range(500):
            s_n = torch.tensor([[obs[0], obs[1], obs[2], obs[3],
                                obs[4]/MAX_V1, obs[5]/MAX_V2]],
                               dtype=torch.float32, device=device)
            with torch.no_grad():
                ap = policy(s_n).squeeze().cpu().numpy()
            action = int(np.argmax(ap))
            obs, _, term, trunc, _ = env.step(action)
            if term:
                successes += 1; all_steps.append(step + 1); ok = True; break
        if not ok:
            all_steps.append(500)
        print(f"  [{trial+1:2d}] {'✓' if ok else '✗'} steps={all_steps[-1]}")

    env.close()
    return successes, all_steps


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_states', type=int, default=5000)
    parser.add_argument('--n_prototypes', type=int, default=16)
    parser.add_argument('--lbfgs_iters', type=int, default=150)
    parser.add_argument('--policy_epochs', type=int, default=300)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--skip_wm', action='store_true', default=False)
    parser.add_argument('--wm_path', type=str, default='/tmp/acrobot_protokAN_wm.pt')
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(42); np.random.seed(42)

    print("=" * 70)
    print("ProtoKAN WM → KAN Policy — Acrobot Decision")
    print("=" * 70)

    # ── 1. Generate data ──
    print(f"\n[1/4] Generating {args.n_states} states ({args.n_states*3} data points)...")
    data_file = f'/tmp/acrobot_ss_{args.n_states}.pt'
    if os.path.exists(data_file):
        X, Y = torch.load(data_file, weights_only=True, map_location=device)
        print(f"  Loaded from cache: {X.shape[0]} samples")
    else:
        X, Y = generate_data(args.n_states, device)
        torch.save((X, Y), data_file)
        print(f"  Generated: {X.shape[0]} samples")

    s_states_all = X[:, :6]
    # Deduplicate states (each state appears 3x, once per action)
    # Use unique states for policy training (don't bias toward any action)
    s_unique = torch.unique(s_states_all, dim=0)
    print(f"  Unique states for policy: {s_unique.shape[0]}")

    # ── 2. Train ProtoKAN WM ──
    print(f"\n[2/4] Training ProtoKAN WM...")
    if args.skip_wm and os.path.exists(args.wm_path):
        wm = ProtoKAN([9, 32, 6], n_prototypes=args.n_prototypes).to(device)
        wm.load_state_dict(torch.load(args.wm_path, weights_only=True, map_location=device))
        wm.eval()
        print(f"  Loaded from {args.wm_path}")
    else:
        wm = train_wm(X, Y, n_proto=args.n_prototypes, n_lbfgs=args.lbfgs_iters, device=device)
        torch.save(wm.state_dict(), args.wm_path)
        print(f"  Saved to {args.wm_path}")

    # ── 3. Train Policy ──
    print(f"\n[3/4] Training KAN Policy via frozen ProtoKAN WM...")
    policy = train_policy(wm, s_unique, epochs=args.policy_epochs, device=device)

    # ── 4. Evaluate ──
    print(f"\n[4/4] Evaluating on Acrobot-v1 (10 trials)...")
    successes, all_steps = evaluate(policy, n_trials=10, device=device)

    print(f"\n{'='*70}")
    print(f"RESULT: {successes}/10 ({successes*10}%)  mean_steps={np.mean(all_steps):.0f}")
    # Compare to previous result
    print(f"Previous (KAN WM): 0/10")
    print(f"Improvement: {'✓' if successes > 0 else '✗'}")
    print("=" * 70)


if __name__ == '__main__':
    main()
