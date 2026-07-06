"""Pendulum: Full ProtoKAN Stack — ProtoKAN WM + ProtoKAN Policy.

Train ProtoKAN World Model (L-BFGS) → Train ProtoKAN Policy via frozen WM gradient.
Compare against KAN stack baseline.
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, time, sys, os, gymnasium as gym

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kanrf import ProtoKAN, ProtoKANLayer, KAN, KANLayer

S_TARGET = torch.tensor([[0.0, 1.0, 0.0]])  # Pendulum upright


# ═══════════════════════════════════════════════════════════════
# ProtoKAN Policy Network
# ═══════════════════════════════════════════════════════════════

class ProtoKANPolicy(nn.Module):
    """ProtoKAN-based policy: learned prototype edges → tanh output."""
    def __init__(self, state_dim=3, action_dim=1, hidden_dim=12,
                 n_prototypes=16):
        super().__init__()
        self.layer1 = ProtoKANLayer(state_dim, hidden_dim, n_prototypes=n_prototypes)
        self.layer2 = ProtoKANLayer(hidden_dim, hidden_dim, n_prototypes=n_prototypes)
        self.out = nn.Linear(hidden_dim, action_dim)

    def forward(self, s, return_activations=False):
        x = s
        Bs, Es = [], []
        for layer in [self.layer1, self.layer2]:
            if return_activations:
                x, B, E = layer(x, return_activations=True)
                Bs.append(B); Es.append(E)
            else:
                x = layer(x)
        a = torch.tanh(self.out(x))
        if return_activations:
            return a, Bs, Es
        return a


# ═══════════════════════════════════════════════════════════════
# Data Generation
# ═══════════════════════════════════════════════════════════════

def generate_pendulum_data(n_states=5000, device='cpu'):
    """Generate (s, a, s') triplets from Pendulum simulator."""
    env = gym.make('Pendulum-v1')
    env.reset()
    xs, ys = [], []

    for _ in range(n_states):
        theta = np.random.uniform(-np.pi, np.pi)
        thd = np.random.uniform(-8.0, 8.0)
        action = np.random.uniform(-2.0, 2.0)
        # Record state BEFORE action
        s_before = np.array([np.cos(theta), np.sin(theta), thd/8.0],
                            dtype=np.float32)
        env.unwrapped.state = (theta, thd)
        obs, _, _, _, _ = env.step([action])
        s_after = np.array([obs[0], obs[1], obs[2]/8.0], dtype=np.float32)
        a_norm = action / 2.0
        xs.append(np.concatenate([s_before, [a_norm]]))
        ys.append(s_after)

    env.close()
    return (torch.tensor(np.array(xs), dtype=torch.float32).to(device),
            torch.tensor(np.array(ys), dtype=torch.float32).to(device))


def generate_policy_states(n=10000, device='cpu'):
    """Generate states for policy training (mix of uniform + rollout)."""
    n_half = n // 2
    angles = np.random.uniform(-np.pi, np.pi, n_half)
    s1 = np.stack([np.cos(angles), np.sin(angles),
                   np.random.uniform(-8.0, 8.0, n_half)], axis=1)
    env = gym.make('Pendulum-v1')
    env.reset()
    states = []; obs, _ = env.reset()
    while len(states) < n_half:
        obs, _, term, trunc, _ = env.step(env.action_space.sample())
        states.append(obs.copy())
        if term or trunc: obs, _ = env.reset()
    env.close()
    s2 = np.array(states[:n_half])
    s_raw = np.vstack([s1, s2])
    np.random.shuffle(s_raw)
    s_norm = s_raw.copy(); s_norm[:, 2] /= 8.0
    return torch.tensor(s_norm, dtype=torch.float32).to(device)


# ═══════════════════════════════════════════════════════════════
# WM Training
# ═══════════════════════════════════════════════════════════════

def train_wm(X, Y, n_proto=16, n_lbfgs=120, device='cpu'):
    """Train ProtoKAN WM [4,12,3] with L-BFGS."""
    n_tr = int(len(X) * 0.85)
    X_tr, Y_tr = X[:n_tr], Y[:n_tr]
    X_val, Y_val = X[n_tr:], Y[n_tr:]

    wm = ProtoKAN([4, 12, 3], n_prototypes=n_proto).to(device)
    print(f"ProtoKAN WM [4,12,3] N={n_proto}: {sum(p.numel() for p in wm.parameters())} params")

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
    print(f"  WM final val_mse={best_val:.6f}")
    return wm


# ═══════════════════════════════════════════════════════════════
# Policy Training
# ═══════════════════════════════════════════════════════════════

def train_protokAN_policy(wm, s_dataset, epochs=200, lr=1e-3,
                          batch_size=256, n_proto=16, device='cpu'):
    """Train ProtoKAN Policy via frozen ProtoKAN WM gradient."""
    policy = ProtoKANPolicy(state_dim=3, action_dim=1, hidden_dim=12,
                            n_prototypes=n_proto).to(device)
    print(f"ProtoKAN Policy [3,12,12,1] N={n_proto}: "
          f"{sum(p.numel() for p in policy.parameters())} params")

    wm.eval()
    for p in wm.parameters():
        p.requires_grad = False

    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    s_target = S_TARGET.to(device)
    N = len(s_dataset)

    for ep in range(1, epochs + 1):
        total_loss = 0.0
        n_batches = max(1, N // batch_size)
        for _ in range(n_batches):
            idx = torch.randint(0, N, (batch_size,), device=device)
            s_b = s_dataset[idx]

            policy.train(); opt.zero_grad()
            a = policy(s_b)
            wm_in = torch.cat([s_b, a], dim=-1)
            s_pred = wm(wm_in)

            # Energy-guided loss (same as KANPolicyTrainer)
            thd = s_b[:, 2] * 8.0
            E_current = 0.5 * thd.pow(2) + 10.0 * s_b[:, 1]
            thd_pred = s_pred[:, 2] * 8.0
            E_pred = 0.5 * thd_pred.pow(2) + 10.0 * s_pred[:, 1]
            E_des = 10.0
            energy_deficit = (E_des - E_current).detach()
            energy_gain = (E_pred - E_current) * torch.sign(energy_deficit)

            sin = s_b[:, 1]
            w_stable = ((1.0 + sin) / 2.0).clamp(0.0, 1.0)

            energy_loss = -energy_gain.mean()
            dist_loss = (w_stable * (s_pred - s_target.expand(batch_size, -1))
                         .pow(2).sum(dim=-1)).mean()
            ctrl_loss = a.pow(2).mean()
            loss = energy_loss + dist_loss + 0.01 * ctrl_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
            opt.step()
            total_loss += loss.item()

        if ep % 40 == 0 or ep == 1:
            print(f"  Policy epoch {ep:3d}  loss={total_loss/n_batches:.4f}")

    return policy


# ═══════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════

def evaluate(policy, n_trials=10, max_steps=300, device='cpu'):
    """Evaluate policy on Pendulum-v1. Success = |θ_error| < 0.2."""
    env = gym.make('Pendulum-v1')
    PI_2 = np.pi / 2
    successes = 0; all_steps = []; all_errors = []

    for trial in range(n_trials):
        seed = 42 + trial * 100
        obs, _ = env.reset(seed=seed)
        ok = False
        for step in range(max_steps):
            s_n = torch.tensor([[obs[0], obs[1], obs[2]/8.0]],
                               dtype=torch.float32, device=device)
            with torch.no_grad():
                a_norm = policy(s_n).item()
            a_raw = a_norm * 2.0
            obs, _, term, trunc, _ = env.step([a_raw])
            err = min(abs(np.arctan2(obs[1], obs[0]) - PI_2),
                     2*np.pi - abs(np.arctan2(obs[1], obs[0]) - PI_2))
            if err < 0.2:
                successes += 1; all_steps.append(step+1); all_errors.append(err)
                ok = True; break
        if not ok:
            all_steps.append(max_steps)
            err = min(abs(np.arctan2(obs[1], obs[0]) - PI_2),
                     2*np.pi - abs(np.arctan2(obs[1], obs[0]) - PI_2))
            all_errors.append(err)
        print(f"  [Trial {trial+1:2d}] {'✓' if ok else '✗'} "
              f"steps={all_steps[-1]:3d}  |Δθ|={all_errors[-1]:.4f}")

    env.close()
    return successes, all_steps, all_errors


# ═══════════════════════════════════════════════════════════════
# KAN Stack Baseline (for comparison)
# ═══════════════════════════════════════════════════════════════

def train_kan_baseline(X, Y, s_policy, epochs=200, device='cpu'):
    """Train KAN WM + KAN Policy as baseline."""
    n_tr = int(len(X) * 0.85)
    X_tr, Y_tr = X[:n_tr], Y[:n_tr]
    X_val, Y_val = X[n_tr:], Y[n_tr:]

    # KAN WM with Adam (L-BFGS too slow for fair comparison)
    kan_wm = KAN([4, 12, 3], grid_size=5, spline_order=3).to(device)
    print(f"KAN WM [4,12,3]: {sum(p.numel() for p in kan_wm.parameters())} params")

    mse_fn = nn.MSELoss()
    opt = torch.optim.Adam(kan_wm.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=200, gamma=0.5)

    for ep in range(1, 401):
        idx = torch.randperm(n_tr)[:1024]
        opt.zero_grad()
        loss = mse_fn(kan_wm(X_tr[idx]), Y_tr[idx])
        loss.backward(); opt.step(); scheduler.step()
        if ep % 100 == 0:
            with torch.no_grad():
                val = mse_fn(kan_wm(X_val), Y_val).item()
            print(f"  KAN WM epoch {ep:3d}  val_mse={val:.6f}")

    kan_wm.eval()
    with torch.no_grad():
        val = mse_fn(kan_wm(X_val), Y_val).item()
    print(f"  KAN WM final val_mse={val:.6f}")

    # KAN Policy
    from control.kan_policy_net import KANPolicy
    kan_pol = KANPolicy(state_dim=3, action_dim=1, hidden_dim=8).to(device)
    print(f"KAN Policy [3,8,1]: {sum(p.numel() for p in kan_pol.parameters())} params")

    kan_wm.eval()
    for p in kan_wm.parameters():
        p.requires_grad = False

    pol_opt = torch.optim.Adam(kan_pol.parameters(), lr=1e-3)
    s_target = S_TARGET.to(device)
    N = len(s_policy)

    for ep in range(1, epochs + 1):
        total_loss = 0
        n_batches = max(1, N // 256)
        for _ in range(n_batches):
            idx = torch.randint(0, N, (256,), device=device)
            s_b = s_policy[idx]
            pol_opt.zero_grad()
            a = kan_pol(s_b)
            s_pred = kan_wm(torch.cat([s_b, a], dim=-1))

            thd = s_b[:, 2] * 8.0; Ec = 0.5*thd.pow(2) + 10.0*s_b[:, 1]
            thdp = s_pred[:, 2]*8.0; Ep = 0.5*thdp.pow(2) + 10.0*s_pred[:, 1]
            deficit = (10.0 - Ec).detach()
            egain = (Ep - Ec) * torch.sign(deficit)
            sin = s_b[:, 1]; ws = ((1.0+sin)/2.0).clamp(0,1)
            loss = -egain.mean() + (ws*(s_pred-s_target.expand(256,-1)).pow(2).sum(-1)).mean() + 0.01*a.pow(2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(kan_pol.parameters(), 10.0)
            pol_opt.step()
            total_loss += loss.item()

        if ep % 40 == 0:
            print(f"  KAN Policy epoch {ep:3d}  loss={total_loss/n_batches:.4f}")

    return kan_pol


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_states', type=int, default=5000,
                        help='States for WM training (x1 per state)')
    parser.add_argument('--n_policy', type=int, default=10000,
                        help='States for policy training')
    parser.add_argument('--n_proto', type=int, default=16)
    parser.add_argument('--lbfgs_iters', type=int, default=120)
    parser.add_argument('--policy_epochs', type=int, default=200)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--baseline', action='store_true', default=False,
                        help='Also run KAN baseline for comparison')
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(42); np.random.seed(42)

    print("=" * 70)
    print("Pendulum: Full ProtoKAN Stack (WM + Policy)")
    print("=" * 70)

    # ── 1. Generate data ──
    print(f"\n[1/5] Generating data...")
    t0 = time.time()
    X, Y = generate_pendulum_data(args.n_states, device)
    s_policy = generate_policy_states(args.n_policy, device)
    print(f"  WM data: {X.shape[0]} (s,a,s') triplets")
    print(f"  Policy states: {s_policy.shape[0]}")
    print(f"  Time: {time.time()-t0:.0f}s")

    # ── 2. Train ProtoKAN WM ──
    print(f"\n[2/5] Training ProtoKAN World Model...")
    t0 = time.time()
    wm = train_wm(X, Y, n_proto=args.n_proto, n_lbfgs=args.lbfgs_iters, device=device)
    print(f"  Time: {time.time()-t0:.0f}s")

    # ── 3. Train ProtoKAN Policy ──
    print(f"\n[3/5] Training ProtoKAN Policy via frozen WM...")
    t0 = time.time()
    policy = train_protokAN_policy(wm, s_policy, epochs=args.policy_epochs,
                                   n_proto=args.n_proto, device=device)
    print(f"  Time: {time.time()-t0:.0f}s")

    # ── 4. Evaluate ──
    print(f"\n[4/5] Evaluating ProtoKAN Stack on Pendulum-v1...")
    successes, all_steps, all_errors = evaluate(policy, n_trials=10, device=device)
    sr = successes / 10
    print(f"\n  ProtoKAN Stack: {successes}/10 ({sr*100:.0f}%)  "
          f"mean_steps={np.mean(all_steps):.0f}  "
          f"mean|Δθ|={np.mean(all_errors):.4f}+/-{np.std(all_errors):.4f}")

    # ── 5. KAN Baseline (optional) ──
    if args.baseline:
        print(f"\n[5/5] Training KAN Stack baseline for comparison...")
        t0 = time.time()
        kan_pol = train_kan_baseline(X, Y, s_policy, epochs=args.policy_epochs,
                                     device=device)
        print(f"  Time: {time.time()-t0:.0f}s")

        print(f"\n  Evaluating KAN Stack...")
        ks, ka, ke = 0, [], []
        env = gym.make('Pendulum-v1')
        PI_2 = np.pi/2
        for trial in range(10):
            seed = 42 + trial*100; obs, _ = env.reset(seed=seed)
            ok = False
            for step in range(300):
                s_n = torch.tensor([[obs[0], obs[1], obs[2]/8.0]],
                                   dtype=torch.float32, device=device)
                with torch.no_grad():
                    a_norm = kan_pol(s_n).item()
                obs, _, _, _, _ = env.step([a_norm*2.0])
                err = min(abs(np.arctan2(obs[1],obs[0])-PI_2),
                         2*np.pi-abs(np.arctan2(obs[1],obs[0])-PI_2))
                if err < 0.2:
                    ks += 1; ka.append(step+1); ke.append(err); ok=True; break
            if not ok:
                ka.append(300)
                ke.append(min(abs(np.arctan2(obs[1],obs[0])-PI_2),
                             2*np.pi-abs(np.arctan2(obs[1],obs[0])-PI_2)))
            print(f"  [KAN Trial {trial+1:2d}] {'✓' if ok else '✗'} steps={ka[-1]:3d}")
        env.close()
        print(f"\n  KAN Stack: {ks}/10 ({ks*10}%)  "
              f"mean_steps={np.mean(ka):.0f}  mean|Δθ|={np.mean(ke):.4f}")

    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"  ProtoKAN Stack: {successes}/10 ({sr*100:.0f}%)")
    if args.baseline:
        print(f"  KAN Stack:      {ks}/10 ({ks*10}%)")


if __name__ == '__main__':
    main()
