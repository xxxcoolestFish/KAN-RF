"""Priority 1: Stable baseline reproduction across multiple seeds.

1. ProtoKAN WM + Shooting MPC (H=8, N=500): 10 seeds
2. ProtoKAN WM + KAN Policy (via KANPolicyTrainer): 10 seeds
"""
import torch, torch.nn as nn, numpy as np, time, sys, os, gymnasium as gym

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kanrf import ProtoKAN
from control.kan_policy_net import KANPolicy, KANPolicyTrainer

PI_2 = np.pi / 2
S_TARGET = torch.tensor([[0.0, 1.0, 0.0]])


# ═══════════════════════════════════════
# WM Training (shared across seeds)
# ═══════════════════════════════════════

def generate_pendulum_data(n=5000, seed=42):
    """Generate (s,a,s') triplets from Pendulum simulator."""
    np.random.seed(seed)
    env = gym.make('Pendulum-v1')
    env.reset()
    xs, ys = [], []
    for _ in range(n):
        theta = np.random.uniform(-np.pi, np.pi)
        thd = np.random.uniform(-8.0, 8.0)
        action = np.random.uniform(-2.0, 2.0)
        s_before = np.array([np.cos(theta), np.sin(theta), thd / 8.0], dtype=np.float32)
        env.unwrapped.state = (theta, thd)
        obs, _, _, _, _ = env.step([action])
        s_after = np.array([obs[0], obs[1], obs[2] / 8.0], dtype=np.float32)
        a_norm = action / 2.0
        xs.append(np.concatenate([s_before, [a_norm]]))
        ys.append(s_after)
    env.close()
    return (torch.tensor(np.array(xs), dtype=torch.float32),
            torch.tensor(np.array(ys), dtype=torch.float32))


def train_wm(X, Y, n_proto=16, n_lbfgs=100):
    """Train ProtoKAN WM [4,12,3] with L-BFGS."""
    n_tr = int(len(X) * 0.85)
    X_tr, Y_tr = X[:n_tr], Y[:n_tr]
    X_val, Y_val = X[n_tr:], Y[n_tr:]
    wm = ProtoKAN([4, 12, 3], n_prototypes=n_proto)
    for layer in wm.layers:
        layer.log_sigma.data.fill_(-1.5)  # sigma ≈ 0.22 for locality

    mse_fn = nn.MSELoss()
    best_val = float('inf'); best_state = None

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

    wm.load_state_dict(best_state); wm.eval()
    return wm, best_val


# ═══════════════════════════════════════
# Policy Training
# ═══════════════════════════════════════

def generate_policy_states(n=10000, seed=42):
    """Generate states for policy training."""
    np.random.seed(seed)
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
    s_raw = np.vstack([s1, s2]); np.random.shuffle(s_raw)
    s_norm = s_raw.copy(); s_norm[:, 2] /= 8.0
    return torch.tensor(s_norm, dtype=torch.float32)


def train_policy(wm, s_dataset, seed=42, epochs=200, lr=1e-3):
    """Train KAN Policy via frozen ProtoKAN WM."""
    torch.manual_seed(seed); np.random.seed(seed)
    policy = KANPolicy(state_dim=3, action_dim=1, hidden_dim=12, n_layers=2)
    trainer = KANPolicyTrainer(wm, policy, S_TARGET, lr=lr)
    for ep in range(1, epochs + 1):
        trainer.train_epoch(s_dataset)
    return policy, trainer


# ═══════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════

def evaluate_mpc(wm, n_trials=10, seed=42):
    """Shooting MPC: H=8, N=500 samples per step."""
    np.random.seed(seed)
    env = gym.make('Pendulum-v1')
    successes = 0; all_steps = []; all_errs = []

    for trial in range(n_trials):
        trial_seed = seed + trial * 100
        obs, _ = env.reset(seed=trial_seed)
        ok = False
        for step in range(300):
            s_n = torch.tensor([obs[0], obs[1], obs[2] / 8.0], dtype=torch.float32)

            # Random shooting H=8
            best_cost = float('inf'); best_a = 0.0
            for _ in range(500):
                seq = np.random.uniform(-1, 1, 8).astype(np.float32)
                s_cur = s_n.clone()
                for a in seq:
                    a_t = torch.tensor([[a]], dtype=torch.float32)
                    with torch.no_grad():
                        s_cur = wm(torch.cat([s_cur.unsqueeze(0), a_t], dim=-1)).squeeze(0)
                cost = -s_cur[1].item()  # maximize sin(theta)
                if cost < best_cost:
                    best_cost = cost; best_a = seq[0]

            obs, _, _, _, _ = env.step([best_a * 2.0])
            err = min(abs(np.arctan2(obs[1], obs[0]) - PI_2),
                     2 * np.pi - abs(np.arctan2(obs[1], obs[0]) - PI_2))
            if err < 0.2:
                successes += 1; all_steps.append(step + 1); all_errs.append(err); ok = True; break
        if not ok:
            all_steps.append(300)
            all_errs.append(min(abs(np.arctan2(obs[1], obs[0]) - PI_2),
                               2 * np.pi - abs(np.arctan2(obs[1], obs[0]) - PI_2)))
    env.close()
    return successes, all_steps, all_errs


def evaluate_policy(trainer, n_trials=10, seed=42):
    """Evaluate KAN Policy on Pendulum-v1."""
    env = gym.make('Pendulum-v1')
    successes = 0; all_steps = []; all_errs = []

    for trial in range(n_trials):
        trial_seed = seed + trial * 100
        obs, _ = env.reset(seed=trial_seed)
        ok = False
        for step in range(300):
            s_n = np.array([obs[0], obs[1], obs[2] / 8.0], dtype=np.float32)
            a_norm = trainer.get_action(s_n)
            obs, _, _, _, _ = env.step([a_norm * 2.0])
            err = min(abs(np.arctan2(obs[1], obs[0]) - PI_2),
                     2 * np.pi - abs(np.arctan2(obs[1], obs[0]) - PI_2))
            if err < 0.2:
                successes += 1; all_steps.append(step + 1); all_errs.append(err); ok = True; break
        if not ok:
            all_steps.append(300)
            all_errs.append(min(abs(np.arctan2(obs[1], obs[0]) - PI_2),
                               2 * np.pi - abs(np.arctan2(obs[1], obs[0]) - PI_2)))
    env.close()
    return successes, all_steps, all_errs


# ═══════════════════════════════════════
# Main
# ═══════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mpc_seeds', type=int, default=10)
    parser.add_argument('--policy_seeds', type=int, default=10)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    device = torch.device(args.device)

    print("=" * 70)
    print("PRIORITY 1: Stable Baseline Reproduction")
    print("=" * 70)

    # ── Train WM (once, seed=42) ──
    print("\n[0] Training ProtoKAN WM [4,12,3]...")
    torch.manual_seed(42)
    X, Y = generate_pendulum_data(5000, seed=42)
    X, Y = X.to(device), Y.to(device)
    t0 = time.time()
    wm, wm_val = train_wm(X, Y)
    print(f"  val_mse={wm_val:.6f}  time={time.time()-t0:.0f}s")

    # ── Experiment 1: Shooting MPC ──
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: ProtoKAN WM + Shooting MPC (H=8, N=500)")
    print("=" * 70)
    mpc_results = []
    for si in range(args.mpc_seeds):
        mpc_seed = 100 + si
        s, st, er = evaluate_mpc(wm, n_trials=10, seed=mpc_seed)
        mpc_results.append((s, st, er))
        print(f"  Seed {mpc_seed}: {s}/10  mean_steps={np.mean(st):.0f}  mean_err={np.mean(er):.3f}")

    mpc_sr = [r[0]/10 for r in mpc_results]
    print(f"\n  MPC Summary: {np.mean(mpc_sr)*100:.0f}% ± {np.std(mpc_sr)*100:.0f}%  "
          f"range=[{min(mpc_sr)*100:.0f}%, {max(mpc_sr)*100:.0f}%]")

    # ── Experiment 2: KAN Policy ──
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: ProtoKAN WM + KAN Policy (retrain per seed)")
    print("=" * 70)
    pol_results = []
    for si in range(args.policy_seeds):
        pol_seed = 200 + si
        s_dataset = generate_policy_states(10000, seed=pol_seed).to(device)
        policy, trainer = train_policy(wm, s_dataset, seed=pol_seed, epochs=200, lr=1e-3)
        s, st, er = evaluate_policy(trainer, n_trials=10, seed=pol_seed)
        pol_results.append((s, st, er))
        print(f"  Seed {pol_seed}: {s}/10  mean_steps={np.mean(st):.0f}  mean_err={np.mean(er):.3f}")

    pol_sr = [r[0]/10 for r in pol_results]
    print(f"\n  Policy Summary: {np.mean(pol_sr)*100:.0f}% ± {np.std(pol_sr)*100:.0f}%  "
          f"range=[{min(pol_sr)*100:.0f}%, {max(pol_sr)*100:.0f}%]")


if __name__ == '__main__':
    main()
