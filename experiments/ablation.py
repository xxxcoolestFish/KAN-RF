"""Ablation experiments: WM accuracy → Policy success.

1. ProtoKAN WM vs CWS-KAN WM → Policy training (10 seeds each)
2. WM accuracy sweep: how WM precision affects Policy success rate
3. Training curve comparison
"""
import torch, numpy as np, time, sys, os, gymnasium as gym

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kanrf import ProtoKAN, KAN
from control.kan_policy_net import KANPolicy, KANPolicyTrainer
from experiments.baseline_sweep import (
    generate_pendulum_data, train_wm, generate_policy_states, evaluate_policy
)

PI_2 = np.pi / 2
S_TARGET = torch.tensor([[0.0, 1.0, 0.0]])
device = 'cpu'


# ═══════════════════════════════════════
# Experiment 1: ProtoKAN vs CWS-KAN WM
# ═══════════════════════════════════════

def train_wm_cws(X, Y, n_lbfgs=100):
    """Train CWS-KAN WM with L-BFGS (same recipe as ProtoKAN)."""
    n_tr = int(len(X) * 0.85)
    X_tr, Y_tr = X[:n_tr], Y[:n_tr]
    X_val, Y_val = X[n_tr:], Y[n_tr:]
    wm = KAN([4, 12, 3], grid_size=5, spline_order=3)
    mse_fn = torch.nn.MSELoss()
    best_val = float('inf'); best_state = None

    def closure():
        opt.zero_grad()
        loss = mse_fn(wm(X_tr), Y_tr)
        loss.backward()
        return loss

    opt = torch.optim.LBFGS(wm.parameters(), lr=1.0, max_iter=20,
                            history_size=50, line_search_fn='strong_wolfe')
    for _ in range(1, n_lbfgs + 1):
        opt.step(closure)
        with torch.no_grad():
            val = mse_fn(wm(X_val), Y_val).item()
        if val < best_val:
            best_val = val
            best_state = {k: v.clone() for k, v in wm.state_dict().items()}
    wm.load_state_dict(best_state); wm.eval()
    return wm, best_val


def exp1_wm_comparison():
    """ProtoKAN vs CWS-KAN: train Policy with each WM, 10 seeds."""
    print("=" * 70)
    print("ABLATION 1: ProtoKAN WM vs CWS-KAN WM → Policy Success")
    print("=" * 70)

    torch.manual_seed(42); np.random.seed(42)
    X, Y = generate_pendulum_data(5000, seed=42)
    X, Y = X.to(device), Y.to(device)

    # Train both WMs
    print("\nTraining ProtoKAN WM...")
    proto_wm, proto_mse = train_wm(X, Y)
    print(f"  val_mse={proto_mse:.6f}")

    print("Training CWS-KAN WM...")
    cws_wm, cws_mse = train_wm_cws(X, Y)
    print(f"  val_mse={cws_mse:.6f}")

    # Train Policies (10 seeds each)
    results = {'ProtoKAN': [], 'CWS-KAN': []}
    for wm, wm_name in [(proto_wm, 'ProtoKAN'), (cws_wm, 'CWS-KAN')]:
        print(f"\n{wm_name} WM → Policy (10 seeds):")
        for si in range(10):
            pol_seed = 300 + si
            s_dataset = generate_policy_states(10000, seed=pol_seed).to(device)
            torch.manual_seed(pol_seed); np.random.seed(pol_seed)
            policy = KANPolicy(state_dim=3, action_dim=1, hidden_dim=12, n_layers=2)
            trainer = KANPolicyTrainer(wm, policy, S_TARGET, lr=1e-3)
            for ep in range(1, 201):
                trainer.train_epoch(s_dataset)
            s, st, er = evaluate_policy(trainer)
            results[wm_name].append((s, np.mean(st), np.mean(er)))
            print(f"  Seed {pol_seed}: {s}/10  steps={np.mean(st):.0f}  err={np.mean(er):.3f}")

    # Summary
    for wm_name in ['ProtoKAN', 'CWS-KAN']:
        sr = [r[0] / 10 for r in results[wm_name]]
        print(f"\n{wm_name}: {np.mean(sr)*100:.0f}% ± {np.std(sr)*100:.0f}%  "
              f"seeds_100%={sum(1 for r in results[wm_name] if r[0]==10)}/10")
    return results, proto_mse, cws_mse


# ═══════════════════════════════════════
# Experiment 2: WM accuracy sweep
# ═══════════════════════════════════════

def train_proto_wm_limited(X, Y, n_lbfgs):
    """Train ProtoKAN with limited L-BFGS iterations to simulate lower accuracy."""
    return train_wm(X, Y, n_lbfgs=n_lbfgs)


def exp2_accuracy_sweep():
    """Train ProtoKAN WMs at different accuracy levels, measure Policy success."""
    print("\n" + "=" * 70)
    print("ABLATION 2: WM Accuracy → Policy Success Rate")
    print("=" * 70)

    torch.manual_seed(42); np.random.seed(42)
    X, Y = generate_pendulum_data(5000, seed=42)
    X, Y = X.to(device), Y.to(device)

    accuracy_levels = []
    for n_iters in [3, 5, 10, 20, 40, 100]:
        torch.manual_seed(42); np.random.seed(42)
        wm, val_mse = train_proto_wm_limited(X, Y, n_iters)
        wm.eval()

        # Train 5 policies at this WM accuracy
        pol_successes = []
        for si in range(5):
            pol_seed = 500 + si
            s_dataset = generate_policy_states(10000, seed=pol_seed).to(device)
            torch.manual_seed(pol_seed); np.random.seed(pol_seed)
            policy = KANPolicy(state_dim=3, action_dim=1, hidden_dim=12, n_layers=2)
            trainer = KANPolicyTrainer(wm, policy, S_TARGET, lr=1e-3)
            for ep in range(1, 201):
                trainer.train_epoch(s_dataset)
            s, st, er = evaluate_policy(trainer)
            pol_successes.append(s)

        avg_sr = np.mean(pol_successes) / 10
        accuracy_levels.append((n_iters, val_mse, avg_sr, pol_successes))
        print(f"  L-BFGS {n_iters:3d}  WM_mse={val_mse:.6f}  "
              f"Policy SR={avg_sr*100:.0f}%  ({pol_successes})")

    print("\n  WM Accuracy → Policy Success:")
    print(f"  {'L-BFGS':>8s}  {'WM val_mse':>12s}  {'Policy SR':>12s}")
    for n_iters, val_mse, avg_sr, _ in accuracy_levels:
        print(f"  {n_iters:8d}  {val_mse:12.6f}  {avg_sr*100:11.0f}%")

    return accuracy_levels


# ═══════════════════════════════════════
# Experiment 3: CartPole failure analysis
# ═══════════════════════════════════════

def exp3_cartpole_analysis():
    """Analyze why CartPole Policy training fails."""
    print("\n" + "=" * 70)
    print("ABLATION 3: CartPole Training Failure Analysis")
    print("=" * 70)

    from experiments.cartpole_continual import (
        generate_wm_data, train_wm as cp_train_wm, generate_policy_states as cp_gen_states,
        X_S, XD_S, TH_S, THD_S, step_cartpole, CartPoleTrainer
    )

    torch.manual_seed(42); np.random.seed(42)
    X, Y = generate_wm_data(g=9.8, n=5000, device=device)
    wm, wm_val = cp_train_wm(X, Y, 'protokan', 100, device)
    print(f"  WM val_mse={wm_val:.6f}")

    # Train a policy and track detailed metrics
    s_pol = cp_gen_states(15000, device)
    policy = KANPolicy(state_dim=4, action_dim=1, hidden_dim=12, n_layers=2).to(device)
    trainer = CartPoleTrainer(wm, policy)

    # Track per-epoch metrics
    loss_history = []
    grad_norms = []
    action_mags = []

    for ep in range(1, 201):
        # Detailed single epoch tracking
        N = s_pol.shape[0]; batch_size = 256
        ep_loss = 0; ep_grad = 0; ep_act = 0; n_batches = 0
        for _ in range(max(1, N // batch_size)):
            idx = torch.randint(0, N, (batch_size,))
            s_b = s_pol[idx]
            policy.train(); trainer.opt.zero_grad()
            a = policy(s_b)
            s_pred = wm(torch.cat([s_b, a], dim=-1))
            loss = (s_pred[:, 2].pow(2).mean() + 0.1 * s_pred[:, 0].pow(2).mean() +
                    0.5 * s_pred[:, 3].pow(2).mean() + 0.01 * a.pow(2).mean())
            loss.backward()
            total_norm = 0
            for p in policy.parameters():
                if p.grad is not None:
                    total_norm += p.grad.norm().item() ** 2
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
            trainer.opt.step()
            ep_loss += loss.item()
            ep_grad += np.sqrt(total_norm)
            ep_act += a.abs().mean().item()
            n_batches += 1

        loss_history.append(ep_loss / n_batches)
        grad_norms.append(ep_grad / n_batches)
        action_mags.append(ep_act / n_batches)

        if ep % 40 == 0:
            print(f"  Epoch {ep:3d}  loss={loss_history[-1]:.4f}  "
                  f"grad={grad_norms[-1]:.4f}  |a|={action_mags[-1]:.4f}")

    # Analyze state distribution in training data
    s_np = s_pol.cpu().numpy()
    print(f"\n  Training state distribution (normalized):")
    for i, name in enumerate(['x', 'xd', 'th', 'thd']):
        print(f"    {name}: [{np.percentile(s_np[:, i], 5):.3f}, "
              f"{np.percentile(s_np[:, i], 95):.3f}]  "
              f"mean={np.mean(s_np[:, i]):.3f}  std={np.std(s_np[:, i]):.3f}")

    # Check WM Jacobian norm distribution on policy states
    wm.eval()
    jac_norms = []
    for i in range(100):
        s_test = s_pol[i:i+1]
        a_t = torch.zeros(1, 1, requires_grad=True)
        sp = wm(torch.cat([s_test, a_t], dim=-1))
        jac_sq = 0
        for dim in range(4):
            g = torch.autograd.grad(sp[0, dim], a_t, retain_graph=True)[0].item()
            jac_sq += g ** 2
        jac_norms.append(np.sqrt(jac_sq))
    print(f"\n  WM Jacobian ||ds/da|| on training states: "
          f"mean={np.mean(jac_norms):.4f}  std={np.std(jac_norms):.4f}  "
          f"min={np.min(jac_norms):.4f}")

    return loss_history, grad_norms, action_mags


# ═══════════════════════════════════════
# Experiment 4: 1-step ceiling data collection
# ═══════════════════════════════════════

def oracle_1step_pendulum(g=10.0, n_trials=100):
    """Oracle 1-step MPC on Pendulum: try 200 actions, pick best via true dynamics."""
    successes = 0
    for trial in range(n_trials):
        seed = 42 + trial * 100
        np.random.seed(seed)
        theta = np.random.uniform(-np.pi, np.pi)
        thd = np.random.uniform(-1.0, 1.0)
        for step in range(300):
            best_cost = float('inf'); best_u = 0.0
            for _ in range(200):
                u = np.random.uniform(-2, 2)
                thd_n = thd + (g * np.sin(theta) + u) * 0.05
                th_n = theta + thd_n * 0.05
                cost = -np.sin(th_n)
                if cost < best_cost:
                    best_cost = cost; best_u = u
            thd_n = thd + (g * np.sin(theta) + best_u) * 0.05
            th_n = theta + thd_n * 0.05
            theta, thd = th_n, thd_n
            err = min(abs(theta - PI_2), 2 * np.pi - abs(theta - PI_2))
            if err < 0.2:
                successes += 1; break
    return successes / n_trials


def exp4_ceiling():
    """Measure 1-step Oracle ceiling across environments."""
    print("\n" + "=" * 70)
    print("ABLATION 4: 1-Step Oracle Ceiling")
    print("=" * 70)

    results = {}
    for g in [10.0, 12.0, 15.0]:
        sr = oracle_1step_pendulum(g=g, n_trials=100)
        results[f'Pendulum g={g:.0f}'] = sr
        print(f"  Pendulum g={g:.0f}: Oracle 1-step SR = {sr*100:.0f}%")

    return results


# ═══════════════════════════════════════
# Main
# ═══════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=int, default=0, help='Experiment to run (0=all)')
    args = parser.parse_args()

    all_results = {}

    if args.exp in [0, 1]:
        r1, proto_mse, cws_mse = exp1_wm_comparison()
        all_results['wm_comparison'] = r1
        all_results['wm_mse'] = {'ProtoKAN': proto_mse, 'CWS-KAN': cws_mse}

    if args.exp in [0, 2]:
        r2 = exp2_accuracy_sweep()
        all_results['accuracy_sweep'] = r2

    if args.exp in [0, 3]:
        r3 = exp3_cartpole_analysis()
        all_results['cartpole_analysis'] = r3

    if args.exp in [0, 4]:
        r4 = exp4_ceiling()
        all_results['ceiling'] = r4

    # Save
    torch.save(all_results, '/tmp/ablation_results.pt')
    print(f"\nResults saved to /tmp/ablation_results.pt")


if __name__ == '__main__':
    main()
