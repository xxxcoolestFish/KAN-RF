"""Test THTP on Pendulum and CartPole."""
import torch, numpy as np, time, sys, os, gymnasium as gym
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kanrf import ProtoKAN
from control.kan_policy_net import KANPolicy
from control.thtp import TemporalHierarchy, TargetPropagation, THTPTrainer

PI_2 = np.pi / 2
S_TARGET = torch.tensor([[0.0, 1.0, 0.0]])
S_TARGET_CP = torch.zeros(1, 4)


# ═══════════════════════════════════════
# Pendulum
# ═══════════════════════════════════════

def test_pendulum():
    from experiments.baseline_sweep import (
        generate_pendulum_data, train_wm, generate_policy_states, evaluate_policy
    )
    device = 'cpu'
    torch.manual_seed(42); np.random.seed(42)

    print("=" * 70)
    print("THTP TEST: Pendulum")
    print("=" * 70)

    # Train WM
    X, Y = generate_pendulum_data(5000, seed=42)
    X, Y = X.to(device), Y.to(device)
    wm, wm_val = train_wm(X, Y)
    print(f"WM val_mse={wm_val:.6f}")

    # Discover hierarchy
    print("\n[1] Temporal Hierarchy Discovery:")
    names = ['cosθ', 'sinθ', 'θ̇']
    hier = TemporalHierarchy(wm, 3, n_samples=200, device=device)
    print(hier.summary(names))

    # Setup THTP
    thtp = TargetPropagation(wm, hier, alpha=0.3, damping=0.1, device=device)

    # Test THTP on a few states
    print("\n[2] THTP routing examples:")
    for label, s_vec in [('bottom', [0.0, -1.0, 0.0]),
                           ('mid-right', [0.0, 1.0, 0.0]),
                           ('swing', [1.0, 0.0, 0.5])]:
        s_t = torch.tensor(s_vec, dtype=torch.float32, device=device)
        a_des, subgoals, diag = thtp.propagate(s_t, S_TARGET.squeeze(0))
        print(f"  {label:10s}: a_des={a_des.item():+.3f}  "
              f"error_t0={diag['error_tier0']}")

    # Train Policy with THTP
    print("\n[3] Training KAN Policy with THTP...")
    s_pol = generate_policy_states(10000, seed=42).to(device)
    policy = KANPolicy(state_dim=3, action_dim=1, hidden_dim=12, n_layers=2).to(device)
    trainer = THTPTrainer(wm, policy, hier, thtp, s_pol, S_TARGET,
                          lr=1e-3, n_distill=500, device=device)
    for ep in range(1, 201):
        ld = trainer.train_epoch(s_pol)
        if ep % 40 == 0:
            print(f"  Epoch {ep:3d}  total={ld['total']:.4f}")

    # Evaluate
    print("\n[4] Evaluate:")
    s, st, er = evaluate_policy(trainer)
    print(f"  Pendulum THTP: {s}/10  steps={np.mean(st):.0f}  err={np.mean(er):.3f}")


# ═══════════════════════════════════════
# CartPole
# ═══════════════════════════════════════

def test_cartpole():
    from experiments.cartpole_continual import (
        generate_wm_data, train_wm, generate_policy_states,
        X_S, XD_S, TH_S, THD_S, step_cartpole
    )
    device = 'cpu'
    torch.manual_seed(42); np.random.seed(42)

    print("\n" + "=" * 70)
    print("THTP TEST: CartPole")
    print("=" * 70)

    # Train WM
    X, Y = generate_wm_data(g=9.8, n=5000, device=device)
    wm, wm_val = train_wm(X, Y, 'protokan', 80, device)
    print(f"WM val_mse={wm_val:.6f}")

    # Discover hierarchy
    print("\n[1] Temporal Hierarchy Discovery:")
    names = ['x', 'ẋ', 'θ', 'θ̇']
    hier = TemporalHierarchy(wm, 4, n_samples=200, device=device)
    print(hier.summary(names))

    # Setup THTP
    thtp = TargetPropagation(wm, hier, alpha=0.3, damping=0.1, device=device)

    # Policy
    print("\n[2] Training KAN Policy with THTP...")
    s_pol = generate_policy_states(15000, device)
    policy = KANPolicy(state_dim=4, action_dim=1, hidden_dim=12, n_layers=2).to(device)
    trainer = THTPTrainer(wm, policy, hier, thtp, s_pol, S_TARGET_CP,
                          lr=1e-3, n_distill=500, device=device)
    for ep in range(1, 201):
        ld = trainer.train_epoch(s_pol)
        if ep % 40 == 0:
            print(f"  Epoch {ep:3d}  total={ld['total']:.4f}")

    # Evaluate
    print("\n[3] Evaluate:")
    succ = 0; steps = []
    for trial in range(20):
        seed = 42 + trial * 100
        np.random.seed(seed)
        th = np.random.uniform(-0.05, 0.05)
        s_raw = torch.tensor([[0.0, 0.0, th, 0.0]], dtype=torch.float32)
        for step in range(500):
            s_n = s_raw.clone()
            s_n[:, 0] /= X_S; s_n[:, 1] /= XD_S
            s_n[:, 2] /= TH_S; s_n[:, 3] /= THD_S
            a_norm = trainer.get_action(s_n[0].numpy())
            s_raw = step_cartpole(s_raw, torch.tensor([a_norm]))
            theta, x = s_raw[0, 2].item(), s_raw[0, 0].item()
            if abs(theta) > 0.21 or abs(x) > 2.4: break
        steps.append(step + 1)
        if step + 1 >= 500: succ += 1
    print(f"  CartPole THTP: {succ}/20 ({succ*5}%)  mean_steps={np.mean(steps):.0f}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='both',
                        choices=['pendulum', 'cartpole', 'both'])
    args = parser.parse_args()

    if args.env in ['pendulum', 'both']:
        test_pendulum()
    if args.env in ['cartpole', 'both']:
        test_cartpole()
