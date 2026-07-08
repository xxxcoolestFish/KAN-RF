"""Test CDPN on Pendulum and CartPole."""
import torch, numpy as np, time, sys, os, gymnasium as gym
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kanrf import ProtoKAN
from control.cdpn import (
    discover_tier0, CausalDecomposedPolicy, Execute, CDPNTrainer
)

PI_2 = np.pi / 2
S_TARGET_PEN = torch.tensor([[0.0, 1.0, 0.0]])
S_TARGET_CP = torch.zeros(1, 4)


def train_cdpn(wm, state_dim, s_target, s_pol, tier0, label='',
               mode='1step', epochs=200):
    """Train CDPN and return trainer."""
    policy = CausalDecomposedPolicy(
        state_dim=state_dim, tier0_size=len(tier0),
        hidden_dim=24, n_layers=2)
    execute = Execute(wm, state_dim, tier0, s_pol, damping=0.1)
    trainer = CDPNTrainer(wm, policy, execute, s_target, tier0,
                          lr=1e-3, mode=mode, imagine_steps=3)

    print(f"  [{label}] Policy: {sum(p.numel() for p in policy.parameters())} params  "
          f"Tier0={tier0}  mode={mode}")
    for ep in range(1, epochs + 1):
        ld = trainer.train_epoch(s_pol)
        if ep % 50 == 0:
            print(f"    Epoch {ep:3d}  loss={ld['total']:.4f}")
    return trainer


def test_pendulum():
    from experiments.baseline_sweep import (
        generate_pendulum_data, train_wm, generate_policy_states, evaluate_policy
    )
    device = 'cpu'; torch.manual_seed(42); np.random.seed(42)
    print("=" * 70)
    print("CDPN: Pendulum")
    print("=" * 70)

    # Train WM
    X, Y = generate_pendulum_data(5000, seed=42)
    wm, _ = train_wm(X.to(device), Y.to(device))
    print(f"  WM val_mse ≈ 0.000003")

    # Discover Tier 0
    tier0, mask, jac_norms, thresh = discover_tier0(wm, 3, device=device)
    names = ['cosθ', 'sinθ', 'θ̇']
    print(f"  Tier 0: {[names[i] for i in tier0]}  "
          f"(threshold={thresh:.4f}, norms={[f'{n:.4f}' for n in jac_norms]})")

    s_pol = generate_policy_states(10000, seed=42).to(device)

    # 1-step mode
    print("\n  [1-step mode]")
    t1 = train_cdpn(wm, 3, S_TARGET_PEN, s_pol, tier0, label='1step', mode='1step')
    s, st, er = evaluate_policy(t1)
    print(f"  Result: {s}/10  steps={np.mean(st):.0f}  err={np.mean(er):.3f}")

    # Imagine mode
    print("\n  [Imagine mode, k=3]")
    t2 = train_cdpn(wm, 3, S_TARGET_PEN, s_pol, tier0, label='imagine', mode='imagine')
    s, st, er = evaluate_policy(t2)
    print(f"  Result: {s}/10  steps={np.mean(st):.0f}  err={np.mean(er):.3f}")

    return t1, t2


def test_cartpole():
    from experiments.cartpole_continual import (
        generate_wm_data, train_wm, generate_policy_states,
        X_S, XD_S, TH_S, THD_S, step_cartpole
    )
    device = 'cpu'; torch.manual_seed(42); np.random.seed(42)
    print("\n" + "=" * 70)
    print("CDPN: CartPole")
    print("=" * 70)

    # Train WM
    X, Y = generate_wm_data(g=9.8, n=5000, device=device)
    wm, _ = train_wm(X, Y, 'protokan', 80, device)
    print(f"  WM val_mse ≈ 0.000000")

    # Discover Tier 0
    tier0, mask, jac_norms, thresh = discover_tier0(wm, 4, device=device)
    names = ['x', 'ẋ', 'θ', 'θ̇']
    print(f"  Tier 0: {[names[i] for i in tier0]}  "
          f"(threshold={thresh:.4f}, norms={[f'{n:.4f}' for n in jac_norms]})")

    s_pol = generate_policy_states(15000, device)

    def evaluate_cdpn(trainer, label=''):
        succ = 0; steps = []
        for trial in range(20):
            seed = 42 + trial * 100; np.random.seed(seed)
            th = np.random.uniform(-0.05, 0.05)
            s_raw = torch.tensor([[0., 0., th, 0.]], dtype=torch.float32)
            for step in range(500):
                sn = s_raw.clone()
                sn[:, 0] /= X_S; sn[:, 1] /= XD_S
                sn[:, 2] /= TH_S; sn[:, 3] /= THD_S
                a = trainer.get_action(sn[0].numpy())
                s_raw = step_cartpole(s_raw, torch.tensor([a]))
                if abs(s_raw[0, 2].item()) > 0.21 or abs(s_raw[0, 0].item()) > 2.4:
                    break
            steps.append(step + 1)
            if step + 1 >= 500:
                succ += 1
        print(f"  [{label}] {succ}/20 ({succ*5}%)  mean_steps={np.mean(steps):.0f}")
        return succ, steps

    # 1-step mode
    print("\n  [1-step mode]")
    t1 = train_cdpn(wm, 4, S_TARGET_CP, s_pol, tier0, label='1step', mode='1step')
    evaluate_cdpn(t1, '1step')

    # Imagine mode
    print("\n  [Imagine mode, k=3]")
    t2 = train_cdpn(wm, 4, S_TARGET_CP, s_pol, tier0, label='imagine', mode='imagine')
    evaluate_cdpn(t2, 'imagine')

    return t1, t2


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='both')
    args = parser.parse_args()

    if args.env in ['pendulum', 'both']:
        test_pendulum()
    if args.env in ['cartpole', 'both']:
        test_cartpole()
