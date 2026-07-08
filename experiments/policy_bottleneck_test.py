"""Isolate: is Policy the bottleneck for continual learning?

Compare after WM adaptation to g=15:
  1. Policy (BPTT retrained via adapted WM)
  2. MPC (H-step random shooting via adapted WM, no Policy)

If MPC >> Policy, Policy is the bottleneck.
"""
import torch, numpy as np, time, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kanrf import ProtoKAN
import torch.nn as nn
from control.kan_policy_net import KANPolicy
from control.bptt_trainer import BPTTTrainer
from experiments.cartpole_continual import (
    generate_wm_data, train_wm, generate_policy_states,
    X_S, XD_S, TH_S, THD_S, step_cartpole
)

S_TARGET = torch.zeros(1, 4)
device = 'cpu'


def mpc_action(wm, s_norm, H=3, N=300):
    """Random shooting via WM: sample N sequences of length H, pick best first action."""
    best_cost = float('inf')
    best_a = 0.0
    for _ in range(N):
        seq = np.random.uniform(-1, 1, H).astype(np.float32)
        s_cur = s_norm.clone().unsqueeze(0)
        total_cost = 0.0
        for t, a in enumerate(seq):
            a_t = torch.tensor([[a]], dtype=torch.float32)
            with torch.no_grad():
                s_cur = wm(torch.cat([s_cur, a_t], dim=-1))
            total_cost += (0.9 ** t) * (s_cur[:, 2].pow(2) + 0.1 * s_cur[:, 0].pow(2)).item()
        if total_cost < best_cost:
            best_cost = total_cost
            best_a = seq[0]
    return best_a


def evaluate_policy(trainer, g, n=20, label=''):
    succ = 0; steps = []
    for t in range(n):
        seed = 42 + t * 100; np.random.seed(seed)
        th = np.random.uniform(-0.05, 0.05)
        s = torch.tensor([[0., 0., th, 0.]], dtype=torch.float32)
        for step in range(500):
            sn = s.clone()
            sn[:, 0] /= X_S; sn[:, 1] /= XD_S; sn[:, 2] /= TH_S; sn[:, 3] /= THD_S
            a = trainer.get_action(sn[0].numpy())
            s = step_cartpole(s, torch.tensor([a]), g=g)
            if abs(s[0, 2].item()) > 0.21 or abs(s[0, 0].item()) > 2.4:
                break
        steps.append(step + 1)
        if step + 1 >= 500: succ += 1
    print(f"  [{label}] {succ}/{n} ({succ*100//n}%)  mean_steps={np.mean(steps):.0f}")
    return succ


def evaluate_mpc(wm, g, H=3, N=100, n=10, label=''):
    succ = 0; steps = []
    for t in range(n):
        seed = 42 + t * 100; np.random.seed(seed)
        th = np.random.uniform(-0.05, 0.05)
        s = torch.tensor([[0., 0., th, 0.]], dtype=torch.float32)
        for step in range(500):
            sn = s.clone()
            sn[:, 0] /= X_S; sn[:, 1] /= XD_S; sn[:, 2] /= TH_S; sn[:, 3] /= THD_S
            a = mpc_action(wm, sn[0], H=H, N=N)
            s = step_cartpole(s, torch.tensor([a]), g=g)
            if abs(s[0, 2].item()) > 0.21 or abs(s[0, 0].item()) > 2.4:
                break
        steps.append(step + 1)
        if step + 1 >= 500: succ += 1
    print(f"  [{label}] {succ}/{n} ({succ*100//n}%)  mean_steps={np.mean(steps):.0f}")
    return succ


def main():
    torch.manual_seed(42); np.random.seed(42)

    print("=" * 60)
    print("Bottleneck Test: Policy vs MPC after WM adaptation")
    print("=" * 60)

    # Train WM on g=9.8
    X, Y = generate_wm_data(g=9.8, n=5000, device=device)
    wm, _ = train_wm(X, Y, 'protokan', 80, device)
    s_pol = generate_policy_states(15000, device)

    # Train Policy on g=9.8
    print("\n[1] Train Policy on g=9.8")
    policy = KANPolicy(state_dim=4, action_dim=1, hidden_dim=12, n_layers=2)
    trainer = BPTTTrainer(wm, policy, S_TARGET, lr=1e-3, horizon=3, device=device)
    for ep in range(1, 101): trainer.train_epoch(s_pol)

    # g=9.8 baselines
    print("\n[2] g=9.8 baselines:")
    evaluate_policy(trainer, g=9.8, label='Policy_g=9.8')
    evaluate_mpc(wm, g=9.8, H=3, label='MPC_g=9.8')

    # g=15 before WM adaptation
    print("\n[3] g=15 BEFORE WM adaptation:")
    evaluate_policy(trainer, g=15.0, label='Policy_g=15_before')
    evaluate_mpc(wm, g=15.0, H=3, label='MPC_g=15_before')

    # WM adaptation
    print("\n[4] WM adaptation to g=15...")
    X_new, Y_new = generate_wm_data(g=15.0, n=500, device=device)
    for p in wm.parameters(): p.requires_grad = True
    opt_wm = torch.optim.Adam(wm.parameters(), lr=1e-3)
    mse_fn = nn.MSELoss()
    for step in range(1, 31):
        idx = np.random.choice(len(X_new), 64, replace=False)
        wm.train(); opt_wm.zero_grad()
        mse_fn(wm(X_new[idx]), Y_new[idx]).backward(); opt_wm.step()
    wm.eval()
    for p in wm.parameters(): p.requires_grad = False
    print("  Done")

    # MPC with adapted WM (no Policy needed)
    print("\n[5] MPC with ADAPTED WM:")
    evaluate_mpc(wm, g=15.0, H=3, label='MPC_g=15_adapted')
    evaluate_mpc(wm, g=9.8, H=3, label='MPC_g=9.8_preserved')

    # Policy recovery via BPTT
    print("\n[6] Policy recovery (BPTT via adapted WM):")
    for ep in range(1, 51):
        trainer.train_epoch(s_pol)
    print("  Done")
    evaluate_policy(trainer, g=15.0, label='Policy_g=15_recovered')
    evaluate_policy(trainer, g=9.8, label='Policy_g=9.8_preserved')


if __name__ == '__main__':
    main()
