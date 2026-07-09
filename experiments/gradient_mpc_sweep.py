"""Batch shooting MPC: broader validation across environments and horizons."""
import torch, numpy as np, time, sys, os, gymnasium as gym
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kanrf import ProtoKAN
import torch.nn as nn
from control.gradient_mpc import GradientMPC
from control.lyapunov_bptt import synthesize_lyapunov

PI_2 = np.pi / 2


# ═══════════════════════════════════════
# CartPole
# ═══════════════════════════════════════

def test_cartpole():
    from experiments.cartpole_continual import (
        generate_wm_data, train_wm, X_S, XD_S, TH_S, THD_S, step_cartpole
    )
    device = 'cpu'; torch.manual_seed(42); np.random.seed(42)

    def evaluate(mpc, g, n=20, label=''):
        succ = 0; steps = []; t0 = time.time()
        for t in range(n):
            seed = 42 + t * 100; np.random.seed(seed)
            th = np.random.uniform(-0.05, 0.05)
            s = torch.tensor([[0., 0., th, 0.]], dtype=torch.float32)
            for step in range(500):
                sn = s.clone()
                sn[:, 0] /= X_S; sn[:, 1] /= XD_S; sn[:, 2] /= TH_S; sn[:, 3] /= THD_S
                a = mpc.get_action(sn[0])
                s = step_cartpole(s, torch.tensor([a]), g=g)
                if abs(s[0, 2].item()) > 0.21 or abs(s[0, 0].item()) > 2.4:
                    break
            steps.append(step + 1)
            if step + 1 >= 500: succ += 1
        print(f"  [{label}] {succ}/{n} ({succ*100//n}%)  "
              f"mean_steps={np.mean(steps):.0f}  time={time.time()-t0:.0f}s")
        return succ, steps

    print("=" * 70)
    print("Batch Shooting MPC: CartPole")
    print("=" * 70)

    # Train WM on g=9.8
    X, Y = generate_wm_data(g=9.8, n=5000, device=device)
    wm, _ = train_wm(X, Y, 'protokan', 80, device)

    # Synthesize Lyapunov P matrix
    s_pol = torch.cat([X[:, :4], torch.randn(5000, 4) * 0.5], dim=0)[:5000].to(device)
    S_TARGET = torch.zeros(1, 4, device=device)
    P, _, _, _ = synthesize_lyapunov(wm, s_pol, S_TARGET, 4, device=device)
    print(f"  P diag: {[f'{x:.1f}' for x in P.diag().tolist()]}")

    # Sweep horizon
    print("\n[1] Horizon sweep (g=9.8, 10 trials each):")
    for H in [2, 3, 4, 5]:
        mpc = GradientMPC(wm, 4, P=P, horizon=H, n_shoot=500, mode='shoot', device=device)
        evaluate(mpc, g=9.8, n=10, label=f'H={H}')

    # Full 20-trial evaluation with H=3
    print("\n[2] Full evaluation (H=3, 20 trials):")
    mpc = GradientMPC(wm, 4, P=P, horizon=3, n_shoot=500, mode='shoot', device=device)
    s10, _ = evaluate(mpc, g=9.8, n=20, label='g=9.8')
    s15_before, _ = evaluate(mpc, g=15.0, n=20, label='g=15_before')

    # WM adaptation
    print("\n[3] WM adaptation to g=15 (30 steps)...")
    X_new, Y_new = generate_wm_data(g=15.0, n=500, device=device)
    for p in wm.parameters(): p.requires_grad = True
    opt = torch.optim.Adam(wm.parameters(), lr=1e-3)
    ms = nn.MSELoss()
    for _ in range(30):
        idx = np.random.choice(len(X_new), 64, replace=False)
        wm.train(); opt.zero_grad()
        ms(wm(X_new[idx]), Y_new[idx]).backward(); opt.step()
    wm.eval()
    for p in wm.parameters(): p.requires_grad = False
    print("  Done")

    # After adaptation
    print("\n[4] After WM adaptation:")
    s15_after, _ = evaluate(mpc, g=15.0, n=20, label='g=15_after')
    s10_after, _ = evaluate(mpc, g=9.8, n=20, label='g=9.8_preserved')

    print(f"\n  SUMMARY:")
    print(f"  g=9.8:          {s10}/20")
    print(f"  g=15 before:    {s15_before}/20")
    print(f"  g=15 after:     {s15_after}/20")
    print(f"  g=9.8 preserved: {s10_after}/20")


# ═══════════════════════════════════════
# Pendulum
# ═══════════════════════════════════════

def test_pendulum():
    from experiments.baseline_sweep import generate_pendulum_data, train_wm
    device = 'cpu'; torch.manual_seed(42); np.random.seed(42)

    def pendulum_step_np(theta, thd, u, g=10.0):
        thd_new = thd + (g * np.sin(theta) + u) * 0.05
        th_new = theta + thd_new * 0.05
        return th_new, thd_new

    def evaluate_pendulum(mpc, g=10.0, n=20, label=''):
        succ = 0; steps = []; errs = []; t0 = time.time()
        for t in range(n):
            seed = 42 + t * 100; np.random.seed(seed)
            theta = np.random.uniform(-np.pi, np.pi)
            thd = np.random.uniform(-1.0, 1.0)
            for step in range(300):
                s_n = torch.tensor([np.cos(theta), np.sin(theta), thd / 8.0],
                                   dtype=torch.float32)
                a_norm = mpc.get_action(s_n)
                u = a_norm * 2.0
                theta, thd = pendulum_step_np(theta, thd, u, g=g)
                err = min(abs(theta - PI_2), 2 * np.pi - abs(theta - PI_2))
                if err < 0.2:
                    succ += 1; steps.append(step + 1); errs.append(err); break
            else:
                steps.append(300)
                errs.append(min(abs(theta - PI_2), 2 * np.pi - abs(theta - PI_2)))
        print(f"  [{label}] {succ}/{n} ({succ*100//n}%)  "
              f"mean_steps={np.mean(steps):.0f}  time={time.time()-t0:.0f}s")
        return succ

    print("\n" + "=" * 70)
    print("Batch Shooting MPC: Pendulum")
    print("=" * 70)

    # Train WM on g=10
    X, Y = generate_pendulum_data(5000, seed=42)
    wm, _ = train_wm(X.to(device), Y.to(device))

    # Synthesize Lyapunov P
    s_pol = torch.randn(5000, 3, device=device) * 0.5
    s_pol[:, :2].clamp_(-1, 1)
    S_TARGET_PEN = torch.tensor([[0., 1., 0.]], device=device)
    P, _, _, _ = synthesize_lyapunov(wm, s_pol, S_TARGET_PEN, 3, device=device)
    print(f"  P diag: {[f'{x:.1f}' for x in P.diag().tolist()]}")

    # Sweep horizon
    print("\n[1] Horizon sweep (g=10, 10 trials each):")
    for H in [2, 3, 4, 5]:
        mpc = GradientMPC(wm, 3, P=P, horizon=H, n_shoot=500, mode='shoot', device=device)
        evaluate_pendulum(mpc, g=10.0, n=10, label=f'H={H}')

    # Full evaluation
    print("\n[2] Full evaluation (H=4, 20 trials):")
    mpc = GradientMPC(wm, 3, P=P, horizon=4, n_shoot=500, mode='shoot', device=device)
    s10 = evaluate_pendulum(mpc, g=10.0, n=20, label='g=10')
    s15_before = evaluate_pendulum(mpc, g=15.0, n=20, label='g=15_before')

    # WM adaptation
    print("\n[3] WM adaptation to g=15...")
    X_new, Y_new = generate_pendulum_data(500, seed=123)
    X_new, Y_new = X_new.to(device), Y_new.to(device)
    for p in wm.parameters(): p.requires_grad = True
    opt = torch.optim.Adam(wm.parameters(), lr=1e-3)
    ms = nn.MSELoss()
    for _ in range(30):
        idx = np.random.choice(len(X_new), 64, replace=False)
        wm.train(); opt.zero_grad()
        ms(wm(X_new[idx]), Y_new[idx]).backward(); opt.step()
    wm.eval()
    for p in wm.parameters(): p.requires_grad = False
    print("  Done")

    print("\n[4] After WM adaptation:")
    s15_after = evaluate_pendulum(mpc, g=15.0, n=20, label='g=15_after')
    s10_after = evaluate_pendulum(mpc, g=10.0, n=20, label='g=10_preserved')

    print(f"\n  SUMMARY:")
    print(f"  g=10:           {s10}/20")
    print(f"  g=15 before:    {s15_before}/20")
    print(f"  g=15 after:     {s15_after}/20")
    print(f"  g=10 preserved: {s10_after}/20")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='both')
    args = parser.parse_args()
    if args.env in ['cartpole', 'both']: test_cartpole()
    if args.env in ['pendulum', 'both']: test_pendulum()
