"""BPTT Policy Training: H-step WM rollout."""
import torch, numpy as np, time, sys, os, gymnasium as gym
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kanrf import ProtoKAN
from control.kan_policy_net import KANPolicy
from control.bptt_trainer import BPTTTrainer

PI_2 = np.pi / 2
S_TARGET = torch.tensor([[0.0, 1.0, 0.0]])
S_TARGET_CP = torch.zeros(1, 4)


def test_pendulum():
    from experiments.baseline_sweep import (generate_pendulum_data, train_wm,
                                              generate_policy_states, evaluate_policy)
    device = 'cpu'; torch.manual_seed(42); np.random.seed(42)
    print("=" * 70)
    print("BPTT: Pendulum")
    print("=" * 70)
    X, Y = generate_pendulum_data(5000, seed=42)
    wm, _ = train_wm(X.to(device), Y.to(device))
    s_pol = generate_policy_states(10000, seed=42).to(device)

    for H in [3, 5]:
        torch.manual_seed(42); np.random.seed(42)
        policy = KANPolicy(state_dim=3, action_dim=1, hidden_dim=12, n_layers=2)
        trainer = BPTTTrainer(wm, policy, S_TARGET, lr=1e-3, horizon=H, device=device)
        t0 = time.time()
        for ep in range(1, 101):
            trainer.train_epoch(s_pol, batch_size=256)
            if ep % 30 == 0:
                print(f"  H={H} Epoch {ep:3d}  loss={trainer.loss_history[-1]['total']:.4f}")
        s, st, er = evaluate_policy(trainer)
        print(f"  H={H}: {s}/10  steps={np.mean(st):.0f}  err={np.mean(er):.3f}  "
              f"time={time.time()-t0:.0f}s")


def test_cartpole():
    from experiments.cartpole_continual import (
        generate_wm_data, train_wm, generate_policy_states,
        X_S, XD_S, TH_S, THD_S, step_cartpole
    )
    device = 'cpu'; torch.manual_seed(42); np.random.seed(42)
    print("\n" + "=" * 70)
    print("BPTT: CartPole")
    print("=" * 70)
    X, Y = generate_wm_data(g=9.8, n=5000, device=device)
    wm, _ = train_wm(X, Y, 'protokan', 80, device)
    s_pol = generate_policy_states(15000, device)

    for H in [3, 5]:
        torch.manual_seed(42); np.random.seed(42)
        policy = KANPolicy(state_dim=4, action_dim=1, hidden_dim=12, n_layers=2)
        trainer = BPTTTrainer(wm, policy, S_TARGET_CP, lr=1e-3, horizon=H, device=device)
        t0 = time.time()
        for ep in range(1, 101):
            trainer.train_epoch(s_pol, batch_size=256)
            if ep % 30 == 0:
                print(f"  H={H} Epoch {ep:3d}  loss={trainer.loss_history[-1]['total']:.4f}")
        succ = 0; steps = []
        for trial in range(20):
            seed = 42 + trial * 100; np.random.seed(seed)
            th = np.random.uniform(-0.05, 0.05)
            s_raw = torch.tensor([[0., 0., th, 0.]], dtype=torch.float32)
            for step in range(500):
                sn = s_raw.clone()
                sn[:, 0] /= X_S; sn[:, 1] /= XD_S; sn[:, 2] /= TH_S; sn[:, 3] /= THD_S
                a = trainer.get_action(sn[0].numpy())
                s_raw = step_cartpole(s_raw, torch.tensor([a]))
                if abs(s_raw[0, 2].item()) > 0.21 or abs(s_raw[0, 0].item()) > 2.4:
                    break
            steps.append(step + 1)
            if step + 1 >= 500: succ += 1
        print(f"  H={H}: {succ}/20 ({succ*5}%)  mean_steps={np.mean(steps):.0f}  "
              f"time={time.time()-t0:.0f}s")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='both')
    args = parser.parse_args()
    if args.env in ['pendulum', 'both']: test_pendulum()
    if args.env in ['cartpole', 'both']: test_cartpole()
