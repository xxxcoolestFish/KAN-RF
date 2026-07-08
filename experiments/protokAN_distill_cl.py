"""ProtoKAN Policy + MPC Distillation: continual learning test."""
import torch, numpy as np, time, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kanrf import ProtoKAN
import torch.nn as nn
from control.protokAN_distill import ProtoKANPolicy, DistillationTrainer
from experiments.cartpole_continual import (
    generate_wm_data, train_wm, generate_policy_states,
    X_S, XD_S, TH_S, THD_S, step_cartpole
)

S_TARGET = torch.zeros(1, 4)
device = 'cpu'


def evaluate(trainer, g, n=20, label=''):
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


def main():
    torch.manual_seed(42); np.random.seed(42)

    print("=" * 60)
    print("ProtoKAN Policy + MPC Distillation: Continual Learning")
    print("=" * 60)

    # Train WM on g=9.8
    print("\n[1] Train ProtoKAN WM on g=9.8")
    X, Y = generate_wm_data(g=9.8, n=5000, device=device)
    wm, _ = train_wm(X, Y, 'protokan', 80, device)
    s_pol = generate_policy_states(15000, device)

    # Train ProtoKAN Policy via distillation
    print("\n[2] Train ProtoKAN Policy via MPC distillation")
    policy = ProtoKANPolicy(state_dim=4, hidden_dim=12, init_log_sigma=-1.5)
    trainer = DistillationTrainer(wm, policy, s_pol, S_TARGET,
                                   H=3, N=200, n_demos=2000, device=device)
    t0 = time.time()
    for ep in range(1, 51):  # 50 epochs distillation
        ld = trainer.train_epoch()
        if ep % 15 == 0:
            print(f"  Distill epoch {ep:3d}  loss={ld['total']:.4f}")
    print(f"  Trained in {time.time()-t0:.0f}s")

    # Baselines
    print("\n[3] Baselines:")
    evaluate(trainer, g=9.8, label='g=9.8')
    s15_before = evaluate(trainer, g=15.0, label='g=15_before')

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

    # Policy fine-tuning via distillation with adapted WM
    print("\n[5] Fine-tune Policy (low LR, local updates via ProtoKAN)")
    s_pol_new = generate_policy_states(5000, device)
    trainer.fine_tune(wm, s_pol_new, S_TARGET, n_demos=500, epochs=30, lr=5e-4)

    # Final evaluation
    print("\n[6] After fine-tuning:")
    evaluate(trainer, g=15.0, label='g=15_after')
    evaluate(trainer, g=9.8, label='g=9.8_preserved')

    print(f"\n{'='*60}")
    print(f"SUMMARY: g=15 {s15_before}/20 → after = see above")


if __name__ == '__main__':
    main()
