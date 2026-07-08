"""CartPole BPTT H=3: Continual Learning g=9.8 → 15."""
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
        if step + 1 >= 500:
            succ += 1
    sr = succ * 100 // n
    print(f"  [{label}] {succ}/{n} ({sr}%)  mean_steps={np.mean(steps):.0f}")
    return succ, steps


def main():
    torch.manual_seed(42); np.random.seed(42)

    print("=" * 60)
    print("CartPole BPTT H=3: Continual Learning g=9.8 -> 15")
    print("=" * 60)

    # Phase 1: Train on g=9.8
    print("\n[1] Train WM + Policy on g=9.8")
    X, Y = generate_wm_data(g=9.8, n=5000, device=device)
    wm, _ = train_wm(X, Y, 'protokan', 80, device)
    s_pol = generate_policy_states(15000, device)

    policy = KANPolicy(state_dim=4, action_dim=1, hidden_dim=12, n_layers=2)
    trainer = BPTTTrainer(wm, policy, S_TARGET, lr=1e-3, horizon=3, device=device)
    t0 = time.time()
    for ep in range(1, 101):
        trainer.train_epoch(s_pol)
        if ep % 30 == 0:
            print(f"  Policy epoch {ep:3d}  loss={trainer.loss_history[-1]['total']:.4f}")
    print(f"  Trained in {time.time()-t0:.0f}s")

    # Phase 2: Before switch
    print("\n[2] Before gravity switch:")
    s10, _ = evaluate(trainer, g=9.8, label='g=9.8')
    s15_before, _ = evaluate(trainer, g=15.0, label='g=15_before')

    # Phase 3: WM adaptation
    print("\n[3] WM adaptation to g=15 (30 online steps)...")
    X_new, Y_new = generate_wm_data(g=15.0, n=500, device=device)
    for p in wm.parameters():
        p.requires_grad = True
    opt_wm = torch.optim.Adam(wm.parameters(), lr=1e-3)
    mse_fn = nn.MSELoss()
    for step in range(1, 31):
        idx = np.random.choice(len(X_new), 64, replace=False)
        wm.train(); opt_wm.zero_grad()
        mse_fn(wm(X_new[idx]), Y_new[idx]).backward()
        opt_wm.step()
    wm.eval()
    for p in wm.parameters():
        p.requires_grad = False
    with torch.no_grad():
        nerr = mse_fn(wm(X_new), Y_new).item()
        oerr = mse_fn(wm(X[:500].to(device)), Y[:500].to(device)).item()
    print(f"  new_mse={nerr:.6f}  old_mse={oerr:.6f}")

    # Phase 4: Policy recovery
    print("\n[4] Policy recovery (BPTT via adapted WM, 50 epochs)...")
    for ep in range(1, 51):
        trainer.train_epoch(s_pol)
        if ep % 20 == 0:
            print(f"  Recover epoch {ep:3d}  loss={trainer.loss_history[-1]['total']:.4f}")

    # Phase 5: After recovery
    print("\n[5] After recovery:")
    s15_after, _ = evaluate(trainer, g=15.0, label='g=15_after')
    s10_after, _ = evaluate(trainer, g=9.8, label='g=9.8_preserved')

    print(f"\n{'='*60}")
    print(f"SUMMARY:")
    print(f"  g=9.8 initial:      {s10}/20")
    print(f"  g=15 before adapt:  {s15_before}/20")
    print(f"  g=15 after recover: {s15_after}/20")
    print(f"  g=9.8 preserved:    {s10_after}/20")


if __name__ == '__main__':
    main()
