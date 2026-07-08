"""CartPole BPTT: multi-seed verification + continual learning."""
import torch, numpy as np, time, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kanrf import ProtoKAN
from control.kan_policy_net import KANPolicy
from control.bptt_trainer import BPTTTrainer
from experiments.cartpole_continual import (
    generate_wm_data, train_wm, generate_policy_states,
    X_S, XD_S, TH_S, THD_S, step_cartpole
)

S_TARGET_CP = torch.zeros(1, 4)
device = 'cpu'


def evaluate(trainer, g=9.8, n_trials=20, label=''):
    succ = 0; steps = []
    for trial in range(n_trials):
        seed = 42 + trial * 100; np.random.seed(seed)
        th = np.random.uniform(-0.05, 0.05)
        s_raw = torch.tensor([[0., 0., th, 0.]], dtype=torch.float32)
        for step in range(500):
            sn = s_raw.clone()
            sn[:, 0] /= X_S; sn[:, 1] /= XD_S; sn[:, 2] /= TH_S; sn[:, 3] /= THD_S
            a = trainer.get_action(sn[0].numpy())
            s_raw = step_cartpole(s_raw, torch.tensor([a]), g=g)
            if abs(s_raw[0, 2].item()) > 0.21 or abs(s_raw[0, 0].item()) > 2.4:
                break
        steps.append(step + 1)
        if step + 1 >= 500: succ += 1
    print(f"  [{label}] {succ}/{n_trials} ({succ*100/n_trials:.0f}%)  "
          f"mean_steps={np.mean(steps):.0f}")
    return succ, steps


# ── 1. Multi-seed verification ──
print("=" * 70)
print("CartPole BPTT H=3: 10-seed verification")
print("=" * 70)
X, Y = generate_wm_data(g=9.8, n=5000, device=device)
wm, _ = train_wm(X, Y, 'protokan', 80, device)
s_pol = generate_policy_states(15000, device)

results = []
for si in range(10):
    pol_seed = 100 + si
    torch.manual_seed(pol_seed); np.random.seed(pol_seed)
    policy = KANPolicy(state_dim=4, action_dim=1, hidden_dim=12, n_layers=2)
    trainer = BPTTTrainer(wm, policy, S_TARGET_CP, lr=1e-3, horizon=3, device=device)
    for ep in range(1, 101):
        trainer.train_epoch(s_pol)
    succ, steps = evaluate(trainer, g=9.8, label=f'seed{pol_seed}')
    results.append(succ)
sr = [r / 20 for r in results]
print(f"\n  BPTT H=3 Summary: {np.mean(sr)*100:.0f}% +/- {np.std(sr)*100:.0f}%  "
      f"range=[{min(sr)*100:.0f}%, {max(sr)*100:.0f}%]  "
      f"seeds_100%={sum(1 for r in results if r==20)}/10")

# ── 2. Continual learning ──
print("\n" + "=" * 70)
print("CartPole BPTT H=3: Continual Learning g=9.8→15")
print("=" * 70)

# Train on g=9.8
torch.manual_seed(42); np.random.seed(42)
policy = KANPolicy(state_dim=4, action_dim=1, hidden_dim=12, n_layers=2)
trainer = BPTTTrainer(wm, policy, S_TARGET_CP, lr=1e-3, horizon=3, device=device)
for ep in range(1, 101):
    trainer.train_epoch(s_pol)

print("  Before adaptation:")
evaluate(trainer, g=9.8, label='g=9.8')
evaluate(trainer, g=15.0, label='g=15')

# WM adaptation
print("\n  Adapting WM to g=15...")
X_new, Y_new = generate_wm_data(g=15.0, n=500, device=device)
for p in wm.parameters(): p.requires_grad = True
import torch.nn as nn
mse_fn = nn.MSELoss()
opt = torch.optim.Adam(wm.parameters(), lr=1e-3)
for step in range(1, 31):
    idx = np.random.choice(len(X_new), 64, replace=False)
    wm.train(); opt.zero_grad()
    loss = mse_fn(wm(X_new[idx]), Y_new[idx])
    loss.backward(); opt.step()
wm.eval()
for p in wm.parameters(): p.requires_grad = False
with torch.no_grad():
    print(f"  new_mse={mse_fn(wm(X_new), Y_new).item():.6f}  "
          f"old_mse={mse_fn(wm(X[:500].to(device)), Y[:500].to(device)).item():.6f}")

# Policy recovery via BPTT with adapted WM
print("\n  Policy recovery (BPTT via adapted WM)...")
for ep in range(1, 51):
    trainer.train_epoch(s_pol)
    if ep % 20 == 0:
        print(f"    Epoch {ep:3d}  loss={trainer.loss_history[-1]['total']:.4f}")

print("\n  After adaptation:")
evaluate(trainer, g=9.8, label='g=9.8_preserved')
evaluate(trainer, g=15.0, label='g=15_recovered')
