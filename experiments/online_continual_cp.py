"""Online continual learning: pre-train on g=9.8, then deploy on g=15
with per-step WM + Policy updates. No manual intervention."""
import torch, torch.nn as nn, numpy as np, time, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kanrf import ProtoKAN
from control.protokAN_distill import ProtoKANPolicy, DistillationTrainer
from experiments.cartpole_continual import (
    generate_wm_data, train_wm, generate_policy_states,
    X_S, XD_S, TH_S, THD_S, step_cartpole
)

S_TARGET = torch.zeros(1, 4)
device = 'cpu'


def main():
    torch.manual_seed(42); np.random.seed(42)

    print("=" * 60)
    print("Online Continual Learning: g=9.8 pre-train → g=15 online")
    print("=" * 60)

    # 1. Pre-train WM + Policy on g=9.8
    print("\n[1] Pre-train WM + Policy on g=9.8...")
    X, Y = generate_wm_data(g=9.8, n=5000, device=device)
    wm, _ = train_wm(X, Y, 'protokan', 80, device)
    s_pol = generate_policy_states(15000, device)
    policy = ProtoKANPolicy(state_dim=4, hidden_dim=12, init_log_sigma=-1.5)
    trainer = DistillationTrainer(wm, policy, s_pol, S_TARGET,
                                   H=3, N=200, n_demos=2000, device=device)
    for ep in range(1, 51): trainer.train_epoch()

    # 2. Online: run episodes on g=15, update WM and Policy per step
    print("\n[2] Online phase: running on g=15 with per-step updates...")
    wm.train()
    for p in wm.parameters(): p.requires_grad = True
    for p in policy.parameters(): p.requires_grad = True
    opt_wm = torch.optim.Adam(wm.parameters(), lr=1e-3)
    opt_pol = torch.optim.Adam(policy.parameters(), lr=1e-4)
    mse_fn = nn.MSELoss()

    episode_steps = []
    g = 15.0

    for ep in range(1, 31):
        seed = 42 + ep * 100; np.random.seed(seed)
        th = np.random.uniform(-0.05, 0.05)
        s_raw = torch.tensor([[0., 0., th, 0.]], dtype=torch.float32)

        for step in range(500):
            # Policy forward
            s_n = s_raw.clone()
            s_n[:, 0] /= X_S; s_n[:, 1] /= XD_S; s_n[:, 2] /= TH_S; s_n[:, 3] /= THD_S
            with torch.no_grad():
                a_norm = policy(s_n).item()

            # Environment step
            a_t = torch.tensor([[a_norm]], dtype=torch.float32)
            s_next = step_cartpole(s_raw, a_t, g=g)
            sn_next = s_next.clone()
            sn_next[:, 0] /= X_S; sn_next[:, 1] /= XD_S
            sn_next[:, 2] /= TH_S; sn_next[:, 3] /= THD_S

            # WM online update
            wm_in = torch.cat([s_n, a_t], dim=-1)
            opt_wm.zero_grad()
            wm_loss = mse_fn(wm(wm_in), sn_next)
            wm_loss.backward()
            opt_wm.step()

            # Policy online update (via updated WM gradient)
            opt_pol.zero_grad()
            a_pol = policy(s_n)
            s_pred = wm(torch.cat([s_n, a_pol], dim=-1))
            pol_loss = (s_pred[:, 2].pow(2) + 0.1 * s_pred[:, 0].pow(2) +
                        0.5 * s_pred[:, 3].pow(2)).mean()
            pol_loss.backward()
            opt_pol.step()

            s_raw = s_next
            theta, x = s_raw[0, 2].item(), s_raw[0, 0].item()
            if abs(theta) > 0.21 or abs(x) > 2.4:
                break

        episode_steps.append(step + 1)
        ok = step + 1 >= 500
        avg10 = np.mean(episode_steps[-10:]) if len(episode_steps) >= 10 else 0
        print(f"  Ep {ep:3d}  {'✓' if ok else '✗'}  steps={step+1:3d}  "
              f"avg10={avg10:.0f}  wm_loss={wm_loss.item():.5f}  pol_loss={pol_loss.item():.4f}")

    # Summary
    early = np.mean(episode_steps[:5])
    late = np.mean(episode_steps[-5:])
    successes = sum(1 for s in episode_steps if s >= 500)
    print(f"\n  Early episodes (1-5):  mean_steps={early:.0f}")
    print(f"  Late episodes (26-30): mean_steps={late:.0f}")
    print(f"  Total successes: {successes}/30")
    print(f"  Recovery: {'YES' if late > early + 50 else 'NO significant recovery'}")


if __name__ == '__main__':
    main()
