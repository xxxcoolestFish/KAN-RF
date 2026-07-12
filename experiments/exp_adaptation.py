
"""Continual Learning Adaptation Experiment.

Tests: pre-train → parameter change → WM adapt → policy retrain → recover.
"""

import sys, os, time, torch, torch.nn as nn, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.baseline_sweep import generate_pendulum_data, train_wm, generate_policy_states, evaluate_policy
from experiments.exp_cost_discovery import energy_loss, CustomKANPolicyTrainer, energy, compute_energy_action
from control.kan_policy_net import KANPolicy

DEVICE = 'cpu'
S_TARGET = torch.tensor([[0.0, 1.0, 0.0]])
G = 10.0; PI_2 = np.pi / 2


def step(s_norm, a_norm, g=G, damping=0.0):
    """Pendulum step matching Gym exactly + optional damping."""
    th = torch.atan2(s_norm[:, 1:2], s_norm[:, 0:1])
    thd = s_norm[:, 2:3] * 8.0
    u = torch.clamp(a_norm * 2.0, -2.0, 2.0)
    newthd = thd + (3 * g / 2.0 * torch.sin(th) + 3.0 * u - damping * thd) * 0.05
    newthd = torch.clamp(newthd, -8.0, 8.0)
    newth = th + newthd * 0.05
    newth = torch.atan2(torch.sin(newth), torch.cos(newth))
    return torch.cat([torch.cos(newth), torch.sin(newth), newthd / 8.0], dim=-1)


def main():
    print('=' * 65)
    print('  Continual Learning Adaptation Experiment')
    print('=' * 65)

    # ── 1. Pre-train WM ──
    print('\n1. Pre-training ProtoKAN WM on g=10...')
    X, Y = generate_pendulum_data(3000, seed=42)
    wm, _ = train_wm(X.to(DEVICE), Y.to(DEVICE))
    wm.eval()

    # ── 2. Discover G from WM ──
    print('2. Discovering energy cost from WM...')
    s_pol = generate_policy_states(10000, seed=42).to(DEVICE)

    g_ests = []
    with torch.no_grad():
        s_cur = s_pol[:1000].clone()
        for t in range(15):
            sn = wm(torch.cat([s_cur, torch.zeros(s_cur.shape[0], 1, device=DEVICE)], dim=-1))[:, :3]
            n1 = s_cur[:, 2] * 8.0; n2 = sn[:, 2] * 8.0
            s1 = s_cur[:, 1]; s2 = sn[:, 1]
            num = n2.pow(2) - n1.pow(2); den = 2.0 * (s1 - s2)
            mask = den.abs() > 1e-4
            g = num[mask] / den[mask]; g_ests.append(g.cpu())
            s_cur = sn
    g_all = torch.cat(g_ests); g_clean = g_all[(g_all > 5) & (g_all < 20)]
    G_hat = g_clean.mean().item()
    print(f'  G estimated = {G_hat:.2f} (true = {G})')

    def disc_cost(sp, sb, _):
        B = sp.shape[0]
        Ep = 0.5 * (sp[:, 2] * 8).pow(2) + G_hat * sp[:, 1]
        Ec = 0.5 * (sb[:, 2] * 8).pow(2) + G_hat * sb[:, 1]
        eg = (Ep - Ec) * torch.sign(G_hat - Ec); el = -eg.mean()
        ws = ((1.0 + sb[:, 1]) / 2.0).clamp(0, 1)
        ms = (ws * (sp - S_TARGET.expand(B, -1)).pow(2).sum(dim=-1)).mean()
        return el + 0.5 * ms, {'el': el.item(), 'ms': ms.item()}

    # ── 3. Pre-train KAN Policy ──
    print('3. Training KAN Policy...')
    torch.manual_seed(42)
    policy = KANPolicy(state_dim=3, action_dim=1, hidden_dim=12, n_layers=2)
    trainer = CustomKANPolicyTrainer(
        wm, policy, S_TARGET, cost_fn=disc_cost, cost_name='disc',
        lr=1e-3, device=DEVICE)
    for ep in range(1, 201):
        trainer.train_epoch(s_pol, batch_size=256)
    print('  Policy trained')

    # ── 4. Baseline on g=10 ──
    print('\n4. Baseline evaluation:')
    bl, _, _ = evaluate_policy(trainer, 10, seed=42)
    print(f'  g=10: {bl}/10')

    # ── 5. Test gravity changes ──
    print('\n5. Testing gravity generalization:')
    for g_test in [3.0, 5.0, 15.0, 20.0]:
        succ = 0
        for tr in range(10):
            np.random.seed(42 + tr * 100)
            theta = np.random.uniform(-np.pi, np.pi)
            thd = np.random.uniform(-1.0, 1.0)
            s = torch.tensor([[np.cos(theta), np.sin(theta), thd / 8.0]], dtype=torch.float32)
            for _ in range(300):
                a = trainer.get_action(s[0].numpy())
                s2 = step(s, torch.tensor([[a]]), g=g_test)
                err = min(abs(np.arctan2(s2[0, 1].item(), s2[0, 0].item()) - PI_2),
                          2 * np.pi - abs(np.arctan2(s2[0, 1].item(), s2[0, 0].item()) - PI_2))
                if err < 0.2:
                    succ += 1; break
                s = s2
        print(f'  g={g_test:5.1f}: {succ}/10')

    # ── 6. Damping adaptation ──
    print('\n6. Adaptation to damping:')
    damping = 2.0

    # Drop
    succ = 0
    for tr in range(10):
        np.random.seed(42 + tr * 100)
        theta = np.random.uniform(-np.pi, np.pi)
        thd = np.random.uniform(-1.0, 1.0)
        s = torch.tensor([[np.cos(theta), np.sin(theta), thd / 8.0]], dtype=torch.float32)
        for _ in range(300):
            a = trainer.get_action(s[0].numpy())
            s2 = step(s, torch.tensor([[a]]), damping=damping)
            err = min(abs(np.arctan2(s2[0, 1].item(), s2[0, 0].item()) - PI_2),
                      2 * np.pi - abs(np.arctan2(s2[0, 1].item(), s2[0, 0].item()) - PI_2))
            if err < 0.2:
                succ += 1; break
            s = s2
    print(f'  Pre-trained policy with damping={damping}: {succ}/10 (drop)')

    # Generate damped data
    print('  Generating damped transitions...')
    n_adapt = 2000
    X_new, Y_new = [], []
    for _ in range(n_adapt):
        theta = np.random.uniform(-np.pi, np.pi)
        thd = np.random.uniform(-8.0, 8.0)
        a_n = np.random.uniform(-1.0, 1.0)
        s = torch.tensor([[np.cos(theta), np.sin(theta), thd / 8.0]], dtype=torch.float32).to(DEVICE)
        sn = step(s.cpu(), torch.tensor([[a_n]]), damping=damping)
        X_new.append(torch.cat([s.cpu(), torch.tensor([[a_n]])], dim=-1))
        Y_new.append(sn)
    X_new = torch.cat(X_new).to(DEVICE)
    Y_new = torch.cat(Y_new).to(DEVICE)

    # WM fine-tuning
    print('  Fine-tuning ProtoKAN WM on damped data...')
    wm.train()
    for p in wm.parameters(): p.requires_grad = True
    opt = torch.optim.Adam(wm.parameters(), lr=1e-3)
    for ep in range(1, 51):
        idx = torch.randint(0, n_adapt, (256,))
        loss = nn.MSELoss()(wm(X_new[idx]), Y_new[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    wm.eval()

    # Policy re-training
    print('  Re-training policy on adapted WM...')
    trainer2 = CustomKANPolicyTrainer(
        wm, policy, S_TARGET, cost_fn=disc_cost, cost_name='disc',
        lr=1e-3, device=DEVICE)
    for ep in range(1, 101):
        trainer2.train_epoch(s_pol, batch_size=256)

    # Recovery
    succ = 0
    for tr in range(10):
        np.random.seed(42 + tr * 100)
        theta = np.random.uniform(-np.pi, np.pi)
        thd = np.random.uniform(-1.0, 1.0)
        s = torch.tensor([[np.cos(theta), np.sin(theta), thd / 8.0]], dtype=torch.float32)
        for _ in range(300):
            a = trainer2.get_action(s[0].numpy())
            s2 = step(s, torch.tensor([[a]]), damping=damping)
            err = min(abs(np.arctan2(s2[0, 1].item(), s2[0, 0].item()) - PI_2),
                      2 * np.pi - abs(np.arctan2(s2[0, 1].item(), s2[0, 0].item()) - PI_2))
            if err < 0.2:
                succ += 1; break
            s = s2
    print(f'  After adaptation: {succ}/10 (recovery)')

    # Forgetting test
    bl2, _, _ = evaluate_policy(trainer2, 10, seed=42)
    print(f'  Back to g=10 (forgetting): {bl2}/10')

    # ── MLP comparison ──
    print('\n7. MLP WM comparison:')
    mlp = nn.Sequential(nn.Linear(4, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 3))
    opt_m = torch.optim.Adam(mlp.parameters(), lr=1e-3)
    for ep in range(1, 301):
        idx = torch.randint(0, 3000, (256,))
        loss = nn.MSELoss()(mlp(X[idx].to(DEVICE)), Y[idx].to(DEVICE))
        opt_m.zero_grad(); loss.backward(); opt_m.step()
    mlp.eval()
    with torch.no_grad():
        val_mlp = nn.MSELoss()(mlp(X[2550:].to(DEVICE)), Y[2550:].to(DEVICE)).item()
    print(f'  MLP WM val: {val_mlp:.6f}')

    # MLP Policy
    torch.manual_seed(42)
    mlp_pol = KANPolicy()
    mlp_tr = CustomKANPolicyTrainer(mlp, mlp_pol, S_TARGET, cost_fn=disc_cost,
                                     cost_name='disc', lr=1e-3, device=DEVICE)
    for ep in range(1, 201):
        mlp_tr.train_epoch(s_pol, batch_size=256)
    bl_mlp, _, _ = evaluate_policy(mlp_tr, 10, seed=42)
    print(f'  MLP Policy on g=10: {bl_mlp}/10')

    # MLP fine-tune on damped
    mlp.train()
    for p in mlp.parameters(): p.requires_grad = True
    opt_m2 = torch.optim.Adam(mlp.parameters(), lr=1e-3)
    for ep in range(1, 51):
        idx = torch.randint(0, n_adapt, (256,))
        loss = nn.MSELoss()(mlp(X_new[idx]), Y_new[idx])
        opt_m2.zero_grad(); loss.backward(); opt_m2.step()
    mlp.eval()

    mlp_tr2 = CustomKANPolicyTrainer(mlp, mlp_pol, S_TARGET, cost_fn=disc_cost,
                                      cost_name='disc', lr=1e-3, device=DEVICE)
    for ep in range(1, 101):
        mlp_tr2.train_epoch(s_pol, batch_size=256)

    rec_mlp, _, _ = evaluate_policy(mlp_tr2, 10, seed=42)
    forget_mlp, _, _ = evaluate_policy(mlp_tr2, 10, seed=42)
    print(f'  MLP adapted: {rec_mlp}/10')
    print(f'  MLP forgetting: {forget_mlp}/10')

    print('\n' + '=' * 65)
    print('  Done.')
    print('=' * 65)


if __name__ == '__main__':
    main()
