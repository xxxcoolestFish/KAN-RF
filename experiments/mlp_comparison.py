"""MLP vs KAN Policy: quantitative comparison of interpretability and continual learning.

Key comparisons:
1. Parameter count at iso-performance (100% Pendulum)
2. Gravity adaptation: g=10→15, test without retraining (catastrophic forgetting)
3. Interpretability: KAN 5 layers, MLP 0
"""
import torch, torch.nn as nn, numpy as np, time, sys, os, gymnasium as gym

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kanrf import ProtoKAN
from control.kan_policy_net import KANPolicy, KANPolicyTrainer
from experiments.baseline_sweep import (
    generate_pendulum_data, train_wm, generate_policy_states, evaluate_policy
)

PI_2 = np.pi / 2
S_TARGET = torch.tensor([[0.0, 1.0, 0.0]])


# ═══════════════════════════════════════
# MLP Policy (trained via same WM gradient method)
# ═══════════════════════════════════════

class MLPPolicy(nn.Module):
    def __init__(self, state_dim=3, action_dim=1, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, s):
        return torch.tanh(self.net(s))


class MLPTrainer:
    """Same training interface as KANPolicyTrainer."""
    def __init__(self, wm, policy, lr=1e-3):
        self.wm = wm; self.policy = policy; self.lr = lr
        self.opt = torch.optim.Adam(policy.parameters(), lr=lr)
        self.wm.eval()
        for p in self.wm.parameters(): p.requires_grad = False

    def train_epoch(self, s_dataset, batch_size=256):
        N = s_dataset.shape[0]; n_batches = max(1, N // batch_size)
        total_loss = 0
        for _ in range(n_batches):
            idx = torch.randint(0, N, (batch_size,))
            s_b = s_dataset[idx]
            self.policy.train(); self.opt.zero_grad()
            a = self.policy(s_b)
            s_pred = self.wm(torch.cat([s_b, a], dim=-1))
            thd = s_b[:, 2] * 8.0; Ec = 0.5 * thd.pow(2) + 10.0 * s_b[:, 1]
            thdp = s_pred[:, 2] * 8.0; Ep = 0.5 * thdp.pow(2) + 10.0 * s_pred[:, 1]
            deficit = (10.0 - Ec).detach()
            egain = (Ep - Ec) * torch.sign(deficit)
            sin = s_b[:, 1]; ws = ((1.0 + sin) / 2.0).clamp(0, 1)
            loss = (-egain.mean() +
                    (ws * (s_pred - S_TARGET.expand(batch_size, -1)).pow(2).sum(-1)).mean() +
                    0.01 * a.pow(2).mean())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 10.0)
            self.opt.step()
            total_loss += loss.item()
        return {'total': total_loss / n_batches}

    def get_action(self, s):
        self.policy.eval()
        if isinstance(s, np.ndarray):
            s = torch.tensor(s, dtype=torch.float32)
        if s.dim() == 1: s = s.unsqueeze(0)
        with torch.no_grad():
            return self.policy(s).squeeze().item()


# ═══════════════════════════════════════
# Gravity adaptation test (no retraining)
# ═══════════════════════════════════════

def evaluate_on_gravity(trainer, g=15.0, n_trials=20):
    """Test policy on different gravity WITHOUT retraining.

    Uses analytical Pendulum dynamics with specified gravity.
    """
    successes = 0; all_steps = []; all_errs = []
    for trial in range(n_trials):
        seed = 42 + trial * 100
        np.random.seed(seed)
        theta = np.random.uniform(-np.pi, np.pi)
        thd = np.random.uniform(-1.0, 1.0)

        for step in range(300):
            s_norm = np.array([np.cos(theta), np.sin(theta), thd / 8.0], dtype=np.float32)
            a_norm = trainer.get_action(s_norm)
            u = a_norm * 2.0
            # Analytical step with specified g
            thd_new = thd + (g * np.sin(theta) + u) * 0.05
            th_new = theta + thd_new * 0.05
            theta, thd = th_new, thd_new
            err = min(abs(theta - PI_2), 2 * np.pi - abs(theta - PI_2))
            if err < 0.2:
                successes += 1; all_steps.append(step + 1); all_errs.append(err); break
        else:
            all_steps.append(300)
            all_errs.append(min(abs(theta - PI_2), 2 * np.pi - abs(theta - PI_2)))
    return successes, all_steps, all_errs


# ═══════════════════════════════════════
# Main
# ═══════════════════════════════════════

def main():
    device = 'cpu'
    torch.manual_seed(42); np.random.seed(42)

    print("=" * 70)
    print("MLP vs KAN Policy Comparison")
    print("=" * 70)

    # ── Train WM (shared) ──
    print("\n[0] Training ProtoKAN WM...")
    X, Y = generate_pendulum_data(5000, seed=42)
    X, Y = X.to(device), Y.to(device)
    wm, wm_val = train_wm(X, Y)
    print(f"  val_mse={wm_val:.6f}")

    s_dataset = generate_policy_states(10000, seed=42).to(device)

    # ── 1. KAN Policy ──
    print("\n[1] Training KAN Policy [3,12,12,1]...")
    kan_pol = KANPolicy(state_dim=3, action_dim=1, hidden_dim=12, n_layers=2)
    kan_trainer = KANPolicyTrainer(wm, kan_pol, S_TARGET, lr=1e-3)
    t0 = time.time()
    for ep in range(1, 201):
        kan_trainer.train_epoch(s_dataset)
    kan_time = time.time() - t0
    kan_params = sum(p.numel() for p in kan_pol.parameters())

    # ── 2. MLP Policy (small — match KAN param count) ──
    print("\n[2] Training MLP Policy (small, ~KAN params)...")
    mlp_small = MLPPolicy(hidden=24)  # 3*24+24 + 24*24+24 + 24*1+1 = 769
    mlp_small_params = sum(p.numel() for p in mlp_small.parameters())
    mlp_small_trainer = MLPTrainer(wm, mlp_small, lr=1e-3)
    t0 = time.time()
    for ep in range(1, 201):
        mlp_small_trainer.train_epoch(s_dataset)
    mlp_small_time = time.time() - t0

    # ── 3. MLP Policy (large — typical size for Pendulum) ──
    print("\n[3] Training MLP Policy (large)...")
    mlp_large = MLPPolicy(hidden=64)  # ~4600 params
    mlp_large_params = sum(p.numel() for p in mlp_large.parameters())
    mlp_large_trainer = MLPTrainer(wm, mlp_large, lr=1e-3)
    t0 = time.time()
    for ep in range(1, 201):
        mlp_large_trainer.train_epoch(s_dataset)
    mlp_large_time = time.time() - t0

    # ── 4. Evaluate all on g=10 ──
    print("\n" + "=" * 70)
    print("STANDARD TEST (g=10)")
    print("=" * 70)

    for name, trainer in [("KAN [3,12,12,1]", kan_trainer),
                           ("MLP small [3,24,24,1]", mlp_small_trainer),
                           ("MLP large [3,64,64,1]", mlp_large_trainer)]:
        s, st, er = evaluate_policy(trainer)
        print(f"  {name:25s}: {s}/10  steps={np.mean(st):.0f}  err={np.mean(er):.3f}")

    # ── 5. Gravity adaptation (NO retraining) ──
    print("\n" + "=" * 70)
    print("GRAVITY ADAPTATION TEST (g=10→15, NO retraining)")
    print("=" * 70)

    grav_results = {}
    for g_test in [10.0, 12.0, 15.0, 18.0]:
        print(f"\n  g={g_test:.0f}:")
        for name, trainer in [("KAN", kan_trainer),
                               ("MLP-small", mlp_small_trainer),
                               ("MLP-large", mlp_large_trainer)]:
            s, st, er = evaluate_on_gravity(trainer, g=g_test, n_trials=20)
            grav_results[(name, g_test)] = (s, st, er)
            print(f"    {name:12s}: {s}/20 ({s*5}%)  steps={np.mean(st):.0f}  "
                  f"err={np.mean(er):.3f}")

    # ── 6. Summary table ──
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  {'Model':25s}  {'Params':>7s}  {'g=10':>6s}  {'g=12':>6s}  "
          f"{'g=15':>6s}  {'g=18':>6s}  {'Interp?':>8s}")
    print(f"  {'-'*70}")

    for name, params, tr in [("KAN [3,12,12,1]", kan_params, kan_trainer),
                               ("MLP small [3,24,24,1]", mlp_small_params, mlp_small_trainer),
                               ("MLP large [3,64,64,1]", mlp_large_params, mlp_large_trainer)]:
        g10 = f"{grav_results.get((name.split()[0], 10.0), (0,))[0]}0%"
        g12 = f"{grav_results.get((name.split()[0] if 'MLP' not in name else name.replace(' [', ' ').split()[0]+'-'+name.split()[1], 12.0), (0,))[0]*5}%"
        # Simplify: just print from stored results
        row = f"  {name:25s}  {params:7d}"
        for g_test in [10.0, 12.0, 15.0, 18.0]:
            key = (name.split()[0] if 'MLP' not in name else
                   'MLP-small' if 'small' in name else 'MLP-large', g_test)
            if key in grav_results:
                s, st, _ = grav_results[key]
                row += f"  {s*5:3d}%"
            else:
                row += f"  {'?':>4s}"
        row += f"  {'✓ KAN' if 'KAN' in name else '✗ MLP'}"
        print(row)

    print(f"\n  Interpretability comparison:")
    print(f"    KAN:  5 layers (causal graph, Jacobian, uncertainty, attribution, symbolic)")
    print(f"    MLP:  0 layers (black box)")
    print(f"\n  Continual learning (g=10→15 without retraining):")
    for name in ['KAN', 'MLP-small', 'MLP-large']:
        if (name, 10.0) in grav_results and (name, 15.0) in grav_results:
            s10 = grav_results[(name, 10.0)][0]
            s15 = grav_results[(name, 15.0)][0]
            drop = (s10 - s15) / max(s10, 1) * 100
            print(f"    {name:12s}: {s10*5}% → {s15*5}%  (drop={drop:.0f}%)")


if __name__ == '__main__':
    main()
