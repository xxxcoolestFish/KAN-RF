"""Jacobian-weighted Policy training: auto-extract controllability from WM.

Key idea: use WM's Jacobian ∂s/∂a to dynamically weight loss dimensions.
Dimensions that the action can directly influence get high weight.
Dimensions that can only change indirectly (through integration) get low weight.

This replaces the hand-crafted Strategy layer with automatic controllability
extraction from the WM — works for ANY system without physics knowledge.
"""
import torch, torch.nn as nn, numpy as np, time, sys, os, gymnasium as gym

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kanrf import ProtoKAN
from control.kan_policy_net import KANPolicy

PI_2 = np.pi / 2
S_TARGET = torch.tensor([[0.0, 1.0, 0.0]])


# ═══════════════════════════════════════
# Jacobian extractor
# ═══════════════════════════════════════

def compute_controllability_weights(wm, s_dataset, n_samples=200, device='cpu'):
    """Compute per-dimension controllability weights from WM Jacobian.

    For each state dimension i, compute avg ||∂s'_i/∂a|| over sample states.
    Returns normalized weights (state_dim,).
    """
    wm.eval()
    N = min(n_samples, len(s_dataset))
    idx = torch.randperm(len(s_dataset))[:N]
    s_batch = s_dataset[idx].to(device)

    jac_norms = torch.zeros(s_batch.shape[1], device=device)

    for i in range(s_batch.shape[1]):  # for each state dimension
        a = torch.zeros(N, 1, device=device, requires_grad=True)
        x = torch.cat([s_batch, a], dim=-1)
        s_pred = wm(x)
        grads = torch.autograd.grad(s_pred[:, i].sum(), a, retain_graph=True)[0]
        jac_norms[i] = grads.abs().mean()

    # Normalize to sum to 1
    weights = jac_norms / (jac_norms.sum() + 1e-8)
    return weights


# ═══════════════════════════════════════
# Jacobian-weighted Policy Trainer
# ═══════════════════════════════════════

class JacobianWeightedTrainer:
    """Train Policy with controllability-weighted loss from WM Jacobian."""

    def __init__(self, wm, policy, s_dataset, lr=1e-3, n_jac_samples=200, device='cpu'):
        self.wm = wm
        self.policy = policy.to(device)
        self.device = device

        # Pre-compute controllability weights once
        print("  Computing controllability weights from WM Jacobian...")
        self.loss_weights = compute_controllability_weights(
            wm, s_dataset, n_samples=n_jac_samples, device=device)
        print(f"  Loss weights: {self.loss_weights.cpu().numpy()}")

        wm.eval()
        for p in wm.parameters():
            p.requires_grad = False

        self.opt = torch.optim.Adam(policy.parameters(), lr=lr)
        self.loss_history = []
        self.s_target = S_TARGET.to(device)

    def train_epoch(self, s_dataset, batch_size=256):
        N = s_dataset.shape[0]; n_batches = max(1, N // batch_size)
        total_loss = 0
        for _ in range(n_batches):
            idx = torch.randint(0, N, (batch_size,), device=self.device)
            s_b = s_dataset[idx]
            self.policy.train(); self.opt.zero_grad()
            a = self.policy(s_b)
            s_pred = self.wm(torch.cat([s_b, a], dim=-1))

            # Weighted loss: dimensions with high Jacobian get high weight
            err = (s_pred - self.s_target.expand(batch_size, -1)) ** 2
            loss = (err * self.loss_weights.unsqueeze(0)).sum(dim=-1).mean()

            # Plus small control penalty
            loss = loss + 0.01 * a.pow(2).mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 10.0)
            self.opt.step()
            total_loss += loss.item()

        avg_loss = total_loss / n_batches
        self.loss_history.append(avg_loss)
        return {'total': avg_loss}

    def get_action(self, s):
        self.policy.eval()
        if isinstance(s, np.ndarray):
            s = torch.tensor(s, dtype=torch.float32, device=self.device)
        if s.dim() == 1: s = s.unsqueeze(0)
        with torch.no_grad():
            return self.policy(s).squeeze().cpu().item()


# ═══════════════════════════════════════
# CartPole version
# ═══════════════════════════════════════

class CartPoleJacobianWeightedTrainer:
    """Jacobian-weighted trainer for CartPole (4D state)."""

    def __init__(self, wm, policy, s_dataset, lr=1e-3, n_jac_samples=200, device='cpu'):
        self.wm = wm; self.policy = policy.to(device); self.device = device
        print("  Computing controllability weights...")
        self.loss_weights = compute_controllability_weights(
            wm, s_dataset, n_samples=n_jac_samples, device=device)
        print(f"  Loss weights: {self.loss_weights.cpu().numpy()}")

        wm.eval()
        for p in wm.parameters(): p.requires_grad = False
        self.opt = torch.optim.Adam(policy.parameters(), lr=lr)
        self.s_target = torch.zeros(1, 4, device=device)

    def train_epoch(self, s_dataset, batch_size=256):
        N = s_dataset.shape[0]; n_batches = max(1, N // batch_size)
        total_loss = 0
        for _ in range(n_batches):
            idx = torch.randint(0, N, (batch_size,), device=self.device)
            s_b = s_dataset[idx]
            self.policy.train(); self.opt.zero_grad()
            a = self.policy(s_b)
            s_pred = self.wm(torch.cat([s_b, a], dim=-1))
            err = (s_pred - self.s_target.expand(batch_size, -1)) ** 2
            loss = (err * self.loss_weights.unsqueeze(0)).sum(dim=-1).mean()
            loss = loss + 0.01 * a.pow(2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 10.0)
            self.opt.step()
            total_loss += loss.item()
        return {'total': total_loss / n_batches}

    def get_action(self, s):
        self.policy.eval()
        if isinstance(s, np.ndarray):
            s = torch.tensor(s, dtype=torch.float32, device=self.device)
        if s.dim() == 1: s = s.unsqueeze(0)
        with torch.no_grad():
            return self.policy(s).squeeze().cpu().item()


# ═══════════════════════════════════════
# Test: Pendulum
# ═══════════════════════════════════════

def test_pendulum():
    from experiments.baseline_sweep import (
        generate_pendulum_data, train_wm, generate_policy_states, evaluate_policy
    )
    device = 'cpu'
    torch.manual_seed(42); np.random.seed(42)

    print("=" * 70)
    print("Jacobian-Weighted Policy: Pendulum")
    print("=" * 70)

    X, Y = generate_pendulum_data(5000, seed=42)
    X, Y = X.to(device), Y.to(device)
    wm, wm_val = train_wm(X, Y)
    print(f"WM val_mse={wm_val:.6f}")

    # Compare: standard trainer vs jacobian-weighted trainer
    # Standard
    print("\n[Baseline] Standard KANPolicyTrainer:")
    torch.manual_seed(42); np.random.seed(42)
    s_pol = generate_policy_states(10000, seed=42).to(device)
    policy_std = KANPolicy(state_dim=3, action_dim=1, hidden_dim=12, n_layers=2)
    from control.kan_policy_net import KANPolicyTrainer
    trainer_std = KANPolicyTrainer(wm, policy_std, S_TARGET, lr=1e-3)
    for ep in range(1, 201):
        trainer_std.train_epoch(s_pol)
        if ep % 60 == 0: print(f"  Epoch {ep:3d}  loss={trainer_std.loss_history[-1]['total']:.4f}")
    s_std, st_std, er_std = evaluate_policy(trainer_std)

    # Jacobian-weighted
    print("\n[Jacobian-Weighted]")
    torch.manual_seed(42); np.random.seed(42)
    policy_jw = KANPolicy(state_dim=3, action_dim=1, hidden_dim=12, n_layers=2)
    trainer_jw = JacobianWeightedTrainer(wm, policy_jw, s_pol, device=device)
    for ep in range(1, 201):
        ld = trainer_jw.train_epoch(s_pol)
        if ep % 60 == 0: print(f"  Epoch {ep:3d}  loss={ld['total']:.4f}")
    s_jw, st_jw, er_jw = evaluate_policy(trainer_jw)

    print(f"\n  Standard:       {s_std}/10  steps={np.mean(st_std):.0f}  err={np.mean(er_std):.3f}")
    print(f"  Jacobian-Weight: {s_jw}/10  steps={np.mean(st_jw):.0f}  err={np.mean(er_jw):.3f}")
    return s_std, s_jw


# ═══════════════════════════════════════
# Test: CartPole
# ═══════════════════════════════════════

def test_cartpole():
    from experiments.cartpole_continual import (
        generate_wm_data, train_wm, generate_policy_states,
        X_S, XD_S, TH_S, THD_S, step_cartpole
    )
    device = 'cpu'
    torch.manual_seed(42); np.random.seed(42)

    print("\n" + "=" * 70)
    print("Jacobian-Weighted Policy: CartPole")
    print("=" * 70)

    X, Y = generate_wm_data(g=9.8, n=5000, device=device)
    wm, wm_val = train_wm(X, Y, 'protokan', 80, device)
    print(f"WM val_mse={wm_val:.6f}")

    s_pol = generate_policy_states(15000, device)

    # Standard
    print("\n[Baseline] Standard CartPole Trainer:")
    torch.manual_seed(42); np.random.seed(42)
    policy_std = KANPolicy(state_dim=4, action_dim=1, hidden_dim=12, n_layers=2)
    from experiments.cartpole_continual import CartPoleTrainer
    trainer_std = CartPoleTrainer(wm, policy_std)
    for ep in range(1, 201):
        ld = trainer_std.train_epoch(s_pol)
        if ep % 60 == 0: print(f"  Epoch {ep:3d}  loss={ld['total']:.4f}")

    # Jacobian-weighted
    print("\n[Jacobian-Weighted]")
    torch.manual_seed(42); np.random.seed(42)
    policy_jw = KANPolicy(state_dim=4, action_dim=1, hidden_dim=12, n_layers=2)
    trainer_jw = CartPoleJacobianWeightedTrainer(wm, policy_jw, s_pol, device=device)
    for ep in range(1, 201):
        ld = trainer_jw.train_epoch(s_pol)
        if ep % 60 == 0: print(f"  Epoch {ep:3d}  loss={ld['total']:.4f}")

    # Evaluate both
    for name, trainer in [("Standard", trainer_std), ("Jacobian-Weighted", trainer_jw)]:
        succ = 0; steps = []
        for trial in range(20):
            seed = 42 + trial * 100
            np.random.seed(seed)
            th = np.random.uniform(-0.05, 0.05)
            s_raw = torch.tensor([[0.0, 0.0, th, 0.0]], dtype=torch.float32)
            for step in range(500):
                s_n = s_raw.clone()
                s_n[:, 0] /= X_S; s_n[:, 1] /= XD_S
                s_n[:, 2] /= TH_S; s_n[:, 3] /= THD_S
                a_norm = trainer.get_action(s_n[0].numpy())
                s_raw = step_cartpole(s_raw, torch.tensor([a_norm]))
                theta, x = s_raw[0, 2].item(), s_raw[0, 0].item()
                if abs(theta) > 0.21 or abs(x) > 2.4: break
            steps.append(step + 1)
            if step + 1 >= 500: succ += 1
        print(f"  {name}: {succ}/20 ({succ*5}%)  mean_steps={np.mean(steps):.0f}")

    return succ  # return JW result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='both', choices=['pendulum', 'cartpole', 'both'])
    args = parser.parse_args()

    if args.env in ['pendulum', 'both']:
        test_pendulum()
    if args.env in ['cartpole', 'both']:
        test_cartpole()
