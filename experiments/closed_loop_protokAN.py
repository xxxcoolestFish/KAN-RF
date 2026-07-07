"""Full closed-loop continual learning: ProtoKAN WM + Policy.

Scenario:
  1. Train ProtoKAN WM on Pendulum g=10
  2. Train ProtoKAN Policy via frozen WM
  3. Verify Policy works on g=10
  4. SWITCH: gravity 10 → 15
  5. Policy fails (WM predicts wrong dynamics)
  6. Online-adapt WM to g=15 (few-shot, no forgetting g=10)
  7. Continue training Policy via adapted WM
  8. Policy recovers on g=15
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, time, sys, os, gymnasium as gym

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kanrf import ProtoKAN, ProtoKANLayer

PI_2 = np.pi / 2
S_TARGET = torch.tensor([[0.0, 1.0, 0.0]])


# ═══════════════════════════════════════════════════════
# Pendulum Dynamics
# ═══════════════════════════════════════════════════════

def pendulum_step_np(theta, thd, u, g=10.0, DT=0.05):
    """Single Pendulum step (numpy)."""
    thd_new = thd + (g * np.sin(theta) + u) * DT
    th_new = theta + thd_new * DT
    return th_new, thd_new


def reset_pendulum(seed):
    np.random.seed(seed)
    theta = np.random.uniform(-np.pi, np.pi)
    thd = np.random.uniform(-1.0, 1.0)
    return np.array([np.cos(theta), np.sin(theta), thd / 8.0], dtype=np.float32)


# ═══════════════════════════════════════════════════════
# Data Generation
# ═══════════════════════════════════════════════════════

def generate_wm_data(g=10.0, n=3000):
    """Generate (s,a,s') triplets for WM training."""
    xs, ys = [], []
    for _ in range(n):
        theta = np.random.uniform(-np.pi, np.pi)
        thd = np.random.uniform(-8.0, 8.0)
        a = np.random.uniform(-1.0, 1.0)
        th_new, thd_new = pendulum_step_np(theta, thd, a * 2.0, g=g)
        s = np.array([np.cos(theta), np.sin(theta), thd / 8.0], dtype=np.float32)
        sn = np.array([np.cos(th_new), np.sin(th_new), thd_new / 8.0], dtype=np.float32)
        xs.append(np.concatenate([s, [a]]))
        ys.append(sn)
    return (torch.tensor(np.array(xs), dtype=torch.float32),
            torch.tensor(np.array(ys), dtype=torch.float32))


def generate_policy_states(n=10000):
    """Generate states for policy training."""
    s_raw = []
    for _ in range(n):
        theta = np.random.uniform(-np.pi, np.pi)
        thd = np.random.uniform(-8.0, 8.0)
        s_raw.append([np.cos(theta), np.sin(theta), thd / 8.0])
    return torch.tensor(np.array(s_raw), dtype=torch.float32)


# ═══════════════════════════════════════════════════════
# WM Training
# ═══════════════════════════════════════════════════════

def train_wm(X, Y, n_proto=16, n_lbfgs=100, init_sigma=-1.5, device='cpu'):
    """Train ProtoKAN WM [4,8,3] with L-BFGS."""
    n_tr = int(len(X) * 0.85)
    X_tr, Y_tr = X[:n_tr], Y[:n_tr]
    X_val, Y_val = X[n_tr:], Y[n_tr:]
    wm = ProtoKAN([4, 8, 3], n_prototypes=n_proto).to(device)
    for layer in wm.layers:
        layer.log_sigma.data.fill_(init_sigma)
    mse_fn = nn.MSELoss()
    best_val = float('inf'); best_state = None

    def closure():
        opt.zero_grad()
        loss = mse_fn(wm(X_tr), Y_tr)
        loss.backward()
        return loss

    opt = torch.optim.LBFGS(wm.parameters(), lr=1.0, max_iter=20,
                            history_size=50, line_search_fn='strong_wolfe')
    for step in range(1, n_lbfgs + 1):
        opt.step(closure)
        with torch.no_grad():
            val = mse_fn(wm(X_val), Y_val).item()
        if val < best_val:
            best_val = val
            best_state = {k: v.clone() for k, v in wm.state_dict().items()}
    wm.load_state_dict(best_state); wm.eval()
    return wm, best_val


# ═══════════════════════════════════════════════════════
# ProtoKAN Policy
# ═══════════════════════════════════════════════════════

class ProtoPolicy(nn.Module):
    def __init__(self, n_proto=16, init_sigma=-1.5):
        super().__init__()
        self.l1 = ProtoKANLayer(3, 8, n_prototypes=n_proto)
        self.l2 = ProtoKANLayer(8, 8, n_prototypes=n_proto)
        self.out = nn.Linear(8, 1)
        for layer in [self.l1, self.l2]:
            layer.log_sigma.data.fill_(init_sigma)

    def forward(self, s):
        return torch.tanh(self.out(self.l2(self.l1(s))))


def train_policy(wm, s_dataset, policy=None, epochs=300, lr=1e-3, device='cpu'):
    """Train ProtoKAN Policy via frozen WM."""
    if policy is None:
        policy = ProtoPolicy().to(device)
    wm.eval()
    for p in wm.parameters():
        p.requires_grad = False
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    s_tgt = torch.tensor([[0.0, 1.0, 0.0]], device=device)
    N = len(s_dataset)

    for ep in range(1, epochs + 1):
        total_loss = 0.0
        for _ in range(max(1, N // 256)):
            idx = torch.randint(0, N, (256,), device=device)
            s_b = s_dataset[idx]
            policy.train(); opt.zero_grad()
            a = policy(s_b)
            s_pred = wm(torch.cat([s_b, a], dim=-1))
            # Energy-guided loss
            thd = s_b[:, 2] * 8.0; Ec = 0.5 * thd.pow(2) + 10.0 * s_b[:, 1]
            thdp = s_pred[:, 2] * 8.0; Ep = 0.5 * thdp.pow(2) + 10.0 * s_pred[:, 1]
            deficit = (10.0 - Ec).detach()
            egain = (Ep - Ec) * torch.sign(deficit)
            sin = s_b[:, 1]; ws = ((1.0 + sin) / 2.0).clamp(0, 1)
            loss = (-egain.mean() +
                    (ws * (s_pred - s_tgt.expand(256, -1)).pow(2).sum(-1)).mean() +
                    0.01 * a.pow(2).mean())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
            opt.step()
            total_loss += loss.item()
        if ep % 60 == 0 or ep == 1:
            print(f"    Policy epoch {ep:3d}  loss={total_loss/(N//256):.4f}")


def evaluate_policy(policy, g=10.0, n_trials=20, max_steps=300, label=''):
    """Evaluate policy on Pendulum with given gravity."""
    successes = 0; all_steps = []; all_errs = []
    for trial in range(n_trials):
        seed = 42 + trial * 100
        s = reset_pendulum(seed)
        ok = False
        for step in range(max_steps):
            theta = np.arctan2(s[1], s[0]); thd = s[2] * 8.0
            with torch.no_grad():
                a_norm = policy(torch.tensor([s], dtype=torch.float32)).item()
            u = a_norm * 2.0
            th_new, thd_new = pendulum_step_np(theta, thd, u, g=g)
            s = np.array([np.cos(th_new), np.sin(th_new), thd_new / 8.0], dtype=np.float32)
            err = min(abs(th_new - PI_2), 2 * np.pi - abs(th_new - PI_2))
            if err < 0.2:
                successes += 1; all_steps.append(step + 1); all_errs.append(err); ok = True; break
        if not ok:
            all_steps.append(max_steps); all_errs.append(err)
        if trial < 5 or not ok:
            print(f"    [{label}] T{trial+1:2d}  {'✓' if ok else '✗'}  steps={all_steps[-1]:3d}  err={all_errs[-1]:.3f}")
    return successes, all_steps, all_errs


# ═══════════════════════════════════════════════════════
# Online Adaptation
# ═══════════════════════════════════════════════════════

def adapt_wm(wm, X_new, Y_new, n_steps=30, lr=1e-3):
    """Online-adapt WM to new physics."""
    for p in wm.parameters():
        p.requires_grad = True
    wm.train()
    opt = torch.optim.Adam(wm.parameters(), lr=lr)
    mse_fn = nn.MSELoss()
    for step in range(1, n_steps + 1):
        idx = np.random.choice(len(X_new), min(64, len(X_new)), replace=False)
        opt.zero_grad()
        loss = mse_fn(wm(X_new[idx]), Y_new[idx])
        loss.backward()
        opt.step()
    wm.eval()


def adapt_policy(wm, policy, s_dataset, n_epochs=100, lr=5e-4):
    """Continue training policy with adapted WM."""
    wm.eval()
    for p in wm.parameters():
        p.requires_grad = False
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    s_tgt = torch.tensor([[0.0, 1.0, 0.0]])
    N = len(s_dataset)
    for ep in range(1, n_epochs + 1):
        total_loss = 0.0
        for _ in range(max(1, N // 256)):
            idx = torch.randint(0, N, (256,))
            s_b = s_dataset[idx]
            policy.train(); opt.zero_grad()
            a = policy(s_b)
            s_pred = wm(torch.cat([s_b, a], dim=-1))
            thd = s_b[:, 2] * 8.0; Ec = 0.5 * thd.pow(2) + 10.0 * s_b[:, 1]
            thdp = s_pred[:, 2] * 8.0; Ep = 0.5 * thdp.pow(2) + 10.0 * s_pred[:, 1]
            deficit = (10.0 - Ec).detach()
            egain = (Ep - Ec) * torch.sign(deficit)
            sin = s_b[:, 1]; ws = ((1.0 + sin) / 2.0).clamp(0, 1)
            loss = (-egain.mean() +
                    (ws * (s_pred - s_tgt.expand(256, -1)).pow(2).sum(-1)).mean() +
                    0.01 * a.pow(2).mean())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
            opt.step()
            total_loss += loss.item()
        if ep % 40 == 0 or ep == 1:
            print(f"    Policy adapt epoch {ep:3d}  loss={total_loss/(N//256):.4f}")


# ═══════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(42); np.random.seed(42)

    print("=" * 70)
    print("FULL CLOSED-LOOP: ProtoKAN WM + Policy Continual Learning")
    print("Gravity: 10 → 15")
    print("=" * 70)

    # ── Phase 1: Initial Training (g=10) ──
    print("\n" + "=" * 70)
    print("PHASE 1: Train on g=10")
    print("=" * 70)

    print("\n[1a] Generating WM data (g=10)...")
    X_old, Y_old = generate_wm_data(g=10.0, n=3000)
    print(f"  {X_old.shape[0]} samples")

    print("\n[1b] Training ProtoKAN WM (g=10)...")
    t0 = time.time()
    wm, wm_val = train_wm(X_old, Y_old, init_sigma=-1.5, device=device)
    sigma_vals = [torch.exp(l.log_sigma).item() for l in wm.layers]
    print(f"  val_mse={wm_val:.6f}  sigma={[f'{s:.2f}' for s in sigma_vals]}  "
          f"time={time.time()-t0:.0f}s")

    print("\n[1c] Training ProtoKAN Policy (via WM, g=10)...")
    s_pol = generate_policy_states(10000).to(device)
    policy = ProtoPolicy().to(device)
    train_policy(wm, s_pol, policy=policy, epochs=200, device=device)

    print("\n[1d] Evaluating on g=10...")
    s10, st10, er10 = evaluate_policy(policy, g=10.0, n_trials=20, label='g=10')
    print(f"  g=10 result: {s10}/20 ({s10*5}%)  mean_steps={np.mean(st10):.0f}  "
          f"mean_err={np.mean(er10):.3f}")

    # ── Phase 2: Gravity Switch ──
    print("\n" + "=" * 70)
    print("PHASE 2: Gravity SWITCH (10 → 15)")
    print("=" * 70)

    print("\n[2a] Policy on g=15 (BEFORE adaptation)...")
    s15_before, st15_before, er15_before = evaluate_policy(
        policy, g=15.0, n_trials=20, label='g=15_before')
    print(f"  g=15 BEFORE: {s15_before}/20 ({s15_before*5}%)  "
          f"mean_steps={np.mean(st15_before):.0f}  mean_err={np.mean(er15_before):.3f}")

    # ── Phase 3: WM Adaptation ──
    print("\n[3] Adapting WM to g=15 (online, 30 steps)...")
    X_new, Y_new = generate_wm_data(g=15.0, n=500)
    X_new, Y_new = X_new.to(device), Y_new.to(device)
    adapt_wm(wm, X_new, Y_new, n_steps=30, lr=1e-3)
    mse_fn = nn.MSELoss()
    with torch.no_grad():
        wm_new_err = mse_fn(wm(X_new), Y_new).item()
        wm_old_err = mse_fn(wm(X_old[:500].to(device)), Y_old[:500].to(device)).item()
    print(f"  After WM adapt: new_mse={wm_new_err:.6f}  old_mse={wm_old_err:.6f}")

    # ── Phase 4: Policy Adaptation ──
    print("\n[4] Adapting Policy via updated WM...")
    adapt_policy(wm, policy, s_pol, n_epochs=100, lr=5e-4)

    print("\n[5] Evaluating Policy on g=15 (AFTER adaptation)...")
    s15_after, st15_after, er15_after = evaluate_policy(
        policy, g=15.0, n_trials=20, label='g=15_after')
    print(f"  g=15 AFTER: {s15_after}/20 ({s15_after*5}%)  "
          f"mean_steps={np.mean(st15_after):.0f}  mean_err={np.mean(er15_after):.3f}")

    # ── Phase 5: Verify old knowledge preserved ──
    print("\n[6] Verifying g=10 knowledge preserved...")
    s10_after, st10_after, er10_after = evaluate_policy(
        policy, g=10.0, n_trials=20, label='g=10_after')
    print(f"  g=10 AFTER: {s10_after}/20 ({s10_after*5}%)  "
          f"mean_steps={np.mean(st10_after):.0f}  mean_err={np.mean(er10_after):.3f}")

    # ── Report ──
    print("\n" + "=" * 70)
    print("CLOSED-LOOP RESULT")
    print("=" * 70)
    print(f"  {'Phase':25s}  {'Success':>8s}  {'Steps':>6s}  {'Error':>6s}")
    print(f"  {'g=10 (initial)':25s}  {s10:4d}/20       {np.mean(st10):5.0f}    {np.mean(er10):5.3f}")
    print(f"  {'g=15 (before adapt)':25s}  {s15_before:4d}/20       {np.mean(st15_before):5.0f}    {np.mean(er15_before):5.3f}")
    print(f"  {'g=15 (AFTER adapt)':25s}  {s15_after:4d}/20       {np.mean(st15_after):5.0f}    {np.mean(er15_after):5.3f}")
    print(f"  {'g=10 (AFTER, verify)':25s}  {s10_after:4d}/20       {np.mean(st10_after):5.0f}    {np.mean(er10_after):5.3f}")


if __name__ == '__main__':
    main()
