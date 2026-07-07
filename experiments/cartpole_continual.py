"""CartPole: Full ProtoKAN framework — Policy + Continual Learning.

Tests:
1. ProtoKAN WM + KAN Policy training on standard CartPole
2. Continual learning: gravity change g=9.8→15.0
3. WM adaptation + Policy recovery
"""
import torch, torch.nn as nn, numpy as np, time, sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kanrf import ProtoKAN, KAN
from control.kan_policy_net import KANPolicy, KANPolicyTrainer

# CartPole dynamics
G_DEF = 9.8; MC = 1.0; MP = 0.1; L = 0.5; DT = 0.02
TOTAL_MASS = MC + MP; PML = MP * L
X_S = 2.5; XD_S = 3.0; TH_S = 0.3; THD_S = 3.0; FM = 10.0
S_TARGET = torch.tensor([[0.0, 0.0, 0.0, 0.0]])  # centered, upright, stationary


def step_cartpole(state, a_norm, g=G_DEF):
    """CartPole step. state: (B,4) raw [x, xd, th, thd]."""
    x, xd, th, thd = state[:, 0], state[:, 1], state[:, 2], state[:, 3]
    force = a_norm * FM
    costh, sinth = torch.cos(th), torch.sin(th)
    temp = (force + PML * thd ** 2 * sinth) / TOTAL_MASS
    denom = 0.5 * (4.0 / 3.0 - MP * costh ** 2 / TOTAL_MASS)
    th_acc = (g * sinth - costh * temp) / (denom + 1e-8)
    x_acc = temp - PML * th_acc * costh / TOTAL_MASS
    xd_n = xd + x_acc * DT; thd_n = thd + th_acc * DT
    x_n = x + xd_n * DT; th_n = th + thd_n * DT
    return torch.stack([x_n, xd_n, th_n, thd_n], dim=-1)


def generate_wm_data(g=G_DEF, n=5000, device='cpu'):
    """Generate (s,a,s') for CartPole WM."""
    xs, ys = [], []
    for _ in range(n):
        x = np.random.uniform(-2.4, 2.4)
        xd = np.random.uniform(-3.0, 3.0)
        th = np.random.uniform(-0.3, 0.3)
        thd = np.random.uniform(-3.0, 3.0)
        a = np.random.uniform(-1.0, 1.0)
        s_raw = torch.tensor([[x, xd, th, thd]], dtype=torch.float32)
        s_next = step_cartpole(s_raw, torch.tensor([a]), g=g)
        s_norm = s_raw.clone(); sn_norm = s_next.clone()
        s_norm[:, 0] /= X_S; s_norm[:, 1] /= XD_S
        s_norm[:, 2] /= TH_S; s_norm[:, 3] /= THD_S
        sn_norm[:, 0] /= X_S; sn_norm[:, 1] /= XD_S
        sn_norm[:, 2] /= TH_S; sn_norm[:, 3] /= THD_S
        xs.append(torch.cat([s_norm, torch.tensor([[a]])], dim=-1))
        ys.append(sn_norm)
    return (torch.cat(xs, dim=0).float().to(device),
            torch.cat(ys, dim=0).float().to(device))


def generate_policy_states(n=15000, device='cpu'):
    """States for policy training."""
    n_near = n // 2; n_wide = n - n_near
    s_near = np.stack([
        np.random.uniform(-1.0, 1.0, n_near) / X_S,
        np.random.uniform(-1.5, 1.5, n_near) / XD_S,
        np.random.uniform(-0.15, 0.15, n_near) / TH_S,
        np.random.uniform(-1.5, 1.5, n_near) / THD_S,
    ], axis=1)
    s_wide = np.stack([
        np.random.uniform(-2.0, 2.0, n_wide) / X_S,
        np.random.uniform(-2.5, 2.5, n_wide) / XD_S,
        np.random.uniform(-0.25, 0.25, n_wide) / TH_S,
        np.random.uniform(-2.5, 2.5, n_wide) / THD_S,
    ], axis=1)
    s_all = np.vstack([s_near, s_wide]); np.random.shuffle(s_all)
    return torch.tensor(s_all[:n], dtype=torch.float32).to(device)


def train_wm(X, Y, wm_type='protokan', n_lbfgs=100, device='cpu'):
    """Train WM [5,16,4]."""
    n_tr = int(len(X) * 0.85)
    X_tr, Y_tr = X[:n_tr], Y[:n_tr]
    X_val, Y_val = X[n_tr:], Y[n_tr:]
    if wm_type == 'protokan':
        wm = ProtoKAN([5, 16, 4], n_prototypes=16).to(device)
        for layer in wm.layers: layer.log_sigma.data.fill_(-1.5)
    else:
        wm = KAN([5, 16, 4], grid_size=5, spline_order=3).to(device)

    mse_fn = nn.MSELoss(); best_val = float('inf'); best_state = None
    def closure():
        opt.zero_grad()
        loss = mse_fn(wm(X_tr), Y_tr)
        loss.backward()
        return loss
    opt = torch.optim.LBFGS(wm.parameters(), lr=1.0, max_iter=20,
                            history_size=50, line_search_fn='strong_wolfe')
    for _ in range(1, n_lbfgs + 1):
        opt.step(closure)
        with torch.no_grad():
            val = mse_fn(wm(X_val), Y_val).item()
        if val < best_val:
            best_val = val
            best_state = {k: v.clone() for k, v in wm.state_dict().items()}
    wm.load_state_dict(best_state); wm.eval()
    return wm, best_val


class CartPoleTrainer:
    """Train Policy for CartPole via frozen WM (no energy term — pure stabilization)."""
    def __init__(self, wm, policy, lr=1e-3):
        self.wm = wm; self.policy = policy
        self.opt = torch.optim.Adam(policy.parameters(), lr=lr)
        self.wm.eval()
        for p in self.wm.parameters(): p.requires_grad = False

    def train_epoch(self, s_dataset, batch_size=256):
        N = s_dataset.shape[0]
        total_loss = 0
        for _ in range(max(1, N // batch_size)):
            idx = torch.randint(0, N, (batch_size,))
            s_b = s_dataset[idx]
            self.policy.train(); self.opt.zero_grad()
            a = self.policy(s_b)
            s_pred = self.wm(torch.cat([s_b, a], dim=-1))
            # Stabilize: minimize theta, x, velocity
            loss = (s_pred[:, 2].pow(2).mean() + 0.1 * s_pred[:, 0].pow(2).mean() +
                    0.5 * s_pred[:, 3].pow(2).mean() + 0.01 * a.pow(2).mean())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 10.0)
            self.opt.step()
            total_loss += loss.item()
        return {'total': total_loss / max(1, N // batch_size)}

    def get_action(self, s):
        self.policy.eval()
        if isinstance(s, np.ndarray):
            s = torch.tensor(s, dtype=torch.float32)
        if s.dim() == 1: s = s.unsqueeze(0)
        with torch.no_grad():
            return self.policy(s).squeeze().item()


def evaluate_policy(trainer, g=G_DEF, n_trials=20, max_steps=500, label=''):
    """Evaluate on CartPole with specified gravity."""
    successes = 0; all_steps = []
    for trial in range(n_trials):
        seed = 42 + trial * 100
        np.random.seed(seed)
        th = np.random.uniform(-0.05, 0.05)
        s_raw = torch.tensor([[0.0, 0.0, th, 0.0]], dtype=torch.float32)
        for step in range(max_steps):
            s_norm = s_raw.clone()
            s_norm[:, 0] /= X_S; s_norm[:, 1] /= XD_S
            s_norm[:, 2] /= TH_S; s_norm[:, 3] /= THD_S
            a_norm = trainer.get_action(s_norm[0].numpy())
            s_raw = step_cartpole(s_raw, torch.tensor([a_norm]), g=g)
            theta, x = s_raw[0, 2].item(), s_raw[0, 0].item()
            if abs(theta) > 0.21 or abs(x) > 2.4:
                break
        all_steps.append(step + 1)
        if step + 1 >= max_steps: successes += 1
        if trial < 5:
            print(f"  [{label}] T{trial+1} {'✓' if step+1>=max_steps else '✗'} "
                  f"steps={step+1}")
    return successes, all_steps


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(42); np.random.seed(42)

    print("=" * 70)
    print("CartPole: Full ProtoKAN Framework + Continual Learning")
    print("=" * 70)

    # 1. Train ProtoKAN WM
    print("\n[1] Training ProtoKAN WM [5,16,4] on CartPole g=9.8...")
    X_old, Y_old = generate_wm_data(g=9.8, n=5000, device=device)
    wm, wm_val = train_wm(X_old, Y_old, 'protokan', 100, device)
    print(f"  val_mse={wm_val:.6f}")

    # 2. Train KAN Policy
    print("\n[2] Training KAN Policy [4,12,12,1] via ProtoKAN WM...")
    s_pol = generate_policy_states(15000, device)
    policy = KANPolicy(state_dim=4, action_dim=1, hidden_dim=12, n_layers=2).to(device)
    trainer = CartPoleTrainer(wm, policy)
    for ep in range(1, 201):
        ld = trainer.train_epoch(s_pol)
        if ep % 50 == 0: print(f"  Epoch {ep:3d}  loss={ld['total']:.4f}")

    # 3. Evaluate on g=9.8
    print("\n[3] Evaluate on g=9.8...")
    s_norm, st_norm = evaluate_policy(trainer, g=9.8, label='g=9.8')
    print(f"  g=9.8: {s_norm}/20 ({s_norm*5}%)  mean_steps={np.mean(st_norm):.0f}")

    # 4. Evaluate on g=15.0 (zero-shot)
    print("\n[4] Evaluate on g=15.0 (zero-shot)...")
    s_15_before, st_15_before = evaluate_policy(trainer, g=15.0, label='g=15_before')
    print(f"  g=15 before: {s_15_before}/20 ({s_15_before*5}%)  "
          f"mean_steps={np.mean(st_15_before):.0f}")

    # 5. WM adaptation
    print("\n[5] WM adaptation to g=15.0...")
    for p in wm.parameters(): p.requires_grad = True
    X_new, Y_new = generate_wm_data(g=15.0, n=500, device=device)
    mse_fn = nn.MSELoss(); opt = torch.optim.Adam(wm.parameters(), lr=1e-3)
    new_errs = []; old_errs = []
    for step in range(1, 31):
        idx = np.random.choice(len(X_new), 64, replace=False)
        wm.train(); opt.zero_grad()
        loss = mse_fn(wm(X_new[idx]), Y_new[idx])
        loss.backward(); opt.step()
        wm.eval()
        with torch.no_grad():
            new_errs.append(mse_fn(wm(X_new), Y_new).item())
            old_errs.append(mse_fn(wm(X_old[:500].to(device)), Y_old[:500].to(device)).item())
    print(f"  new_mse: {new_errs[0]:.6f} → {new_errs[-1]:.6f}")
    print(f"  old_mse: {old_errs[0]:.6f} → {old_errs[-1]:.6f}  "
          f"(forgetting={old_errs[-1]/old_errs[0]:.3f})")

    # 6. Policy recovery
    print("\n[6] Policy recovery (retrain via adapted WM)...")
    for p in wm.parameters(): p.requires_grad = False
    wm.eval()
    for ep in range(1, 101):
        ld = trainer.train_epoch(s_pol)
        if ep % 30 == 0: print(f"  Retrain epoch {ep:3d}  loss={ld['total']:.4f}")

    # 7. Final evaluation
    print("\n[7] Evaluate on g=15.0 (AFTER recovery)...")
    s_15_after, st_15_after = evaluate_policy(trainer, g=15.0, label='g=15_after')
    print(f"  g=15 after: {s_15_after}/20 ({s_15_after*5}%)  "
          f"mean_steps={np.mean(st_15_after):.0f}")

    print("\n[8] Verify g=9.8 preserved...")
    s_norm_after, st_norm_after = evaluate_policy(trainer, g=9.8, label='g=9.8_after')
    print(f"  g=9.8 after: {s_norm_after}/20 ({s_norm_after*5}%)  "
          f"mean_steps={np.mean(st_norm_after):.0f}")

    # Report
    print("\n" + "=" * 70)
    print("CART POLE CONTINUAL LEARNING RESULT")
    print("=" * 70)
    print(f"  g=9.8 initial:        {s_norm}/20 ({s_norm*5}%)")
    print(f"  g=15.0 before adapt:  {s_15_before}/20 ({s_15_before*5}%)")
    print(f"  g=15.0 AFTER recovery: {s_15_after}/20 ({s_15_after*5}%)")
    print(f"  g=9.8 preserved:      {s_norm_after}/20 ({s_norm_after*5}%)")
    print(f"  WM forgetting ratio:   {old_errs[-1]/old_errs[0]:.3f}")


if __name__ == '__main__':
    main()
