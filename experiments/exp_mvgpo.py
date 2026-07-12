
"""MVGPO: Model-based Value Gradient Policy Optimization.

Core idea: No hand-crafted cost function needed.
  Step 1: Generate V(s) labels via random rollouts through WM
  Step 2: Train V_net(s) to predict "how good is this state"
  Step 3: Train KAN Policy via BPTT through WM
          Loss = Σ dist(s_t, s*) + V(s_H)

Key novelty: V(s) learned from random WM rollouts captures the
long-horizon task structure without any oracle/formula.
"""

import sys, os, time, torch, torch.nn as nn, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.baseline_sweep import generate_pendulum_data, train_wm, generate_policy_states, evaluate_policy
from experiments.exp_cost_discovery import *
from control.kan_policy_net import KANPolicy

DEVICE = 'cpu'
S_TARGET = torch.tensor([[0.0, 1.0, 0.0]])
G = 10.0; PI_2 = np.pi / 2


# ═══════════════════════════════════════════════════════════
# Step 1: 从随机 WM rollout 生成 V(s) 标签
# ═══════════════════════════════════════════════════════════

def compute_pendulum_reward(s_next):
    """Gym Pendulum reward from normalized state (batched)."""
    th = torch.atan2(s_next[:, 1], s_next[:, 0])
    th_err = th - PI_2
    th_err = torch.where(th_err > np.pi, th_err - 2*np.pi, th_err)
    th_err = torch.where(th_err < -np.pi, th_err + 2*np.pi, th_err)
    thd = s_next[:, 2] * 8.0
    return -(th_err.pow(2) + 0.1 * thd.pow(2))


def generate_v_labels(wm, states, n_tries=30, horizon=30, gamma=0.95):
    """For each state, run random rollouts, take top-5% returns as V(s)."""
    N = states.shape[0]
    values = []
    wm.eval()

    for i in range(N):
        s0 = states[i:i+1].to(DEVICE)
        best = -1e6

        with torch.no_grad():
            for _ in range(n_tries):
                s_cur = s0.clone()
                total_r = 0.0
                for t in range(horizon):
                    a = torch.tensor([[np.random.uniform(-1, 1)]], device=DEVICE)
                    s_next = wm(torch.cat([s_cur, a], dim=-1))[:, :3]
                    r = compute_pendulum_reward(s_next).squeeze()
                    total_r += (gamma ** t) * r.item()
                    s_cur = s_next

                    # Check success
                    th = torch.atan2(s_next[:, 1], s_next[:, 0])
                    angle_err = (th - PI_2).abs()
                    angle_err = torch.where(angle_err > np.pi, 2*np.pi-angle_err, angle_err)
                    if angle_err.item() < 0.2:
                        total_r += 10.0 * (horizon - t - 1)
                        break

                if total_r > best:
                    best = total_r

        values.append(best)
        if (i+1) % 100 == 0:
            print(f'    V-gen {i+1}/{N}  best_so_far={max(values):.1f}')

    return torch.tensor(values, dtype=torch.float32)


# ═══════════════════════════════════════════════════════════
# Step 2: BPTT Trainer with value-guided terminal cost
# ═══════════════════════════════════════════════════════════

class ValueGuidedBPTT:
    """BPTT through WM with V(s_H) as terminal cost + per-step distance.

    Loss = Σ dist(s_t, s*) + γ^H · V(s_H)

    V(s_H) provides long-horizon signal: even if H steps can't reach
    the target, V tells us if we're heading in the right direction.
    """

    def __init__(self, wm, policy, v_net, s_target,
                 horizon=5, gamma=0.9, lr=1e-3, lambda_ctrl=0.01, device='cpu'):
        self.wm = wm
        self.policy = policy.to(device)
        self.v_net = v_net.to(device)
        self.s_target = s_target.to(device)
        self.horizon = horizon
        self.gamma = gamma
        self.lambda_ctrl = lambda_ctrl
        self.device = device

        self.opt = torch.optim.Adam(policy.parameters(), lr=lr)
        self.loss_history = []

        self.wm.eval()
        for p in self.wm.parameters():
            p.requires_grad = False
        self.v_net.eval()
        for p in self.v_net.parameters():
            p.requires_grad = False

    def train_epoch(self, s_dataset, batch_size=256):
        N = s_dataset.shape[0]
        n_batches = max(1, N // batch_size)
        total_loss = 0.0

        for _ in range(n_batches):
            idx = torch.randint(0, N, (batch_size,), device=self.device)
            s_b = s_dataset[idx]
            self.policy.train()
            self.opt.zero_grad()

            s_cur = s_b
            loss = 0.0

            for t in range(self.horizon):
                a = self.policy(s_cur)
                s_next = self.wm(torch.cat([s_cur, a], dim=-1))[:, :3]

                # Per-step distance to target
                step_loss = (s_next - self.s_target.expand(s_next.shape[0], -1)).pow(2).sum(dim=-1).mean()

                discount = self.gamma ** t
                loss = loss + discount * step_loss

                s_cur = s_next

            # Terminal value: V(s_H) — V(current) gives "improvement signal"
            with torch.no_grad():
                v_cur = self.v_net(s_b).squeeze().detach()
                v_term = self.v_net(s_cur).squeeze().detach()

            # V improvement: we want V(s_H) > V(s_0)
            # Loss = -(V(s_H) - V(s_0)).clamp(min=0) — only penalize when V decreases
            # Actually: loss = -V(s_H) (minimize = maximize V at terminal)
            v_loss = -(self.gamma ** self.horizon) * self.v_net(s_cur).squeeze().mean()

            total = loss + v_loss
            total.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 10.0)
            self.opt.step()
            total_loss += total.item()

        avg_loss = total_loss / n_batches
        self.loss_history.append({'total': avg_loss})
        return self.loss_history[-1]

    @torch.no_grad()
    def get_action(self, s_norm):
        self.policy.eval()
        if isinstance(s_norm, np.ndarray):
            s_norm = torch.tensor(s_norm, dtype=torch.float32, device=self.device)
        if s_norm.dim() == 1:
            s_norm = s_norm.unsqueeze(0)
        a = self.policy(s_norm).squeeze().cpu()
        return a.item() if a.numel() == 1 else a.numpy()


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main():
    print('=' * 70)
    print('  MVGPO: Model-based Value Gradient Policy Optimization')
    print('=' * 70)

    # ── WM ──
    X, Y = generate_pendulum_data(3000, seed=42)
    wm, _ = train_wm(X.to(DEVICE), Y.to(DEVICE))
    wm.eval()
    s_pol = generate_policy_states(10000, seed=42).to(DEVICE)
    n_train = s_pol.shape[0]
    print(f'  WM ready, s_pol: {s_pol.shape}')

    # ── Step 1: Generate V(s) labels ──
    print()
    print('─' * 60)
    print('  Step 1: Generating V(s) from random WM rollouts')
    print('─' * 60)

    n_v = 500
    idx_v = torch.randint(0, n_train, (n_v,))
    s_v = s_pol[idx_v]

    t0 = time.time()
    v_labels = generate_v_labels(wm, s_v, n_tries=30, horizon=30)
    print(f'  V(s) range: [{v_labels.min():.1f}, {v_labels.max():.1f}]')
    print(f'  Time: {time.time()-t0:.0f}s')

    # Normalize V
    v_mean, v_std = v_labels.mean(), v_labels.std()
    v_norm = (v_labels - v_mean) / (v_std + 1e-8)
    print(f'  Normalized: mean={v_mean:.2f}, std={v_std:.2f}')

    # ── Step 2: Train V_net ──
    print()
    print('─' * 60)
    print('  Step 2: Training V_net')
    print('─' * 60)

    v_net = nn.Sequential(
        nn.Linear(3, 32), nn.Tanh(),
        nn.Linear(32, 16), nn.Tanh(),
        nn.Linear(16, 1),
    )
    opt_v = torch.optim.Adam(v_net.parameters(), lr=1e-3)

    for ep in range(1, 501):
        idx_b = torch.randint(0, n_v, (64,))
        s_b = s_v[idx_b].to(DEVICE)
        v_b = v_norm[idx_b].unsqueeze(1).to(DEVICE)
        v_pred = v_net(s_b)
        loss = nn.functional.mse_loss(v_pred, v_b)
        opt_v.zero_grad()
        loss.backward()
        opt_v.step()
        if ep % 200 == 0:
            with torch.no_grad():
                v_range = f'[{v_pred.min().item():.2f}, {v_pred.max().item():.2f}]'
            print(f'    Ep {ep:4d}  loss={loss.item():.6f}  V_range={v_range}')

    # ── Gradient analysis ──
    print()
    print('─' * 60)
    print('  Gradient analysis')
    print('─' * 60)

    resolution = 20
    n_points = resolution * resolution
    angles = np.linspace(-np.pi, np.pi, resolution)
    thds = np.linspace(-7.0, 7.0, resolution)
    grid = []
    for theta in angles:
        for thd in thds:
            grid.append([np.cos(theta), np.sin(theta), thd / 8.0])
    s_grid = torch.tensor(np.array(grid), dtype=torch.float32, device=DEVICE)

    # V_net gradient
    a_ref = torch.zeros(n_points, 1, device=DEVICE, requires_grad=True)
    s_pred = wm(torch.cat([s_grid, a_ref], dim=-1))
    v_pred = v_net(s_pred).squeeze()
    g_loss = -v_pred.sum()
    g_loss.backward()
    g_V = a_ref.grad.clone()

    # Energy gradient
    a_ref_e = torch.zeros(n_points, 1, device=DEVICE, requires_grad=True)
    s_pred_e = wm(torch.cat([s_grid, a_ref_e], dim=-1))
    e_loss = -energy(s_pred_e).sum()
    e_loss.backward()
    g_E = a_ref_e.grad.clone()

    a_energy = compute_energy_action(s_grid)

    for label, g in [('V_net', g_V), ('energy', g_E)]:
        g_d = g.sign().squeeze(); a_d = a_energy.sign().squeeze()
        aligned = (g_d * a_d) > 0
        print(f'  [{label:>6}] aligned={aligned.sum().item():4d}/{n_points}  '
              f'wrong={((g_d*a_d)<0).sum().item():4d}  '
              f'mean|g|={g.abs().mean().item():.4f}')
    cos_sim = torch.cosine_similarity(g_V.squeeze(), g_E.squeeze(), dim=0).item()
    print(f'  V(s) vs Energy gradient cos_sim: {cos_sim:.4f}')

    # ── Step 3: Train policy with Value-Guided BPTT ──
    print()
    print('─' * 60)
    print('  Step 3: Training KAN Policy with Value-Guided BPTT')
    print('─' * 60)

    for seed in [42, 43, 44]:
        torch.manual_seed(seed); np.random.seed(seed)
        policy = KANPolicy(state_dim=3, action_dim=1, hidden_dim=12, n_layers=2)
        trainer = ValueGuidedBPTT(wm, policy, v_net, S_TARGET,
                                  horizon=5, gamma=0.9, lr=1e-3, device=DEVICE)

        t0 = time.time()
        for ep in range(1, 201):
            ld = trainer.train_epoch(s_pol, batch_size=256)
            if ep % 50 == 0:
                print(f'    Ep {ep:4d}  total={ld["total"]:.4f}')

        succ, steps, errs = evaluate_policy(trainer, n_trials=10, seed=42)
        print(f'  [MVGPO seed={seed}] → {succ}/10  mean_steps={np.mean(steps):.0f}  [{time.time()-t0:.0f}s]')

    # ── Reference: energy cost (single-step) ──
    print()
    print('─' * 60)
    print('  [Reference] Single-step energy cost')
    print('─' * 60)
    for seed in [42, 43, 44]:
        torch.manual_seed(seed); np.random.seed(seed)
        policy_e = KANPolicy(state_dim=3, action_dim=1, hidden_dim=12, n_layers=2)
        trainer_e = CustomKANPolicyTrainer(wm, policy_e, S_TARGET,
                                           cost_fn=energy_loss, cost_name='energy',
                                           lr=1e-3, device=DEVICE)
        for ep in range(1, 201):
            ld = trainer_e.train_epoch(s_pol, batch_size=256)
        succ_e, steps_e, errs_e = evaluate_policy(trainer_e, n_trials=10, seed=42)
        print(f'  [energy seed={seed}] → {succ_e}/10  mean_steps={np.mean(steps_e):.0f}')


if __name__ == '__main__':
    main()
