
"""Experiment: Analyze and auto-discover energy-like cost functions from WM.

Phase 1 — 复现基线差距
Phase 2 — 梯度诊断：不同代价函数的梯度方向对比
Phase 3 — 从 WM 结构中自动发现类能量代价函数
"""

import torch
import torch.nn as nn
import numpy as np
import sys, os, time, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kanrf import ProtoKAN
from control.kan_policy_net import KANPolicy, KANPolicyTrainer
from experiments.baseline_sweep import (
    generate_pendulum_data, train_wm, generate_policy_states, evaluate_policy
)

PI_2 = np.pi / 2
S_TARGET = torch.tensor([[0.0, 1.0, 0.0]])
G = 10.0

DEVICE = 'cpu'
torch.set_printoptions(precision=6, sci_mode=False)


# ═══════════════════════════════════════════════════════════
# Cost Function Definitions
# ═══════════════════════════════════════════════════════════

def energy(s):
    """Pendulum energy from normalized state [cosθ, sinθ, θ̇/8]."""
    thd = s[:, 2] * 8.0
    return 0.5 * thd.pow(2) + G * s[:, 1]

def energy_loss(s_pred, s_batch, s_target):
    """Energy-guided loss: maximize energy gain when below target."""
    B = s_pred.shape[0]
    E_cur = energy(s_batch)
    E_pred = energy(s_pred)
    E_des = G
    energy_deficit = E_des - E_cur
    energy_gain = (E_pred - E_cur) * torch.sign(energy_deficit)

    sin = s_batch[:, 1:2]
    w_swing = ((1.0 - sin) / 2.0).clamp(0.0, 1.0)
    w_stable = ((1.0 + sin) / 2.0).clamp(0.0, 1.0)

    e_loss = -energy_gain.mean()
    d_loss = (w_stable * (s_pred - s_target.expand(B, -1)).pow(2).sum(dim=-1, keepdim=True)).mean()
    return e_loss + d_loss, {'energy': e_loss.item(), 'dist': d_loss.item()}


def mse_loss(s_pred, s_batch, s_target):
    """Standard MSE to target state."""
    B = s_pred.shape[0]
    loss = (s_pred - s_target.expand(B, -1)).pow(2).sum(dim=-1).mean()
    return loss, {'mse': loss.item()}


def synthesize_lyapunov_P(wm, s_dataset, state_dim=3, causal=True, n_samples=300):
    """Auto-synthesize P matrix from WM Jacobian via Riccati.

    This replicates the Lyapunov synthesis from lyapunov_bptt.py
    and lyapunov_policy.py.

    Args:
        causal: If True, use causal Q weighting (Tier-based)
    Returns:
        P: (state_dim, state_dim) Lyapunov matrix
    """
    from control.lyapunov_bptt import synthesize_lyapunov
    P, _, _, _ = synthesize_lyapunov(
        wm, s_dataset, S_TARGET, state_dim,
        horizon=10, n_samples=n_samples, device=DEVICE
    )
    return P


def lyapunov_loss(s_pred, s_batch, s_target, P=None):
    """Lyapunov loss: (s - s*)^T P (s - s*)."""
    B = s_pred.shape[0]
    err = s_pred - s_target.expand(B, -1)
    if P is None:
        # Default: identity → same as MSE
        loss = err.pow(2).sum(dim=-1).mean()
        return loss, {'lyap': loss.item()}
    P_t = P.to(s_pred.device)
    loss = (err @ P_t @ err.T).diag().mean()
    return loss, {'lyap': loss.item()}


# ═══════════════════════════════════════════════════════════
# Phase 1: Baseline Reproduction
# ═══════════════════════════════════════════════════════════

class CustomKANPolicyTrainer(KANPolicyTrainer):
    """Extended KAN Policy trainer supporting multiple cost functions.

    Args:
        cost_fn: callable(s_pred, s_batch, s_target) → (loss_tensor, diagnostics_dict)
        cost_name: string identifier for logging
    """

    def __init__(self, wm, policy, s_target, cost_fn=None, cost_name='custom',
                 lr=1e-3, lambda_ctrl=0.01, device='cpu'):
        super().__init__(wm, policy, s_target, lr=lr, lambda_ctrl=lambda_ctrl, device=device)
        self.cost_fn = cost_fn
        self.cost_name = cost_name

    def train_step(self, s_batch):
        B = s_batch.shape[0]
        self.policy.train()
        self.optimizer.zero_grad()

        a = self.policy(s_batch)
        wm_in = torch.cat([s_batch, a], dim=-1)
        s_pred = self.wm(wm_in)

        if self.cost_fn is not None:
            pred_loss, diag = self.cost_fn(s_pred, s_batch, self.s_target)
        else:
            pred_loss, diag = energy_loss(s_pred, s_batch, self.s_target)

        ctrl_loss = a.pow(2).mean()
        total_loss = pred_loss + self.lambda_ctrl * ctrl_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.clip_grad)
        self.optimizer.step()

        ld = {'total': total_loss.item(), 'pred': pred_loss.item(), 'ctrl': ctrl_loss.item()}
        ld.update(diag)
        self.loss_history.append(ld)
        return ld


def baseline_sweep(wm, s_pol, n_seeds=3):
    """Run baseline comparison across cost functions for multiple seeds."""
    cost_configs = [
        ('energy',  energy_loss,  None),
        ('mse',     mse_loss,     None),
        ('lyapunov', lyapunov_loss, 'P'),
    ]

    results = {}
    for cost_name, cost_fn, extra in cost_configs:
        print(f'\n{"="*60}')
        print(f'  Cost: {cost_name}')
        print(f'{"="*60}')
        seed_results = []

        for seed in range(n_seeds):
            torch.manual_seed(42 + seed)
            np.random.seed(42 + seed)

            policy = KANPolicy(state_dim=3, action_dim=1, hidden_dim=12, n_layers=2)
            n_params = sum(p.numel() for p in policy.parameters())
            print(f'  Seed {seed}: Policy params={n_params}')

            P = None
            if extra == 'P':
                P = synthesize_lyapunov_P(wm, s_pol, causal=True)

            trainer = CustomKANPolicyTrainer(
                wm, policy, S_TARGET, cost_fn=cost_fn, cost_name=cost_name,
                lr=1e-3, lambda_ctrl=0.01, device=DEVICE)

            for ep in range(1, 201):
                ld = trainer.train_epoch(s_pol, batch_size=256)
                if ep % 50 == 0:
                    print(f'    Ep {ep:4d}  total={ld["total"]:.4f}  pred={ld["pred"]:.4f}')

            successes, steps, errors = evaluate_policy(trainer, n_trials=10, seed=42 + seed * 100)
            print(f'    → {successes}/10  mean_steps={np.mean(steps):.0f}  mean_err={np.mean(errors):.3f}')
            seed_results.append({
                'seed': seed, 'successes': successes, 'steps': steps, 'errors': errors,
                'loss_history': trainer.loss_history,
            })

        results[cost_name] = seed_results
        mean_sr = np.mean([r['successes'] for r in seed_results])
        print(f'  [{cost_name}] Mean: {mean_sr:.1f}/10 over {n_seeds} seeds')

    return results


# ═══════════════════════════════════════════════════════════
# Phase 2: Gradient Analysis
# ═══════════════════════════════════════════════════════════

def compute_action_gradient(wm, s_batch, cost_type='energy'):
    """Compute ∂cost/∂a at each state in the batch.

    This reveals WHERE each cost function gives useful gradient signal.
    """
    B = s_batch.shape[0]
    a_ref = torch.zeros(B, 1, device=DEVICE, requires_grad=True)
    s_pred = wm(torch.cat([s_batch, a_ref], dim=-1))

    if cost_type == 'energy':
        E_cur = energy(s_batch)
        E_pred = energy(s_pred)
        E_des = G
        energy_deficit = E_des - E_cur
        energy_gain = (E_pred - E_cur) * torch.sign(energy_deficit)
        loss = -energy_gain.sum()  # sum for gradient per sample

    elif cost_type == 'mse':
        err = (s_pred - S_TARGET.expand(B, -1)).pow(2).sum(dim=-1)
        loss = err.sum()

    elif cost_type == 'lyapunov':
        P = synthesize_lyapunov_P(wm, s_batch, causal=True)
        err = s_pred - S_TARGET.expand(B, -1)
        loss = (err @ P.to(DEVICE) @ err.T).diag().sum()

    else:
        raise ValueError(f'Unknown cost_type: {cost_type}')

    loss.backward()
    grad = a_ref.grad.clone()  # (B, 1)
    return grad


def compute_energy_action(s_batch):
    """Oracle energy controller action for each state."""
    norm = s_batch.clone()
    sin = norm[:, 1].cpu().numpy()
    thd = (norm[:, 2] * 8.0).cpu().numpy()
    E = 0.5 * thd**2 + G * sin
    E_des = G
    a_norm = np.clip(1.5 * (E - E_des) * thd, -1.0, 1.0)
    return torch.tensor(a_norm, dtype=torch.float32, device=DEVICE).unsqueeze(1)


def compute_gradient_metrics(wm, resolution=15):
    """Compute gradient quality metrics across the state space.

    For each cost function, at every grid point:
      1. ∂cost/∂a = g(s) — gradient direction
      2. g(s)·a_energy(s) — alignment with oracle action
      3. ||g(s)|| — gradient magnitude

    Returns dict of metrics per cost type.
    """
    angles = np.linspace(-np.pi, np.pi, resolution)
    thds = np.linspace(-7.0, 7.0, resolution)
    n_points = resolution * resolution

    grid_states = []
    for theta in angles:
        for thd in thds:
            grid_states.append([np.cos(theta), np.sin(theta), thd / 8.0])
    s_grid = torch.tensor(np.array(grid_states), dtype=torch.float32, device=DEVICE)
    print(f'  Grid: {n_points} points ({resolution}×{resolution})')

    a_energy = compute_energy_action(s_grid)  # oracle reference direction

    metrics = {}
    for cost_type in ['energy', 'mse', 'lyapunov']:
        g = compute_action_gradient(wm, s_grid, cost_type)  # (B, 1)
        g_norm = g / (g.norm() + 1e-8)
        a_norm = a_energy / (a_energy.norm() + 1e-8)

        cos_sim = (g_norm * a_norm).sum().item() / n_points
        mean_mag = g.norm().item() / n_points
        max_mag = g.abs().max().item()

        # Where does the gradient point in the WRONG direction?
        wrong_mask = (g * a_energy).squeeze() < 0  # opposite sign
        n_wrong = wrong_mask.sum().item()

        # Where is the gradient near-zero?
        zero_mask = g.abs().squeeze() < 1e-4
        n_zero = zero_mask.sum().item()

        # Analyze spatial distribution of wrong gradients
        wrong_angles = []
        if n_wrong > 0:
            wrong_idx = torch.where(wrong_mask)[0]
            for idx in wrong_idx[:5]:
                s = s_grid[idx]
                th_rad = np.arctan2(s[1].item(), s[0].item())
                thd_val = s[2].item() * 8.0
                wrong_angles.append((th_rad, thd_val))

        print(f'  [{cost_type}] cos_sim={cos_sim:.3f}  mean|g|={mean_mag:.5f}  '
              f'wrong={n_wrong}/{n_points}  zero={n_zero}/{n_points}')
        if wrong_angles:
            print(f'           wrong examples: θ={wrong_angles[0][0]:.2f} θ̇={wrong_angles[0][1]:.2f}  '
                  f'θ={wrong_angles[-1][0]:.2f} θ̇={wrong_angles[-1][1]:.2f}')

        metrics[cost_type] = {
            'cos_sim': cos_sim,
            'mean_mag': mean_mag,
            'max_mag': max_mag,
            'n_wrong': n_wrong,
            'n_zero': n_zero,
            'n_total': n_points,
        }
    return metrics, s_grid


# ═══════════════════════════════════════════════════════════
# Phase 3: Auto-Discover Cost Function (核心创新)
# ═══════════════════════════════════════════════════════════

class DiscoveredCost:
    """Auto-discovered Lyapunov-like cost from WM multi-step rollout.

    Core idea: Learn a parameterized cost function L_ψ(s) that is:
      1. A minimum at the target s* (L_ψ(s*) = 0)
      2. Globally monotonic: the gradient of L_ψ w.r.t. action aligns
         with the true improvement direction everywhere

    We discover L_ψ by analyzing the WM's multi-step predictions:
      - Roll out the WM H steps from each state with action a=0
      - Compute the "natural dynamics trajectory" s_0, s_1, ..., s_H
      - Identify directions in state space that change SLOWLY under
        natural dynamics (these are the "energy-like" coordinates)
      - Learn L_ψ to penalize deviation from target in these directions

    Args:
        wm: ProtoKAN world model
        state_dim: state dimension
        n_components: number of slow-mode directions to discover
        horizon: WM rollout horizon for slow-mode discovery
    """

    def __init__(self, wm, state_dim=3, n_components=1, horizon=20):
        self.wm = wm
        self.state_dim = state_dim
        self.n_components = n_components
        self.horizon = horizon

        # Learned slow-mode directions (linear projection for now)
        # W: (state_dim, n_components) — projection from state to slow features
        # w_target: (n_components,) — target value of slow features
        self.W = None
        self.w_target = None

    def discover(self, s_dataset):
        """Discover slow-mode directions from WM rollouts.

        Method:
          1. For each state s_i, roll out WM H steps with a=0
          2. Compute the trajectory matrix T_i: (H, state_dim)
          3. Compute the "change matrix" ΔT_i: (H-1, state_dim) = differences
          4. States that change slowly → small differences along those dims
          5. Find directions v where ||ΔT_i · v||² is small on average
             → these are the slow-mode directions
          6. Also identify directions where action has most effect
             → these are the controllable (fast) directions
          7. The energy-like quantity is the SLOW direction that the
             action needs to change for the task

        Returns:
            W: (state_dim, n_components) slow-mode directions
            w_target: (n_components,) target values
        """
        N = s_dataset.shape[0]
        wm.eval()
        print(f'  Discovering slow modes from {N} states (H={self.horizon})...')

        # Collect trajectory difference matrices
        # For each state: roll out H steps, compute Δs at each step
        all_diffs = []  # each: (H, state_dim)

        n_samples = min(N, 500)
        idx = torch.randperm(N)[:n_samples]
        batch_size = 50

        for i in range(0, n_samples, batch_size):
            b_idx = idx[i:i+batch_size]
            s_b = s_dataset[b_idx].to(DEVICE)

            with torch.no_grad():
                s_cur = s_b.clone()
                for _ in range(self.horizon):
                    a_zero = torch.zeros(s_cur.shape[0], 1, device=DEVICE)
                    x = torch.cat([s_cur, a_zero], dim=-1)
                    s_next = self.wm(x)[:, :self.state_dim]
                    diff = (s_next - s_cur).cpu()  # (B, state_dim)
                    all_diffs.append(diff)
                    s_cur = s_next

        D = torch.cat(all_diffs, dim=0)  # (n_samples * H, state_dim)

        # Find slow-mode directions: directions with SMALL variance in D
        # These are directions where the natural dynamics barely change
        # → they need control input to change → they define the task
        cov = D.T @ D / D.shape[0]  # (state_dim, state_dim)
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)

        # Slow modes = eigenvectors with SMALLEST eigenvalues
        # (the dynamics don't change much in these directions naturally)
        idx_sorted = torch.argsort(eigenvalues)
        slow_dirs = eigenvectors[:, idx_sorted[:self.n_components]]  # (state_dim, n_components)

        # Target: project s_target onto slow directions
        s_target_np = S_TARGET.float()
        w_target = (s_target_np @ slow_dirs).squeeze(0)  # (n_components,)

        eig_str = ', '.join([f'{eigenvalues[i]:.6f}' for i in idx_sorted])
        print(f'  Eigenvalues: {eig_str}')

        # Verify: for pendulum, the slowest direction should roughly
        # correspond to the energy (sinθ + const·θ̇²) direction
        print(f'  Slow directions (column = weight per state dim [cos, sin, θ̇/8]):')
        for i in range(self.n_components):
            col = slow_dirs[:, i].cpu().numpy()
            print(f'    v{i}: [{col[0]:.4f}, {col[1]:.4f}, {col[2]:.4f}]  '
                  f'eig={eigenvalues[idx_sorted[i]]:.6f}  '
                  f'target={w_target[i]:.4f}')

        self.W = slow_dirs.cpu()
        self.w_target = w_target.cpu()
        return self.W, self.w_target

    def __call__(self, s_pred, s_batch, s_target):
        """Compute loss = (W^T·(s_pred - s*))^2 in slow-mode space.

        This only penalizes deviations in the slow-mode directions
        (which are the task-relevant directions).
        """
        W = self.W.to(s_pred.device)
        w_tgt = self.w_target.to(s_pred.device)
        B = s_pred.shape[0]

        # Project onto slow modes
        pred_proj = s_pred @ W  # (B, n_components)
        target_proj = w_tgt.unsqueeze(0).expand(B, -1)  # (B, n_components)

        loss = (pred_proj - target_proj).pow(2).sum(dim=-1).mean()
        return loss, {'disc': loss.item()}


def run_gradient_analysis(wm, s_pol):
    """Run Phase 2 gradient analysis with visual diagnostics."""
    print(f'\n{"="*60}')
    print(f'  Phase 2: Gradient Analysis')
    print(f'{"="*60}')

    metrics, s_grid = compute_gradient_metrics(wm, resolution=20)
    return metrics


def run_discovery_experiment(wm, s_pol, n_trials=10):
    """Run Phase 3: discover cost and train policy with it."""
    print(f'\n{"="*60}')
    print(f'  Phase 3: Auto-Discovered Cost')
    print(f'{"="*60}')

    # Discover slow modes
    disc = DiscoveredCost(wm, state_dim=3, n_components=1, horizon=20)
    W, w_target = disc.discover(s_pol)

    # Train policy with discovered cost
    print(f'\n  Training policy with discovered cost...')
    policy = KANPolicy(state_dim=3, action_dim=1, hidden_dim=12, n_layers=2)
    trainer = CustomKANPolicyTrainer(
        wm, policy, S_TARGET, cost_fn=disc, cost_name='discovered',
        lr=1e-3, lambda_ctrl=0.01, device=DEVICE)

    for ep in range(1, 201):
        ld = trainer.train_epoch(s_pol, batch_size=256)
        if ep % 50 == 0:
            print(f'    Ep {ep:4d}  total={ld["total"]:.4f}  pred={ld["pred"]:.4f}')

    successes, steps, errors = evaluate_policy(trainer, n_trials=n_trials, seed=42)
    print(f'  [discovered] → {successes}/{n_trials}  mean_steps={np.mean(steps):.0f}  '
          f'mean_err={np.mean(errors):.3f}')
    return successes, steps, disc


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main():
    print(f'{"="*70}')
    print(f'  Cost Function Discovery Experiment — Pendulum')
    print(f'{"="*70}')

    # Train WM
    print(f'\n{"─"*60}')
    print(f'  Training ProtoKAN WM...')
    print(f'{"─"*60}')
    X, Y = generate_pendulum_data(5000, seed=42)
    wm, _ = train_wm(X.to(DEVICE), Y.to(DEVICE))
    wm.eval()

    # Generate policy training states
    s_pol = generate_policy_states(10000, seed=42).to(DEVICE)

    # Phase 1: Baseline sweep (single seed for speed)
    print(f'\n{"─"*60}')
    print(f'  Phase 1: Baseline Reproduction')
    print(f'{"─"*60}')
    baseline_sweep(wm, s_pol, n_seeds=1)

    # Phase 2: Gradient analysis
    metrics = run_gradient_analysis(wm, s_pol)

    # Phase 3: Auto-discovered cost
    run_discovery_experiment(wm, s_pol, n_trials=10)

    print(f'\n{"="*70}')
    print(f'  Done.')
    print(f'{"="*70}')


if __name__ == '__main__':
    main()
