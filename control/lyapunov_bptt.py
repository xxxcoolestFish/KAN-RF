"""Lyapunov-BPTT: Auto-synthesize Lyapunov function from WM dynamics,
then train Policy to minimize it over multi-step imagined rollouts.

Core idea: instead of hand-crafting loss weights per dimension,
solve Riccati to get P matrix. V(s) = (s-s*)^T P (s-s*) is a
Lyapunov function automatically tailored to the system's dynamics.
"""
import torch, torch.nn as nn
import numpy as np


def discover_tiers_from_wm(wm, state_dim, n_samples=200, device='cpu'):
    """Discover Tier 0 (directly controllable) from WM Jacobian.

    Returns list of tier assignments: tier_of[i] = 0, 1, 2, ...
    """
    wm.eval()
    was_frozen = not next(wm.parameters()).requires_grad
    if was_frozen:
        for p in wm.parameters(): p.requires_grad = True

    jac_acc = torch.zeros(state_dim, device=device)
    for _ in range(n_samples):
        s = torch.randn(1, state_dim, device=device).clamp(-1, 1)
        a = torch.zeros(1, 1, device=device, requires_grad=True)
        sp = wm(torch.cat([s, a], dim=-1))
        for i in range(state_dim):
            g = torch.autograd.grad(sp[0, i], a, retain_graph=True)[0]
            jac_acc[i] += g[0, 0].abs()

    if was_frozen:
        for p in wm.parameters(): p.requires_grad = False

    jac = (jac_acc / n_samples).cpu().numpy()
    # Tier 0: top 40% most controllable
    t = np.percentile(jac, 60)
    tier_of = np.zeros(state_dim, dtype=int)
    tier_of[jac >= t] = 0
    tier_of[jac < t] = 1
    return tier_of


def synthesize_lyapunov(wm, s_dataset, s_target, state_dim,
                         horizon=10, r_weight=0.1, n_samples=300,
                         q_goal=10.0, q_means=1.0, device='cpu'):
    """Auto-synthesize Lyapunov function from WM via Riccati.

    Returns P, A, B, tier_of. Same as before.
    """
    wm.eval()
    was_frozen = not next(wm.parameters()).requires_grad
    if was_frozen:
        for p in wm.parameters(): p.requires_grad = True

    n = state_dim
    N = min(n_samples, s_dataset.shape[0])
    idx = torch.randperm(s_dataset.shape[0])[:N]
    s_batch = s_dataset[idx].to(device)

    A_sum = torch.zeros(n, n, device=device)
    B_sum = torch.zeros(n, 1, device=device)

    for i in range(N):
        s = s_batch[i:i+1]
        a = torch.zeros(1, 1, device=device, requires_grad=True)
        s_pred = wm(torch.cat([s, a], dim=-1))
        for j in range(n):
            g = torch.autograd.grad(s_pred[0, j], a, retain_graph=True)[0]
            B_sum[j, 0] += g[0, 0].item()
        s_j = s.clone().detach().requires_grad_(True)
        a0 = torch.zeros(1, 1, device=device)
        sp = wm(torch.cat([s_j, a0], dim=-1))
        for j in range(n):
            for k in range(n):
                g = torch.autograd.grad(sp[0, j], s_j, retain_graph=True)[0]
                A_sum[j, k] += g[0, k].item()

    if was_frozen:
        for p in wm.parameters(): p.requires_grad = False

    A = (A_sum / N).to(device)
    B = (B_sum / N).to(device)

    tier_of = discover_tiers_from_wm(wm, state_dim, device=device)
    Q = torch.diag(torch.tensor(
        [q_goal if tier_of[i] > 0 else q_means for i in range(n)],
        dtype=torch.float32, device=device))

    P = Q.clone()
    R = torch.tensor([[r_weight]], device=device)
    for _ in range(horizon):
        BtPB = B.T @ P @ B
        BtPA = B.T @ P @ A
        AtPA = A.T @ P @ A
        AtPB = A.T @ P @ B
        P = Q + AtPA - (AtPB @ BtPA) / (BtPB + R)

    return P, A, B, tier_of


def synthesize_dual_lyapunov(wm, s_dataset, s_target, state_dim,
                               horizon=10, r_weight=0.1, n_samples=300,
                               q_goal=10.0, q_means_low=0.1, q_means_high=5.0,
                               device='cpu'):
    """Synthesize TWO Lyapunov functions for swing-up tasks.

    P_swing: low Q on Tier 0 (means) → encourages velocity for swing-up
    P_stable: high Q on Tier 0 → damps velocity for stabilization

    Mode is selected by thresholding V_stable(s):
      - V_stable(s) > threshold → use P_swing (far, need to swing)
      - V_stable(s) ≤ threshold → use P_stable (near, need to stabilize)

    Threshold is automatically set as the median V_stable on training data.
    """
    P_stable, A, B, tier_of = synthesize_lyapunov(
        wm, s_dataset, s_target, state_dim,
        horizon=horizon, r_weight=r_weight, n_samples=n_samples,
        q_goal=q_goal, q_means=q_means_high, device=device)

    P_swing, _, _, _ = synthesize_lyapunov(
        wm, s_dataset, s_target, state_dim,
        horizon=horizon, r_weight=r_weight, n_samples=n_samples,
        q_goal=q_goal, q_means=q_means_low, device=device)

    # Auto-compute threshold from training data
    N = min(500, s_dataset.shape[0])
    idx = torch.randperm(s_dataset.shape[0])[:N]
    s_sample = s_dataset[idx].to(device)
    with torch.no_grad():
        err = s_sample - s_target.to(device)
        v_vals = (err @ P_stable @ err.T).diag()
        threshold = v_vals.median().item()

    print(f"  Dual Lyapunov: P_swing(Tier0={q_means_low})  "
          f"P_stable(Tier0={q_means_high})  threshold={threshold:.3f}")

    return P_stable, P_swing, threshold, A, B, tier_of


class LyapunovBPTTTrainer:
    """Train Policy via BPTT with auto-synthesized Lyapunov stability.

    For each batch of states s_0, unroll H steps:
        V(s_t) = (s_t - s*)^T P (s_t - s*)

    Loss has two terms:
        1. L_value: average V(s_t) over the trajectory (want states close to target)
        2. L_stable: max(0, V(s_{t+1}) - V(s_t) + α·V(s_t))
           (Lyapunov descent condition — penalize if V increases)
    """

    def __init__(self, wm, policy, s_target, P, A, B,
                 lr=1e-3, horizon=3, gamma=0.9, alpha=0.1, device='cpu'):
        self.wm = wm
        self.policy = policy.to(device)
        self.s_target = s_target.to(device)
        self.P = P.to(device)
        self.horizon = horizon
        self.gamma = gamma
        self.alpha = alpha
        self.device = device

        wm.eval()
        for p in wm.parameters():
            p.requires_grad = False

        self.opt = torch.optim.Adam(policy.parameters(), lr=lr)
        self.loss_history = []

    def _lyapunov(self, s):
        """V(s) = (s - s*)^T P (s - s*)"""
        e = s - self.s_target.expand(s.shape[0], -1)
        return (e @ self.P @ e.T).diag()  # (B,)

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
            V_prev = self._lyapunov(s_cur)
            total = 0.0

            for t in range(self.horizon):
                a = self.policy(s_cur)
                s_cur = self.wm(torch.cat([s_cur, a], dim=-1))
                V_cur = self._lyapunov(s_cur)

                # Term 1: minimize V (standard)
                value_loss = V_cur.mean()

                # Term 2: Lyapunov descent V_{t+1} - V_t ≤ -α·V_t
                violation = torch.clamp(V_cur - V_prev + self.alpha * V_prev, min=0)
                stable_loss = violation.mean()

                step_loss = value_loss + 0.5 * stable_loss
                total = total + (self.gamma ** t) * step_loss
                V_prev = V_cur

            loss = total / self.horizon
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 10.0)
            self.opt.step()
            total_loss += loss.item()

        avg_loss = total_loss / n_batches
        self.loss_history.append({'total': avg_loss})
        return self.loss_history[-1]

    def get_action(self, s):
        self.policy.eval()
        if isinstance(s, np.ndarray):
            s = torch.tensor(s, dtype=torch.float32, device=self.device)
        if s.dim() == 1:
            s = s.unsqueeze(0)
        with torch.no_grad():
            return self.policy(s).squeeze().cpu().item()
