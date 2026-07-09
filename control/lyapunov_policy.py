"""Lyapunov-BPTT with ProtoKAN Policy.

Policy learns to minimize V(s) = (s-s*)^T P (s-s*) via BPTT through WM.
Policy is ProtoKAN (small sigma) — for online fine-tuning with local updates.

Key design:
  - Policy learns "minimize quadratic cost" as a meta-ability, not tied to specific P
  - Pre-training: multi-physics (different WMs + Ps), same Policy
  - Deployment: WM adapts → new P → Policy fine-tunes online (few gradient steps)
  - Local ProtoKAN updates → zero forgetting of old knowledge
"""
import torch, torch.nn as nn, numpy as np
from kanrf import ProtoKANLayer


class ProtoKANPolicyNet(nn.Module):
    """ProtoKAN Policy with small sigma for local adaptation."""

    def __init__(self, state_dim=3, hidden=12, n_prototypes=16,
                 init_log_sigma=-1.5):
        super().__init__()
        self.l1 = ProtoKANLayer(state_dim, hidden, n_prototypes=n_prototypes)
        self.l2 = ProtoKANLayer(hidden, hidden, n_prototypes=n_prototypes)
        self.out = nn.Linear(hidden, 1)
        for l in [self.l1, self.l2]:
            l.log_sigma.data.fill_(init_log_sigma)

    def forward(self, s):
        return torch.tanh(self.out(self.l2(self.l1(s))))


class LyapunovPolicyTrainer:
    """Train ProtoKAN Policy via BPTT with Lyapunov loss V(s)=(s-s*)^T P (s-s*).

    Supports multi-physics pre-training and online fine-tuning.
    """

    def __init__(self, state_dim=3, horizon=3, gamma=0.9, lr=1e-3, device='cpu'):
        self.policy = ProtoKANPolicyNet(state_dim).to(device)
        self.state_dim = state_dim
        self.horizon = horizon
        self.gamma = gamma
        self.device = device
        self.opt = torch.optim.Adam(self.policy.parameters(), lr=lr)

    def _lyapunov_cost(self, s, P, s_target):
        """V(s) = (s - s*)^T P (s - s*), batched."""
        err = s - s_target.expand(s.shape[0], -1)
        return (err @ P.to(s.device) @ err.T).diag()  # (B,)

    def train_epoch(self, wm, P, s_dataset, s_target, batch_size=256):
        """One epoch of BPTT with Lyapunov loss."""
        P = P.to(self.device)
        wm.eval()
        for p in wm.parameters(): p.requires_grad = False

        N = s_dataset.shape[0]
        n_batches = max(1, N // batch_size)
        total_loss = 0.0

        for _ in range(n_batches):
            idx = torch.randint(0, N, (batch_size,), device=self.device)
            s_b = s_dataset[idx]
            self.policy.train(); self.opt.zero_grad()

            s_cur = s_b
            loss = 0.0
            for t in range(self.horizon):
                a = self.policy(s_cur)
                s_cur = wm(torch.cat([s_cur, a], dim=-1))
                loss = loss + (self.gamma ** t) * self._lyapunov_cost(
                    s_cur, P, s_target).mean()

            loss = loss / self.horizon
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 10.0)
            self.opt.step()
            total_loss += loss.item()

        return total_loss / n_batches

    def fine_tune_online(self, wm, P, s, s_target, lr=1e-4):
        """One online gradient step with new P. Low LR for local adaptation."""
        P = P.to(self.device)
        wm.eval()
        for p in wm.parameters(): p.requires_grad = False
        s_b = s.unsqueeze(0)

        for pg in self.opt.param_groups:
            orig_lr = pg['lr']
            pg['lr'] = lr

        self.policy.train(); self.opt.zero_grad()
        s_cur = s_b
        loss = 0.0
        for t in range(self.horizon):
            a = self.policy(s_cur)
            s_cur = wm(torch.cat([s_cur, a], dim=-1))
            loss = loss + (self.gamma ** t) * self._lyapunov_cost(
                s_cur, P, s_target).mean()
        loss.backward()
        self.opt.step()

        for pg in self.opt.param_groups:
            pg['lr'] = orig_lr

        return loss.item()

    def get_action(self, s):
        self.policy.eval()
        if isinstance(s, np.ndarray):
            s = torch.tensor(s, dtype=torch.float32, device=self.device)
        if s.dim() == 1: s = s.unsqueeze(0)
        with torch.no_grad():
            return self.policy(s).squeeze().cpu().item()
