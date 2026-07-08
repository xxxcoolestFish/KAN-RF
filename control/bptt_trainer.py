"""BPTT Policy Trainer: H-step WM rollout with accumulated loss.

Key idea: Policy is trained not on single-step WM prediction, but on
cumulative loss over a multi-step imagined trajectory. This:
1. Amplifies gradient signal (sum over H steps)
2. Breaks 1-step myopia (Policy sees long-term consequences)
"""
import torch, torch.nn as nn
import numpy as np


class BPTTTrainer:
    """Train Policy via Backpropagation Through Time through frozen WM.

    For each batch of states s_0:
        1. Policy(s_0) → a_0 → WM → s_1
        2. Policy(s_1) → a_1 → WM → s_2
        ...
        H. Policy(s_{H-1}) → a_{H-1} → WM → s_H

        L = Σ_{t=1}^{H} γ^{t-1} · loss(s_t, s_target)

    Gradient flows back through the entire chain to Policy params.
    """

    def __init__(self, wm, policy, s_target, lr=1e-3, horizon=4,
                 gamma=0.9, device='cpu'):
        self.wm = wm
        self.policy = policy.to(device)
        self.s_target = s_target.to(device)
        self.horizon = horizon
        self.gamma = gamma
        self.device = device

        wm.eval()
        for p in wm.parameters():
            p.requires_grad = False

        self.opt = torch.optim.Adam(policy.parameters(), lr=lr)
        self.loss_history = []

    def train_epoch(self, s_dataset, batch_size=256):
        N = s_dataset.shape[0]
        n_batches = max(1, N // batch_size)
        total_loss = 0.0

        for _ in range(n_batches):
            idx = torch.randint(0, N, (batch_size,), device=self.device)
            s_b = s_dataset[idx]
            self.policy.train()
            self.opt.zero_grad()

            loss = self._bptt_loss(s_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 10.0)
            self.opt.step()
            total_loss += loss.item()

        avg_loss = total_loss / n_batches
        self.loss_history.append({'total': avg_loss})
        return self.loss_history[-1]

    def _bptt_loss(self, s_0):
        """H-step imagined rollout with accumulated loss."""
        B = s_0.shape[0]
        s_cur = s_0
        total = 0.0

        for t in range(self.horizon):
            a = self.policy(s_cur)
            s_cur = self.wm(torch.cat([s_cur, a], dim=-1))

            # Per-step loss
            if s_0.shape[1] == 3:  # Pendulum
                loss_t = self._pendulum_loss(s_cur)
            else:  # CartPole / generic
                loss_t = self._stabilization_loss(s_cur)

            total = total + (self.gamma ** t) * loss_t

        # Average over horizon and batch
        return total / self.horizon

    def _pendulum_loss(self, s):
        """Energy-guided loss for Pendulum."""
        B = s.shape[0]
        thd = s[:, 2] * 8.0
        E = 0.5 * thd.pow(2) + 10.0 * s[:, 1]  # current energy
        E_des = 10.0
        energy_loss = -(E - E_des).abs().mean() * 0.1

        # Also penalize deviation from target for near-upright states
        sin = s[:, 1]
        w_stable = ((1.0 + sin) / 2.0).clamp(0, 1)
        dist_loss = (w_stable * (s - self.s_target.expand(B, -1)).pow(2).sum(-1)).mean()

        return energy_loss + dist_loss

    def _stabilization_loss(self, s):
        """Stabilization loss for CartPole: keep pole upright, cart centered."""
        return (s[:, 2].pow(2).mean() + 0.1 * s[:, 0].pow(2).mean() +
                0.5 * s[:, 3].pow(2).mean())

    def get_action(self, s):
        """Deployment: single forward pass."""
        self.policy.eval()
        if isinstance(s, np.ndarray):
            s = torch.tensor(s, dtype=torch.float32, device=self.device)
        if s.dim() == 1:
            s = s.unsqueeze(0)
        with torch.no_grad():
            return self.policy(s).squeeze().cpu().item()
