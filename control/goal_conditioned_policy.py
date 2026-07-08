"""Goal-conditioned Policy: input = (s, s_goal), output = a.

Key change from V1.0: Policy receives explicit target state, enabling
generalization across goals and a richer training signal.
"""
import torch, torch.nn as nn, numpy as np
from kanrf import KANLayer


class GoalConditionedPolicy(nn.Module):
    """KAN Policy that takes (s, s_goal) as input.

    Architecture: (s || s_goal) → KANLayer → KANLayer → Linear → tanh → a
    """

    def __init__(self, state_dim=3, hidden_dim=12, n_layers=2,
                 grid_size=5, spline_order=3):
        super().__init__()
        self.state_dim = state_dim
        in_dim = state_dim * 2  # s + s_goal

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(KANLayer(in_dim, hidden_dim,
                                        grid_size=grid_size,
                                        spline_order=spline_order))
            in_dim = hidden_dim

        self.output_layer = nn.Linear(in_dim, 1)

    def forward(self, s, s_goal):
        """s: (B, n), s_goal: (B, n) → a: (B, 1) ∈ [-1, 1]"""
        x = torch.cat([s, s_goal], dim=-1)
        for layer in self.layers:
            x = layer(x)
        return torch.tanh(self.output_layer(x))


class GoalConditionedTrainer:
    """Train goal-conditioned Policy via frozen WM gradient.

    During training, s_goal is sampled from a distribution around the
    standard target, teaching the Policy to handle diverse goal conditions.
    """

    def __init__(self, wm, policy, s_target_standard, s_dataset,
                 lr=1e-3, goal_noise=0.05, device='cpu'):
        self.wm = wm
        self.policy = policy.to(device)
        self.s_target_std = s_target_standard.to(device)
        self.goal_noise = goal_noise
        self.device = device
        self.state_dim = s_target_standard.shape[1]

        wm.eval()
        for p in wm.parameters():
            p.requires_grad = False

        self.opt = torch.optim.Adam(policy.parameters(), lr=lr)
        self.loss_history = []

    def _sample_goal(self, batch_size):
        """Sample s_goal around standard target with noise."""
        base = self.s_target_std.expand(batch_size, -1)
        noise = torch.randn(batch_size, self.state_dim, device=self.device)
        noise = noise * self.goal_noise
        return (base + noise).clamp(-1, 1)

    def train_epoch(self, s_dataset, batch_size=256):
        N = s_dataset.shape[0]
        n_batches = max(1, N // batch_size)
        total_loss = 0.0

        for _ in range(n_batches):
            idx = torch.randint(0, N, (batch_size,), device=self.device)
            s_b = s_dataset[idx]
            s_goal = self._sample_goal(batch_size)

            self.policy.train()
            self.opt.zero_grad()

            a = self.policy(s_b, s_goal)
            s_pred = self.wm(torch.cat([s_b, a], dim=-1))

            # Loss: distance between predicted next state and goal
            if self.state_dim == 3:  # Pendulum
                thd = s_b[:, 2] * 8.0
                Ec = 0.5 * thd.pow(2) + 10.0 * s_b[:, 1]
                thdp = s_pred[:, 2] * 8.0
                Ep = 0.5 * thdp.pow(2) + 10.0 * s_pred[:, 1]
                deficit = (10.0 - Ec).detach()
                egain = (Ep - Ec) * torch.sign(deficit)
                sin = s_b[:, 1]
                ws = ((1.0 + sin) / 2.0).clamp(0, 1)
                loss = (-egain.mean() +
                        (ws * (s_pred - s_goal).pow(2).sum(-1)).mean() +
                        0.01 * a.pow(2).mean())
            else:  # CartPole / generic
                loss = ((s_pred - s_goal).pow(2).sum(-1)).mean()
                loss = loss + 0.01 * a.pow(2).mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 10.0)
            self.opt.step()
            total_loss += loss.item()

        avg_loss = total_loss / n_batches
        self.loss_history.append({'total': avg_loss})
        return self.loss_history[-1]

    def get_action(self, s, s_goal=None):
        """Get action for given state and goal. If s_goal is None, use standard target."""
        self.policy.eval()
        if isinstance(s, np.ndarray):
            s = torch.tensor(s, dtype=torch.float32, device=self.device)
        if s.dim() == 1:
            s = s.unsqueeze(0)
        if s_goal is None:
            s_goal = self.s_target_std.expand(s.shape[0], -1)
        elif isinstance(s_goal, np.ndarray):
            s_goal = torch.tensor(s_goal, dtype=torch.float32, device=self.device)
            if s_goal.dim() == 1:
                s_goal = s_goal.unsqueeze(0)
        with torch.no_grad():
            return self.policy(s, s_goal).squeeze().cpu().item()
