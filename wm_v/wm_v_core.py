"""World Model + Value Network controller.  No k-selection needed.

World model (KAN): f(s,a) → s'   single-step, most accurate
Value network (MLP): V(s) → scalar   estimates future cumulative reward

MPC: argmax_a [R(s,a,s'_pred) + γ·V(s'_pred)]
Update: V(s) ← V(s) + α[r_real + γ·V(s'_real) - V(s)]  (TD(0))
"""
import torch, numpy as np, random
from collections import deque


class MLPValue(torch.nn.Module):
    """Tiny MLP: state_dim → 32 → 1. ~200 params."""
    def __init__(self, state_dim, hidden=32):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, 1),
        )
        # Initialize near-zero for stable TD learning start
        for p in self.net.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p, gain=0.1)

    def forward(self, s):
        return self.net(s)


class ReplayBuffer:
    """Small circular buffer for decorrelated TD updates."""
    def __init__(self, capacity=500):
        self.buf = deque(maxlen=capacity)

    def push(self, s, s_next, r):
        self.buf.append((s.detach().cpu(), s_next.detach().cpu(), r))

    def sample(self, n):
        batch = random.sample(self.buf, min(n, len(self.buf)))
        return (torch.cat([b[0] for b in batch], dim=0),
                torch.cat([b[1] for b in batch], dim=0),
                torch.tensor([b[2] for b in batch], dtype=torch.float32))


class WmVController:
    """MPC with world model + learned value function."""
    def __init__(self, world_model, state_dim, action_set, reward_fn,
                 gamma=0.95, lr=1e-3, buffer_size=500, device='cpu'):
        self.wm = world_model
        self.state_dim = state_dim
        self.action_set = action_set      # list of action tensors
        self.reward_fn = reward_fn        # (s, a, s'_pred) → float
        self.gamma = gamma
        self.device = device

        self.V = MLPValue(state_dim).to(device)
        self.opt = torch.optim.Adam(self.V.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size)

    def select_action(self, s_norm, explore_eps=0.0):
        """MPC: pick action with highest R + γ·V(s'_pred).

        Args:
            s_norm: (1, state_dim) normalized state
        Returns:
            best_a: action tensor, best_score: float
        """
        self.wm.eval()
        for p in self.wm.parameters():
            p.requires_grad = False

        best_a, best_score = self.action_set[0], -float('inf')

        for a in self.action_set:
            # World model predicts next state
            a_t = a.reshape(1, -1) if a.dim() == 1 else a
            x = torch.cat([s_norm, a_t], dim=-1)
            with torch.no_grad():
                s_pred = self.wm(x)[:, :self.state_dim]

            # Immediate reward + estimated future value
            r = self.reward_fn(s_norm, a, s_pred)
            with torch.no_grad():
                v = self.V(s_pred).item()
            score = r + self.gamma * v

            if score > best_score:
                best_score = score
                best_a = a

        if explore_eps > 0 and np.random.random() < explore_eps:
            best_a = random.choice(self.action_set)

        return best_a, best_score

    def update(self, s_norm, a_norm, s_next_norm, r_real):
        """TD(0) update: V(s) ← V(s) + α[r_real + γ·V(s') - V(s)]."""
        self.buffer.push(s_norm, s_next_norm, r_real)

        if len(self.buffer.buf) < 16:
            return 0.0

        s_batch, s_next_batch, r_batch = self.buffer.sample(32)
        s_batch = s_batch.to(self.device)
        s_next_batch = s_next_batch.to(self.device)
        r_batch = r_batch.to(self.device).unsqueeze(1)

        with torch.no_grad():
            target = r_batch + self.gamma * self.V(s_next_batch)

        self.V.train()
        self.opt.zero_grad()
        loss = torch.nn.functional.mse_loss(self.V(s_batch), target)
        loss.backward()
        self.opt.step()
        self.V.eval()

        return loss.item()
