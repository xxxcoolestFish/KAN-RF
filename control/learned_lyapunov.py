"""Learned Nonlinear Lyapunov Function via WM self-play.

Key insight: Riccati-based V(s) = (s-s*)^T P (s-s*) is a local quadratic
approximation. For tasks requiring swing-up (temporary divergence from target),
we need a global nonlinear V(s).

Approach:
  1. Use WM + long-horizon MPC to estimate V*(s) = cost-to-go from state s
  2. Train a lightweight KAN V_θ(s) ≈ V*(s) on these estimates
  3. Replace per-step cost in batch MPC with terminal V_θ(s_H)

This is fully automatic — no physics knowledge, no hand-crafted formulas.
"""
import torch, torch.nn as nn, numpy as np
from kanrf import KANLayer


class VNet(nn.Module):
    """Small KAN that learns V(s) = cost-to-go from state s to target."""

    def __init__(self, state_dim, hidden=16):
        super().__init__()
        self.layer1 = KANLayer(state_dim, hidden, grid_size=5, spline_order=3)
        self.layer2 = KANLayer(hidden, hidden, grid_size=5, spline_order=3)
        self.out = nn.Linear(hidden, 1)

    def forward(self, s):
        x = self.layer1(s)
        x = self.layer2(x)
        return self.out(x).squeeze(-1)  # (B,)


def generate_vstar_data(wm, state_dim, s_target, n_states=500,
                          horizon=12, n_shoot=200, device='cpu'):
    """Use long-horizon batch MPC to estimate V*(s) for random states.

    For each state, we need to know: "if we act optimally for H=12 steps,
    how close can we get to the target?" This is the label for V_θ.

    Returns:
        s_data: (n_states, state_dim) training states
        v_labels: (n_states,) V*(s) labels
    """
    print(f"  Generating V* data ({n_states} states, H={horizon})...")

    # Sample random states
    s_data = torch.randn(n_states, state_dim, device=device) * 0.5
    if state_dim >= 3:  # pendulum: cos, sin in [-1,1]
        s_data[:, :2].clamp_(-1, 1)

    v_labels = torch.zeros(n_states, device=device)

    for i in range(n_states):
        if i % 100 == 0:
            print(f"    {i}/{n_states}...")

        s = s_data[i]
        # Batch MPC: find best trajectory of length H
        B = n_shoot
        H = horizon
        seq = torch.FloatTensor(B, H).uniform_(-1, 1)
        s_cur = s.unsqueeze(0).expand(B, -1).clone()
        best_terminal_cost = float('inf')

        for t in range(H):
            a_t = seq[:, t:t+1]
            with torch.no_grad():
                s_cur = wm(torch.cat([s_cur, a_t], dim=-1))

        # Terminal cost: simple L2 distance (no physics needed)
        err = s_cur - s_target
        costs = err.pow(2).sum(dim=-1)  # (B,)
        best_idx = costs.argmin().item()
        v_labels[i] = costs[best_idx].item()

    print(f"    Done. V* range: [{v_labels.min().item():.3f}, {v_labels.max().item():.3f}]")
    return s_data, v_labels


def train_vnet(wm, state_dim, s_target, n_states=500, horizon=12,
               epochs=200, lr=1e-3, device='cpu'):
    """Generate V* data and train V_θ(s).

    Returns trained V_θ model.
    """
    s_data, v_labels = generate_vstar_data(
        wm, state_dim, s_target, n_states=n_states,
        horizon=horizon, n_shoot=200, device=device)

    # Normalize labels for stable training
    v_mean = v_labels.mean()
    v_std = v_labels.std().clamp(min=1e-6)
    v_norm = (v_labels - v_mean) / v_std

    vnet = VNet(state_dim).to(device)
    opt = torch.optim.Adam(vnet.parameters(), lr=lr)
    mse = nn.MSELoss()

    print(f"  Training V_theta ({epochs} epochs)...")
    for ep in range(1, epochs + 1):
        vnet.train(); opt.zero_grad()
        pred = vnet(s_data)
        loss = mse(pred, v_norm)
        loss.backward()
        opt.step()
        if ep % 50 == 0:
            print(f"    Epoch {ep:3d}  loss={loss.item():.4f}")

    # Store normalization stats for inference
    vnet.v_mean = v_mean.item()
    vnet.v_std = v_std.item()
    return vnet


class LearnedValueMPC:
    """Batch shooting MPC using learned V_θ as terminal cost.

    Instead of accumulating per-step costs over H steps:
        cost = Σ γ^t · ||s_t - s*||^2

    We use the terminal V_θ:
        cost = V_θ(s_H)

    V_θ already encodes "how hard is it to reach the target from s_H",
    learned from long-horizon MPC results.
    """

    def __init__(self, wm, vnet, state_dim, horizon=4, n_shoot=500, device='cpu'):
        self.wm = wm
        self.vnet = vnet
        self.state_dim = state_dim
        self.horizon = horizon
        self.n_shoot = n_shoot
        self.device = device
        wm.eval()
        vnet.eval()

    def get_action(self, s, s_target=None):
        if isinstance(s, np.ndarray):
            s = torch.tensor(s, dtype=torch.float32, device=self.device)
        if s_target is None:
            s_target = torch.zeros(self.state_dim, device=self.device)
        elif isinstance(s_target, np.ndarray):
            s_target = torch.tensor(s_target, dtype=torch.float32, device=self.device)

        B = self.n_shoot
        H = self.horizon
        seq = torch.FloatTensor(B, H).uniform_(-1, 1)
        s_cur = s.unsqueeze(0).expand(B, -1).clone()

        for t in range(H):
            a_t = seq[:, t:t+1]
            with torch.no_grad():
                s_cur = self.wm(torch.cat([s_cur, a_t], dim=-1))

        # Terminal cost via learned V_θ
        with torch.no_grad():
            v_pred = self.vnet(s_cur)  # normalized prediction
            v_raw = v_pred * self.vnet.v_std + self.vnet.v_mean

        best_idx = v_raw.argmin().item()
        return seq[best_idx, 0].item()
