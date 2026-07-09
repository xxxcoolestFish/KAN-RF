"""Structure-Aware Policy: Policy(s, s_goal, P_diag) → a.

Key insight: Policy receives system structure (tier hierarchy encoded in P_diag)
as explicit input. When physics changes, P_diag changes → Policy auto-adapts
without retraining. No WM gradients needed — trained via multi-physics MPC
distillation.
"""
import torch, torch.nn as nn, numpy as np
from kanrf import KANLayer


class StructureAwarePolicy(nn.Module):
    """Policy that receives system structure as input.

    Input: (s, s_goal, P_diag) where P_diag encodes per-dim importance
           from Lyapunov synthesis (higher = more important / goal dimension).
    Output: a ∈ [-1, 1]
    """

    def __init__(self, state_dim, hidden=16, n_layers=2):
        super().__init__()
        # s (n) + s_goal (n) + P_diag (n) = 3n
        in_dim = state_dim * 3
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(KANLayer(in_dim, hidden, grid_size=5, spline_order=3))
            in_dim = hidden
        self.out = nn.Linear(in_dim, 1)

    def forward(self, s, s_goal, P_diag):
        x = torch.cat([s, s_goal, P_diag], dim=-1)
        for layer in self.layers:
            x = layer(x)
        return torch.tanh(self.out(x))


class StructureAwareTrainer:
    """Train Structure-Aware Policy via multi-physics MPC distillation.

    1. For each physics parameter (e.g. gravity g), train a WM
    2. Synthesize Lyapunov P from each WM
    3. Use MPC (batch shooting) to generate (s, P_diag, a_optimal) pairs
    4. Train Policy on all pairs — learns to use P_diag as contextual signal

    After training: Policy generalizes to new physics parameters without
    retraining — just compute new P_diag from adapted WM and feed to Policy.
    """

    def __init__(self, state_dim, s_target, hidden=16, n_layers=2,
                 lr=1e-3, device='cpu'):
        self.policy = StructureAwarePolicy(state_dim, hidden, n_layers).to(device)
        self.state_dim = state_dim
        self.s_target = s_target.to(device)
        self.device = device
        self.opt = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.demo_s = []
        self.demo_sg = []
        self.demo_P = []
        self.demo_a = []

    def add_physics(self, wm, s_dataset, P, n_demos=500):
        """Generate MPC demos for one physics configuration."""
        from control.gradient_mpc import GradientMPC
        P_diag = P.diag().unsqueeze(0)  # (1, n)

        mpc = GradientMPC(wm, self.state_dim, P=P, horizon=3, n_shoot=500,
                           mode='shoot', device=self.device)
        N = min(n_demos, s_dataset.shape[0])
        idx = torch.randperm(s_dataset.shape[0])[:N]
        s_sel = s_dataset[idx].to(self.device)

        for i in range(N):
            a_opt, _ = mpc._optimize_shoot(s_sel[i], self.s_target.squeeze(0))
            self.demo_s.append(s_sel[i].cpu())
            self.demo_sg.append(self.s_target.squeeze(0).cpu())
            self.demo_P.append(P_diag.squeeze(0).cpu())
            self.demo_a.append(a_opt)

    def train_epochs(self, epochs=200, batch_size=256):
        """Train Policy on all accumulated demos."""
        if len(self.demo_s) == 0:
            return

        S = torch.stack(self.demo_s).to(self.device)
        SG = torch.stack(self.demo_sg).to(self.device)
        PD = torch.stack(self.demo_P).to(self.device)
        A = torch.tensor(self.demo_a, dtype=torch.float32).unsqueeze(1).to(self.device)

        D = len(S)
        for ep in range(1, epochs + 1):
            total_loss = 0.0
            n_batches = max(1, D // batch_size)
            for _ in range(n_batches):
                idx = torch.randint(0, D, (batch_size,), device=self.device)
                self.policy.train(); self.opt.zero_grad()
                a_pred = self.policy(S[idx], SG[idx], PD[idx])
                loss = (a_pred - A[idx]).pow(2).mean()
                loss.backward(); self.opt.step()
                total_loss += loss.item()
            if ep % 50 == 0:
                print(f"  Epoch {ep:3d}  loss={total_loss/n_batches:.4f}")

    def get_action(self, s, P_diag, s_goal=None):
        """Deployment: Policy(s, s_goal, P_diag) → a."""
        self.policy.eval()
        if isinstance(s, np.ndarray):
            s = torch.tensor(s, dtype=torch.float32, device=self.device)
        if s.dim() == 1: s = s.unsqueeze(0)
        if s_goal is None:
            s_goal = self.s_target.expand(s.shape[0], -1)
        elif isinstance(s_goal, np.ndarray):
            s_goal = torch.tensor(s_goal, dtype=torch.float32, device=self.device)
            if s_goal.dim() == 1: s_goal = s_goal.unsqueeze(0)
        if isinstance(P_diag, np.ndarray):
            P_diag = torch.tensor(P_diag, dtype=torch.float32, device=self.device)
        if P_diag.dim() == 1: P_diag = P_diag.unsqueeze(0)
        with torch.no_grad():
            return self.policy(s, s_goal, P_diag).squeeze().cpu().item()
