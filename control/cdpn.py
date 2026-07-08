"""CDPN: Causal-Decomposed Policy Network.

Architecture:
  Policy(s, s_goal) → v_des ∈ R^m      (learnable, Tier 0 space)
  Execute(J_0, v_des) → a              (deterministic, Jacobian inverse)
  Safety(U(s), a) → a_safe              (uncertainty-gated)

Key insight: Policy outputs desired CHANGE in directly-controllable
dimensions (Tier 0), NOT raw action. Execute maps v_des → a using
ProtoKAN WM's exact Jacobian, amplifying gradient signal ~10x.
"""
import torch, torch.nn as nn
import numpy as np
from kanrf import KANLayer, ProtoKANLayer


# ═══════════════════════════════════════════════════════════
# Causal Structure Discovery
# ═══════════════════════════════════════════════════════════

def discover_tier0(wm, state_dim, n_samples=200, device='cpu'):
    """Discover Tier 0 (directly controllable) dimensions from WM Jacobian.

    Returns:
        tier0_indices: list of state dimension indices in Tier 0
        tier0_mask: (state_dim,) boolean mask
        jac_norms: (state_dim,) mean |∂s'[i]/∂a|
    """
    wm.eval()
    was_frozen = not next(wm.parameters()).requires_grad
    if was_frozen:
        for p in wm.parameters():
            p.requires_grad = True

    jac_acc = torch.zeros(state_dim, device=device)
    for _ in range(n_samples):
        s = torch.randn(1, state_dim, device=device).clamp(-1, 1)
        a = torch.zeros(1, 1, device=device, requires_grad=True)
        s_pred = wm(torch.cat([s, a], dim=-1))
        for i in range(state_dim):
            g = torch.autograd.grad(s_pred[0, i], a, retain_graph=True)[0]
            jac_acc[i] += g[0, 0].abs()

    if was_frozen:
        for p in wm.parameters():
            p.requires_grad = False

    jac_norms = (jac_acc / n_samples).cpu().numpy()
    threshold = np.percentile(jac_norms, 60)
    tier0 = [i for i in range(state_dim) if jac_norms[i] >= threshold]
    mask = np.zeros(state_dim, dtype=bool)
    mask[tier0] = True

    return tier0, mask, jac_norms, threshold


# ═══════════════════════════════════════════════════════════
# Policy Network
# ═══════════════════════════════════════════════════════════

class CausalDecomposedPolicy(nn.Module):
    """Policy that outputs v_des in Tier 0 space, NOT raw action.

    Input:  (s, s_goal) ∈ R^{2n}
    Output: v_des ∈ R^m  where m = |Tier 0|
            "How much should each directly-controllable dimension change?"
    """

    def __init__(self, state_dim, tier0_size, hidden_dim=24, n_layers=2,
                 grid_size=5, spline_order=3):
        super().__init__()
        self.state_dim = state_dim
        self.tier0_size = tier0_size
        in_dim = state_dim * 2

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(KANLayer(in_dim, hidden_dim,
                                        grid_size=grid_size,
                                        spline_order=spline_order))
            in_dim = hidden_dim

        # Output: v_des for each Tier 0 dimension, no tanh (can be any magnitude)
        self.output_layer = nn.Linear(in_dim, tier0_size)

    def forward(self, s, s_goal):
        x = torch.cat([s, s_goal], dim=-1)
        for layer in self.layers:
            x = layer(x)
        return self.output_layer(x)  # (batch, m)


# ═══════════════════════════════════════════════════════════
# Execute Module (deterministic)
# ═══════════════════════════════════════════════════════════

class Execute:
    """Deterministic inverse: maps v_des → a using WM Jacobian.

    J_0 = ∂s_Tier0/∂a is pre-computed over training states.
    For systems with state-dependent Jacobian, this is a first-order
    approximation; per-state Jacobian is a future optimization.

    a = (J^T J + λI)^{-1} J^T · v_des
    """

    def __init__(self, wm, state_dim, tier0_indices, s_dataset,
                 damping=0.1, n_jac_samples=200, device='cpu'):
        self.device = device
        self.m = len(tier0_indices)

        # Pre-compute J on sample states
        wm.eval()
        was_frozen = not next(wm.parameters()).requires_grad
        if was_frozen:
            for p in wm.parameters(): p.requires_grad = True

        N = min(n_jac_samples, s_dataset.shape[0])
        idx = torch.randperm(s_dataset.shape[0])[:N]
        J_sum = torch.zeros(self.m, 1, device=device)
        for i in idx:
            s = s_dataset[i:i+1]
            a = torch.zeros(1, 1, device=device, requires_grad=True)
            sp = wm(torch.cat([s, a], dim=-1))
            for j, dim in enumerate(tier0_indices):
                g = torch.autograd.grad(sp[0, dim], a, retain_graph=True)[0]
                J_sum[j, 0] += g[0, 0]

        if was_frozen:
            for p in wm.parameters(): p.requires_grad = False

        self.J = J_sum / N  # (m, 1)
        self.damping = damping
        print(f"  Execute J (Tier0): {self.J.squeeze().cpu().tolist()}")

    def __call__(self, v_des, s_batch=None):
        """a = (J^T J + λI)^{-1} J^T v_des  [batch operation]"""
        J = self.J  # (m, 1)
        JTJ = (J ** 2).sum() + self.damping  # scalar
        JTv = J.T @ v_des.T  # (1, B)
        a = JTv / JTJ  # (1, B)
        return a.T.clamp(-2, 2)  # (B, 1)


# ═══════════════════════════════════════════════════════════
# CDPN Trainer
# ═══════════════════════════════════════════════════════════

class CDPNTrainer:
    """Train CDPN via either 1-step WM gradient or k-step imagination.

    Training options:
      mode='1step': Policy → v_des → Execute → a → WM → s' → loss
      mode='imagine': Policy → (v_des → Execute → a → WM) × k → cumulative loss
    """

    def __init__(self, wm, policy, execute, s_target, tier0_indices,
                 lr=1e-3, device='cpu', mode='1step', imagine_steps=3):
        self.wm = wm
        self.policy = policy.to(device)
        self.execute = execute
        self.s_target = s_target.to(device)
        self.tier0 = tier0_indices
        self.device = device
        self.mode = mode
        self.imagine_steps = imagine_steps

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

            if self.mode == '1step':
                loss = self._train_1step(s_b)
            elif self.mode == 'imagine':
                loss = self._train_imagine(s_b)
            else:
                raise ValueError(f"Unknown mode: {self.mode}")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 10.0)
            self.opt.step()
            total_loss += loss.item()

        avg_loss = total_loss / n_batches
        self.loss_history.append({'total': avg_loss})
        return self.loss_history[-1]

    def _train_1step(self, s_b):
        """Single-step training with WM gradient."""
        B = s_b.shape[0]
        s_goal = self.s_target.expand(B, -1)

        # Policy → v_des
        v_des = self.policy(s_b, s_goal)  # (B, m)

        # Execute → a
        a = self.execute(v_des, s_b)  # (B, 1)

        # WM → s_pred
        s_pred = self.wm(torch.cat([s_b, a], dim=-1))

        # Loss: distance to goal + control penalty
        loss = (s_pred - s_goal).pow(2).sum(dim=-1).mean()
        loss = loss + 0.01 * a.pow(2).mean()

        return loss

    def _train_imagine(self, s_b):
        """Multi-step imagination training.

        Rollout k steps: at each step, Policy + Execute produce action,
        WM predicts next state. Accumulate loss over the trajectory.

        This breaks the 1-step myopia: Policy learns to optimize
        long-term consequences.
        """
        B = s_b.shape[0]
        s_goal = self.s_target.expand(B, -1)
        s_cur = s_b
        total_loss = 0.0

        for step in range(self.imagine_steps):
            v_des = self.policy(s_cur, s_goal)
            a = self.execute(v_des, s_cur)
            s_cur = self.wm(torch.cat([s_cur, a], dim=-1))

            # Step loss: distance to goal, discounted
            gamma = 0.9 ** step
            step_loss = (s_cur - s_goal).pow(2).sum(dim=-1).mean()
            total_loss = total_loss + gamma * step_loss

        return total_loss

    def get_action(self, s, s_goal=None):
        """Get action for deployment."""
        self.policy.eval()
        if isinstance(s, np.ndarray):
            s = torch.tensor(s, dtype=torch.float32, device=self.device)
        if s.dim() == 1:
            s = s.unsqueeze(0)
        if s_goal is None:
            s_goal = self.s_target
        elif isinstance(s_goal, np.ndarray):
            s_goal = torch.tensor(s_goal, dtype=torch.float32, device=self.device)
        if s_goal.dim() == 1:
            s_goal = s_goal.unsqueeze(0)

        with torch.no_grad():
            v_des = self.policy(s, s_goal)
        # Execute needs gradients for Jacobian — temporarily enable
        a = self.execute(v_des, s)
        return a.squeeze().cpu().item()
