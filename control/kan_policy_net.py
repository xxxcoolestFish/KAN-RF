"""KAN Policy Network — trained via frozen CWS-KAN world model gradient.

Architecture:
  Training:  s → [KAN π_θ] → a → [frozen KAN f] → s'_pred → loss(s'_pred, s*)
             gradient: ∂loss/∂θ = ∂loss/∂s' · ∂s'/∂a · ∂a/∂θ
                       ↑                 ↑              ↑
                    task loss     CWS Jacobian    policy params
                                  (cos≈0.98)

  Deployment: s → [KAN π_θ] → a   (pure forward, world model not used)

  Safety at deployment (optional):
    - Uncertainty gating: U(s, π(s)) > threshold → fallback
    - Lipschitz action clamping: |Δa| ≤ δ/L

Key advantages over decision_v3 (MLP policy):
  1. KAN policy preserves interpretability: edge functions, pruning, attribution
  2. Can extract symbolic control law from trained edges
  3. Continual learning: local B-spline updates without forgetting
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# KAN Policy Network
# ═══════════════════════════════════════════════════════════════════════════════

class KANPolicy(nn.Module):
    """KAN-based policy network — learnable B-spline edge functions.

    Input:  state (optionally concatenated with goal)
    Output: action ∈ [-1, 1] via tanh

    Args:
        state_dim: state dimension
        action_dim: action dimension
        hidden_dim: hidden layer width
        n_layers: number of hidden KAN layers (default 1)
        grid_size: B-spline grid fineness
        spline_order: B-spline degree
    """

    def __init__(self, state_dim=3, action_dim=1, hidden_dim=8,
                 n_layers=1, grid_size=5, spline_order=3):
        super().__init__()
        from kanrf import KANLayer

        self.state_dim = state_dim
        self.action_dim = action_dim
        in_dim = state_dim

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(KANLayer(in_dim, hidden_dim,
                                        grid_size=grid_size,
                                        spline_order=spline_order))
            in_dim = hidden_dim

        # Output layer: linear (no B-spline) + tanh
        self.output_layer = nn.Linear(in_dim, action_dim)

    def forward(self, s, return_activations=False):
        """s: (B, state_dim) → a: (B, action_dim) ∈ [-1, 1]"""
        x = s
        B_list_all, E_list_all = [], []

        for layer in self.layers:
            if return_activations:
                x, B_mat, E_mat = layer(x, return_activations=True)
                B_list_all.append(B_mat)
                E_list_all.append(E_mat)
            else:
                x = layer(x)

        a = torch.tanh(self.output_layer(x))

        if return_activations:
            return a, B_list_all, E_list_all
        return a

    def get_edge_functions(self):
        """Extract all edge function parameters for visualization/analysis.

        Returns list of dicts: [{in_dim, out_dim, spline_weight, base_weight, grid}]
        """
        edges = []
        for layer in self.layers:
            edges.append({
                'in_dim': layer.in_dim,
                'out_dim': layer.out_dim,
                'spline_weight': layer.spline_weight.detach().cpu().numpy(),
                'base_weight': layer.base_weight.detach().cpu().numpy(),
                'grid': layer.grid.detach().cpu().numpy(),
            })
        return edges

    def prune(self, threshold=0.01):
        """Prune edges with small spline weights. Returns number pruned."""
        n_pruned = 0
        for layer in self.layers:
            c = layer.spline_weight  # (out, in, n_basis)
            mask = c.abs().mean(dim=-1) > threshold  # (out, in)
            n_pruned += (~mask).sum().item()
            layer.spline_weight.data[~mask] = 0.0
        return n_pruned

    @torch.no_grad()
    def attribute(self, s):
        """Attribute action to input dimensions via additive decomposition.

        For the first KAN layer: output_j = Σ_i φ_{i,j}(x_i).
        We accumulate contributions through the output layer.

        Returns:
            contributions: (state_dim,) relative contribution of each input
        """
        if s.dim() == 1:
            s = s.unsqueeze(0)

        # Forward through first layer with activations
        x = s
        for layer in self.layers:
            x, B_mat, E_mat = layer(x, return_activations=True)

        # Contributions at first layer output
        # E_mat: (B, out_dim, in_dim) — energy per edge
        contributions = E_mat[0].mean(dim=0)  # (in_dim,) average per input
        contributions = contributions / (contributions.sum() + 1e-8)
        return contributions.cpu().numpy()


# ═══════════════════════════════════════════════════════════════════════════════
# KAN Policy Trainer (using frozen CWS-KAN world model)
# ═══════════════════════════════════════════════════════════════════════════════

class KANPolicyTrainer:
    """Train KAN policy using frozen CWS-KAN world model as gradient provider.

    The world model is FROZEN — it only provides the gradient ∂s'/∂a
    (CWS-trained, cos≈0.98). This gradient tells the policy which direction
    to adjust its output to improve task performance.

    Args:
        world_model: frozen CWS-KAN world model f(s,a) → s'
        policy: KANPolicy to train
        s_target: target state (e.g., Pendulum upright [0, 1, 0])
        lr, lambda_ctrl, clip_grad: standard training params
        device: torch device
    """

    def __init__(self, world_model, policy, s_target,
                 lr=1e-3, lambda_ctrl=0.01, clip_grad=10.0,
                 device='cpu'):
        self.wm = world_model
        self.policy = policy.to(device)
        self.s_target = s_target.to(device)
        self.device = device
        self.lambda_ctrl = lambda_ctrl
        self.clip_grad = clip_grad

        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
        self.loss_history = []

        # Freeze world model
        self.wm.eval()
        for p in self.wm.parameters():
            p.requires_grad = False

    def _compute_energy(self, s):
        """Pendulum energy from normalized state [cos, sin, thd/8]."""
        thd = s[:, 2] * 8.0
        sin = s[:, 1]
        return 0.5 * thd.pow(2) + 10.0 * sin

    def train_step(self, s_batch):
        """Single training step — energy-guided loss through frozen world model."""
        B = s_batch.shape[0]
        self.policy.train()
        self.optimizer.zero_grad()

        # Forward: state → policy → action → frozen WM → predicted next state
        a = self.policy(s_batch)
        wm_input = torch.cat([s_batch, a], dim=-1)
        s_pred = self.wm(wm_input)

        # Energy-guided loss
        E_current = self._compute_energy(s_batch)
        E_pred = self._compute_energy(s_pred)
        E_des = 10.0
        energy_deficit = E_des - E_current
        energy_gain = (E_pred - E_current) * torch.sign(energy_deficit)

        sin = s_batch[:, 1:2]
        w_swing = ((1.0 - sin) / 2.0).clamp(0.0, 1.0)
        w_stable = ((1.0 + sin) / 2.0).clamp(0.0, 1.0)

        energy_loss = -energy_gain.mean()
        dist_loss = (w_stable * (s_pred - self.s_target.expand(B, -1)).pow(2).sum(dim=-1, keepdim=True)).mean()
        pred_loss = energy_loss + dist_loss
        ctrl_loss = a.pow(2).mean()
        total_loss = pred_loss + self.lambda_ctrl * ctrl_loss

        # Backward: gradient flows WM → action → policy
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.clip_grad)
        self.optimizer.step()

        ld = {
            'total': total_loss.item(),
            'pred': pred_loss.item(),
            'energy': energy_loss.item(),
            'dist': dist_loss.item(),
            'ctrl': ctrl_loss.item(),
        }
        self.loss_history.append(ld)
        return ld

    def train_epoch(self, s_dataset, batch_size=256, n_batches=None):
        N = s_dataset.shape[0]
        if n_batches is None:
            n_batches = max(1, N // batch_size)
        losses = []
        for _ in range(n_batches):
            idx = torch.randint(0, N, (batch_size,), device=self.device)
            losses.append(self.train_step(s_dataset[idx]))
        return {k: np.mean([l[k] for l in losses]) for k in losses[0]}

    @torch.no_grad()
    def get_action(self, s):
        """Deployment: pure forward pass through KAN policy."""
        self.policy.eval()
        if isinstance(s, np.ndarray):
            s = torch.tensor(s, dtype=torch.float32, device=self.device)
        if s.dim() == 1:
            s = s.unsqueeze(0)
        a = self.policy(s).squeeze().cpu()
        return a.item() if a.numel() == 1 else a.numpy()


# ═══════════════════════════════════════════════════════════════════════════════
# Safety Wrapper: uncertainty-gated fallback + Lipschitz clamping
# ═══════════════════════════════════════════════════════════════════════════════

class SafeKANPolicy:
    """Wraps trained KAN policy with safety constraints from world model.

    - Uncertainty gating: if U(s, a) > threshold, fall back to safe controller
    - Lipschitz clamping: |a_t - a_{t-1}| ≤ δ/L
    """

    def __init__(self, policy, trainer, knowledge,
                 eta_safe=0.30, device='cpu'):
        self.policy = policy
        self.trainer = trainer
        self.kk = knowledge
        self.eta_safe = eta_safe
        self.device = device
        self.prev_action = 0.0

    def get_action(self, s_norm):
        """Safe action: KAN policy with uncertainty fallback."""
        if isinstance(s_norm, np.ndarray):
            s_t = torch.tensor(s_norm, dtype=torch.float32, device=self.device)
        else:
            s_t = s_norm

        # Check uncertainty
        with torch.no_grad():
            rho = self.kk.uncertainty(s_t.unsqueeze(0)).item()
            U = 1.0 - rho

        if U > self.eta_safe:
            # Fallback to energy heuristic
            sin = s_norm[1]
            thd = s_norm[2] * 8.0
            E = 0.5 * thd**2 + 10.0 * sin
            a = np.clip(1.5 * (E - 10.0) * thd / 10.0, -1.0, 1.0)
            return a, {'method': 'fallback', 'U': U}

        # KAN policy
        a = self.trainer.get_action(s_norm)

        # Lipschitz clamping
        delta_max = self.kk.trust_region_radius(0.1)
        a = self.prev_action + np.clip(a - self.prev_action, -delta_max, delta_max)
        self.prev_action = a

        return a, {'method': 'kan_policy', 'U': U}
