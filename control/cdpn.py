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
    """Policy that outputs NORMALIZED v_des_norm in [-1, 1]^m (Tier 0 space).

    Input:  (s, s_goal) in R^{2n}
    Output: v_des_norm in [-1, 1]^m  where m = |Tier 0|

    Supports KAN (default) and MLP layers.
    MLP mode avoids KAN gradient vanishing issues.
    """

    def __init__(self, state_dim, tier0_size, hidden_dim=24, n_layers=2,
                 grid_size=5, spline_order=3, use_tanh=True, use_mlp=False):
        super().__init__()
        self.state_dim = state_dim
        self.tier0_size = tier0_size
        self.use_tanh = use_tanh
        self.use_mlp = use_mlp
        in_dim = state_dim * 2

        if use_mlp:
            mlp = []
            for _ in range(n_layers):
                mlp.extend([nn.Linear(in_dim, hidden_dim), nn.Tanh()])
                in_dim = hidden_dim
            self.mlp = nn.Sequential(*mlp)
        else:
            self.layers = nn.ModuleList()
            for _ in range(n_layers):
                self.layers.append(KANLayer(in_dim, hidden_dim,
                                            grid_size=grid_size,
                                            spline_order=spline_order))
                in_dim = hidden_dim
        self.output_layer = nn.Linear(in_dim, tier0_size)

    def forward(self, s, s_goal):
        x = torch.cat([s, s_goal], dim=-1)
        if self.use_mlp:
            x = self.mlp(x)
        else:
            for layer in self.layers:
                x = layer(x)
        v = self.output_layer(x)
        if self.use_tanh:
            v = torch.tanh(v)
        return v  # (batch, m)

    def get_strategy(self, s, s_goal=None):
        """Alias for forward: emphasizes this is a STRATEGY instruction."""
        if s_goal is not None:
            return self.forward(s, s_goal)
        return self.forward(s, s)

class Execute:
    """Deterministic inverse: maps v_des → a using WM Jacobian.

    J_0 = ∂s_Tier0/∂a is pre-computed over training states.
    For systems with state-dependent Jacobian, this is a first-order
    approximation; per-state Jacobian is a future optimization.

    a = (J^T J + λI)^{-1} J^T · v_des
    """

    def __init__(self, wm, state_dim, tier0_indices, s_dataset,
                 damping=0.1, n_jac_samples=200, device='cpu',
                 bridge=None):
        self.device = device
        self.state_dim = state_dim
        self.tier0_indices = tier0_indices
        self.m = len(tier0_indices)
        self.damping = damping
        self.bridge = bridge  # CausalBridge instance (optional)
        self._wm = wm
        self._s_dataset = s_dataset
        self.compute_jacobian(wm, s_dataset, n_jac_samples)

    def compute_jacobian(self, wm, s_dataset=None, n_jac_samples=200):
        if s_dataset is None:
            s_dataset = self._s_dataset
        s_dataset = s_dataset.to(self.device)
        N = min(n_jac_samples, s_dataset.shape[0])
        idx = torch.randperm(s_dataset.shape[0])[:N]
        wm.eval()
        was_frozen = not next(wm.parameters()).requires_grad
        if was_frozen:
            for p in wm.parameters(): p.requires_grad = True
        J_sum = torch.zeros(self.m, 1, device=self.device)
        for i in idx:
            s = s_dataset[i:i+1]
            a = torch.zeros(1, 1, device=self.device, requires_grad=True)
            sp = wm(torch.cat([s, a], dim=-1))
            for j, dim in enumerate(self.tier0_indices):
                g = torch.autograd.grad(sp[0, dim], a, retain_graph=True)[0]
                J_sum[j, 0] += g[0, 0]
        if was_frozen:
            for p in wm.parameters(): p.requires_grad = False
        self.J = J_sum / N
        self.damping = self.damping
        print(f"  Execute J (Tier0): {self.J.squeeze().cpu().tolist()}")

    def update(self, wm=None, s_dataset=None, n_jac_samples=200):
        if wm is not None: self._wm = wm
        if s_dataset is not None: self._s_dataset = s_dataset
        self.compute_jacobian(self._wm, self._s_dataset, n_jac_samples)
        if self.bridge is not None:
            self.bridge.update(self._wm, self._s_dataset)

    def __call__(self, v_des, s_batch=None):
        """a = (J^T J + λI)^{-1} J^T v_des  [batch operation]"""
        J = self.J
        v_des = v_des.to(J.dtype)  # (m, 1)
        j_sq = (J ** 2).sum(); JTJ = j_sq + self.damping * j_sq + 1e-8  # adaptive
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


# ======================================================================
# CDPN v2 additions
# ======================================================================

def estimate_gravity_from_wm(wm, s_dataset, n_rollout=1, device='cpu'):
    """Estimate G from WM a=0 rollout (1 step, robust median filtering)."""
    wm.eval()
    with torch.no_grad():
        s_cur = s_dataset[:min(1000, len(s_dataset))].clone().to(device)
        sn = wm(torch.cat([s_cur, torch.zeros(s_cur.shape[0], 1, device=device)], dim=-1))[:, :3]
        n1 = s_cur[:, 2] * 8.0; n2 = sn[:, 2] * 8.0
        s1 = s_cur[:, 1]; s2 = sn[:, 1]
        num = n2.pow(2) - n1.pow(2); den = 2.0 * (s1 - s2)
        mask = den.abs() > 0.01
        if mask.any():
            g = num[mask] / den[mask]
            g_clean = g[(g > 1) & (g < 50)]
            if len(g_clean) > 0:
                return float(g_clean.median().item())
    return 10.0



def estimate_a_fit_from_wm(wm, s_dataset, dt=0.05, device='cpu'):
    """Estimate gravity coefficient a_fit from WM a=0 rollout."""
    wm.eval()
    with torch.no_grad():
        s_cur = s_dataset[:min(1000, len(s_dataset))].clone().to(device)
        sn = wm(torch.cat([s_cur, torch.zeros(s_cur.shape[0], 1, device=device)], dim=-1))[:, :3]
        dthd = (sn[:, 2] - s_cur[:, 2]) * 8.0
        sin_th = s_cur[:, 1]
        den = sin_th * dt
        mask = den.abs() > 0.001
        if mask.any():
            a = dthd[mask] / den[mask]
            a_clean = a[(a > 1) & (a < 50)]
            if len(a_clean) > 0:
                return float(a_clean.median().item())
    return 15.0

class CausalBridge:
    """Bridge between cognitive (WM) and decision (Policy) modules.
    
    Extracts env-invariant quantities from WM:
    1. max_delta: for each Tier 0 dim, max state change per unit v_des
    2. G_est: estimated gravity (Pendulum energy loss)
    
    All quantities are HOT-UPDATABLE via update() after WM adaptation.
    """
    
    def __init__(self, wm, state_dim, tier0_indices, s_dataset,
                 device='cpu', n_samples=500, g_true=None):
        self.device = device
        self.state_dim = state_dim
        self.tier0 = tier0_indices
        self.m = len(tier0_indices)
        self._g_true = g_true
        self.a_fit = estimate_a_fit_from_wm(wm, s_dataset, device=self.device)
        print(f"  [CausalBridge] a_fit={self.a_fit:.2f}")
        self.compute(wm, s_dataset, n_samples)
    
    def compute(self, wm, s_dataset, n_samples=500):
        wm.eval()
        s_dataset = s_dataset.to(self.device)
        N = min(n_samples, s_dataset.shape[0])
        idx = torch.randperm(s_dataset.shape[0])[:N]
        s_samples = s_dataset[idx]
        
        was_frozen = not next(wm.parameters()).requires_grad
        if was_frozen:
            for p in wm.parameters(): p.requires_grad = True
        
        jac_vals = {j: [] for j in range(self.m)}
        for i in range(N):
            s = s_samples[i:i+1]
            a = torch.zeros(1, 1, device=self.device, requires_grad=True)
            sp = wm(torch.cat([s, a], dim=-1))
            for j, dim in enumerate(self.tier0):
                g = torch.autograd.grad(sp[0, dim], a, retain_graph=True)[0]
                jac_vals[j].append(g[0, 0].abs().item())
        
        if was_frozen:
            for p in wm.parameters(): p.requires_grad = False
        
        self.max_delta = torch.tensor(
            [np.percentile(jac_vals[j], 90) for j in range(self.m)],
            dtype=torch.float32, device=self.device)
        self.controllability = self.max_delta / (self.max_delta.sum() + 1e-8)
        if self._g_true is not None:
            self.G_est = float(self._g_true)
        else:
            self.G_est = estimate_gravity_from_wm(wm, s_dataset, device=self.device)
        print(f"  [CausalBridge] max_delta={self.max_delta.cpu().tolist()}, G_est={self.G_est:.2f}")

        # --- 4. Lyapunov P matrix from WM dynamics ---
        self.P = None
        try:
            from control.lyapunov_bptt import synthesize_lyapunov
            from experiments.baseline_sweep import S_TARGET as st
            P, A, B, tiers = synthesize_lyapunov(
                wm, s_dataset, st, self.state_dim,
                horizon=10, r_weight=0.1, n_samples=200,
                q_goal=10.0, q_means=1.0, device=self.device)
            self.P = P
            print(f"  [CausalBridge] P matrix computed, tr(P)={P.trace().item():.2f}")
        except Exception as e:
            print(f"  [CausalBridge] P computation skipped: {e}")

        return self
    
    def update(self, wm, s_dataset=None, n_samples=500):
        self.a_fit = estimate_a_fit_from_wm(wm, s_dataset, device=self.device)
        print(f"  [CausalBridge] a_fit updated: {self.a_fit:.2f}")
        self.compute(wm, s_dataset, n_samples)


class AbstractPendulumDynamics:
    """Smooth analytic Pendulum dynamics for policy training (no WM gradient).

    Loss uses squared energy error + smooth stabilization transition.
    """

    DT = 0.05

    def __init__(self, bridge, s_target=None):
        self.bridge = bridge
        self.s_target = s_target

    def predict_next(self, s, v_des_norm):
        """Physics-aware abstract dynamics with gravity term.

        thd_next = thd + a_fit * sin(th) * dt   (gravity)
                     + v_des_norm * max_delta * 8   (control)

        Key: a_fit comes from CausalBridge (estimated from WM).
        Gradient still PURELY analytic, no WM involved.
        """
        cos_th, sin_th = s[:, 0], s[:, 1]
        thd = s[:, 2] * 8.0
        th = torch.atan2(sin_th, cos_th)
        if v_des_norm.dim() == 1:
            v_des_norm = v_des_norm.unsqueeze(1)
        # Gravity acceleration: a_fit * sin(th) * dt
        # Control acceleration: v_des_norm * max_delta * 8.0
        a_fit = self.bridge.a_fit
        gravity = a_fit * sin_th * self.DT  # sin(th) not sin_th? fix: sin(th) = sin_th ✓
        control = v_des_norm[:, 0] * self.bridge.max_delta[0] * 8.0
        thd_next = thd + gravity + control
        th_next = th + thd_next * self.DT
        return torch.stack([torch.cos(th_next), torch.sin(th_next), thd_next / 8.0], dim=-1)

    def compute_loss(self, s, v_des_norm):
        """Lyapunov-based loss using P matrix from WM.

        loss = (s_pred - s_target)^T * P * (s_pred - s_target)
               + lambda * ||v_des_norm||^2

        P is derived from WM Jacobian via Riccati equation.
        No sign(), no sigmoid, no G_est for loss computation.
        """
        s_pred = self.predict_next(s, v_des_norm)
        err = s_pred - self.s_target

        if hasattr(self.bridge, 'P') and self.bridge.P is not None:
            P = self.bridge.P.to(s_pred.device)
            loss = (err @ P @ err.T).diag().mean()
        else:
            loss = err.pow(2).sum(dim=-1).mean()

        ctrl = v_des_norm.pow(2).mean()
        total = loss + 0.01 * ctrl
        return total, {'total': total.item(), 'loss': loss.item(),
                       'ctrl': ctrl.item()}
class AbstractCartPoleDynamics:
    """Abstract CartPole dynamics for policy training (no WM gradient)."""
    DT = 0.02
    X_S, XD_S, TH_S, THD_S = 2.5, 3.0, 0.3, 3.0
    
    def __init__(self, bridge):
        self.bridge = bridge
    
    def predict_next(self, s, v_des_norm):
        x, xd = s[:, 0] * self.X_S, s[:, 1] * self.XD_S
        th, thd = s[:, 2] * self.TH_S, s[:, 3] * self.THD_S
        delta = v_des_norm[:, :self.bridge.m] * self.bridge.max_delta.unsqueeze(0)
        thd_next, xd_next = thd.clone(), xd.clone()
        for j, dim in enumerate(self.bridge.tier0):
            cr = delta[:, j]
            if dim == 3: thd_next = thd + cr * self.THD_S
            elif dim == 1: xd_next = xd + cr * self.XD_S
        x_next = x + xd_next * self.DT
        th_next = th + thd_next * self.DT
        return torch.stack([x_next/self.X_S, xd_next/self.XD_S,
                            th_next/self.TH_S, thd_next/self.THD_S], dim=-1)
    
    def compute_loss(self, s, v_des_norm):
        sp = self.predict_next(s, v_des_norm)
        loss = (sp[:, 2].pow(2).mean() + 0.1*sp[:, 0].pow(2).mean() +
                0.5*sp[:, 3].pow(2).mean() + 0.1*sp[:, 1].pow(2).mean())
        ctrl = v_des_norm.pow(2).mean()
        total = loss + 0.01 * ctrl
        return total, {'total': total.item(), 'loss': loss.item(), 'ctrl': ctrl.item()}


class AbstractPlannerTrainer:
    """Train policy using abstract planner (NO gradient through WM).
    
    Policy learns PURE STRATEGY: v_des_norm in [-1,1]^m.
    Training uses AbstractDynamics.compute_loss which is purely analytic.
    No WM Jacobian gradient flows to policy.
    """
    
    def __init__(self, wm, policy, execute, bridge, s_target,
                 lr=1e-3, device='cpu', env='pendulum'):
        self.wm = wm; self.policy = policy.to(device)
        self.execute = execute; self.bridge = bridge
        self.s_target = s_target.to(device); self.device = device
        wm.eval()
        for p in wm.parameters(): p.requires_grad = False
        if env == 'pendulum':
            self.abstract = AbstractPendulumDynamics(bridge, self.s_target)
        elif env == 'cartpole':
            self.abstract = AbstractCartPoleDynamics(bridge)
        else:
            raise ValueError(f'Unknown env: {env}')
        self.opt = torch.optim.Adam(policy.parameters(), lr=lr)
        self.loss_history = []
    
    def train_epoch(self, s_dataset, batch_size=256):
        N = s_dataset.shape[0]
        n_batches = max(1, N // batch_size)
        total_loss = 0.0
        for _ in range(n_batches):
            idx = torch.randint(0, N, (batch_size,), device=self.device)
            s_b = s_dataset[idx]
            self.policy.train(); self.opt.zero_grad()
            s_goal = self.s_target.expand(s_b.shape[0], -1)
            v_des_norm = self.policy(s_b, s_goal)
            loss, diag = self.abstract.compute_loss(s_b, v_des_norm)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 10.0)
            self.opt.step()
            total_loss += loss.item()
        avg = total_loss / n_batches
        diag['total'] = avg
        self.loss_history.append(diag)
        return diag
    
    def get_action(self, s):
        self.policy.eval()
        if isinstance(s, np.ndarray):
            s = torch.tensor(s, dtype=torch.float32, device=self.device)
        if s.dim() == 1: s = s.unsqueeze(0)
        with torch.no_grad():
            s_goal = self.s_target.expand(s.shape[0], -1)
            v_des_norm = self.policy(s, s_goal)
        delta = v_des_norm * self.bridge.max_delta.unsqueeze(0)
        a = self.execute(delta, s)
        return a.squeeze().cpu().item()



# ======================================================================
# StudentPlannerTrainer: train policy through distilled Student Dynamics
# ======================================================================

class StudentPlannerTrainer:
    """Train policy through distilled StudentDynamics (learned from WM).
    
    Phase 1 (distill): Student learns WM(s, Execute(v_des)) via supervised learning.
    Phase 2 (policy):  Policy trains through frozen Student. Gradient flows
                       through Student MLP (no vanishing, no hand-crafted loss).
    
    Key advantage: Student captures the FULL nonlinear dynamics of WM+Execute
    (unlike AbstractDynamics which uses a hand-crafted linear formula).
    """
    
    def __init__(self, student, policy, execute, bridge, s_target, lr=3e-3, device="cpu"):
        self.student = student.to(device)
        self.policy = policy.to(device)
        self.execute = execute
        self.bridge = bridge
        self.s_target = s_target.to(device)
        self.device = device
        student.eval()
        for p in student.parameters(): p.requires_grad = False
        self.opt = torch.optim.Adam(policy.parameters(), lr=lr)
    
    def train_epoch(self, s_dataset, batch_size=256):
        N = s_dataset.shape[0]
        n_batches = max(1, N // batch_size)
        total = 0.0
        for _ in range(n_batches):
            idx = torch.randint(0, N, (batch_size,), device=self.device)
            s_b = s_dataset[idx]
            self.policy.train(); self.opt.zero_grad()
            s_goal = self.s_target.expand(s_b.shape[0], -1)
            v_des = self.policy(s_b, s_goal)
            s_pred = self.student(s_b, v_des)  # differentiable!
            loss = (s_pred - s_goal).pow(2).sum(dim=-1).mean()
            loss = loss + 0.01 * v_des.pow(2).mean()
            loss.backward()  # gradient through Student -> Policy
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 10.0)
            self.opt.step()
            total += loss.item()
        return {"total": total / n_batches}
    
    def get_action(self, s):
        self.policy.eval()
        if isinstance(s, np.ndarray):
            s = torch.tensor(s, dtype=torch.float32, device=self.device)
        if s.dim() == 1: s = s.unsqueeze(0)
        with torch.no_grad():
            s_goal = self.s_target.expand(s.shape[0], -1)
            v_des = self.policy(s, s_goal)
        delta = v_des * self.bridge.max_delta.unsqueeze(0)
        a = self.execute(delta, s)
        return a.squeeze().cpu().item()

def evaluate_abstract_policy(trainer, env_name='pendulum', n_trials=10, seed=42, g=10.0):
    '''Evaluate AbstractPlannerTrainer policy on real environment.'''
    if env_name == 'pendulum':
        import gymnasium as gym
        PI_2 = 3.14159 / 2
        env = gym.make('Pendulum-v1')
        successes = 0; all_steps = []
        for trial in range(n_trials):
            obs, _ = env.reset(seed=seed + trial * 100)
            for step in range(300):
                s_n = np.array([obs[0], obs[1], obs[2] / 8.0], dtype=np.float32)
                a = trainer.get_action(s_n)
                obs, _, _, _, _ = env.step([a * 2.0])
                err = min(abs(np.arctan2(obs[1], obs[0]) - PI_2),
                          2 * 3.14159 - abs(np.arctan2(obs[1], obs[0]) - PI_2))
                if err < 0.2:
                    successes += 1; all_steps.append(step + 1); break
            else:
                all_steps.append(300)
        env.close()
        return successes, all_steps
    return 0, []


# ======================================================================
# CDPN v3: Cognitive-Representation Policy with Domain Randomization
# ======================================================================

def train_cognitive_head(wm, bridge, execute, s_dataset, n_samples=5000, n_epochs=300, device='cpu'):
    """Train a CognitiveHead: h (12-dim WM hidden) -> s' (3-dim state prediction).
    
    The Head learns to decode the WM's hidden representation back to the state space.
    This gives us a differentiable path from h to s' for policy training.
    """
    mse = nn.MSELoss()
    head = nn.Sequential(nn.Linear(12, 32), nn.SiLU(), nn.Linear(32, 3)).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    
    wm.eval(); execute.eval = lambda: None
    S, V = [], []
    for _ in range(n_samples):
        idx = torch.randint(0, len(s_dataset), (1,))
        s = s_dataset[idx].to(device)
        v = torch.rand(1, bridge.m, device=device) * 2 - 1
        with torch.no_grad():
            a = execute(v * bridge.max_delta.unsqueeze(0), s)
            h = wm.layers[0](torch.cat([s, a], dim=-1))
            t = wm(torch.cat([s, a], dim=-1))
        S.append(h.squeeze(0)); V.append(t.squeeze(0))
    S = torch.stack(S); V = torch.stack(V)
    
    for ep in range(1, n_epochs + 1):
        perm = torch.randperm(n_samples); tl = 0.0
        for i in range(0, n_samples, 256):
            idx = perm[i:i+256]
            head.train(); opt.zero_grad()
            loss = mse(head(S[idx]), V[idx])
            loss.backward(); opt.step(); tl += loss.item()
        if ep % 100 == 0:
            print(f"  Head epoch {ep:4d}  mse={tl/(n_samples/256):.8f}")
    head.eval()
    return head


class AdaptivePolicy(nn.Module):
    """Policy that operates in cognitive h-space with environment parameter injection.
    
    Inputs:
      h:         (B, h_dim) cognitive state encoding from WM.layers[0]
      env_params:(B, e_dim) environment parameters from CausalBridge
      h_goal:    (B, h_dim) goal state in h-space
      
    Output:
      v_des_norm: (B, 1) normalized strategy instruction in [-1, 1]
      
    Key insight: env_params (a_fit, G_est, max_delta) explicitly encode the
    current environment. When WM adapts -> Bridge updates env_params ->
    Policy automatically adapts WITHOUT retraining.
    """
    
    def __init__(self, h_dim=12, e_dim=3, hidden=24, n_layers=2):
        super().__init__()
        in_dim = h_dim + e_dim + h_dim  # h + env_params + h_goal
        layers = []
        for _ in range(n_layers):
            layers.extend([nn.Linear(in_dim, hidden), nn.Tanh()])
            in_dim = hidden
        layers.append(nn.Linear(in_dim, 1))
        layers.append(nn.Tanh())  # output in [-1, 1]
        self.net = nn.Sequential(*layers)
    
    def forward(self, h, env_params, h_goal):
        """h: (B, h_dim), env_params: (B, e_dim), h_goal: (B, h_dim) -> v: (B, 1)"""
        x = torch.cat([h, env_params, h_goal], dim=-1)
        return self.net(x)


def make_env_params(bridge, batch_size=1, device='cpu'):
    """Create environment parameter tensor from CausalBridge.
    
    Returns (batch_size, 3) tensor: [a_fit, G_est, max_delta]
    These parameters are automatically updated when bridge.update() is called.
    """
    return torch.tensor([[bridge.a_fit, bridge.G_est, bridge.max_delta[0].item()]],
                        dtype=torch.float32, device=device).expand(batch_size, -1)


class CognitiveTrainer:
    """Train AdaptivePolicy in h-space with Domain Randomization.
    
    Training flow:
      s -> WM.encode -> h -> Policy(h, env_params_rand, h_goal) -> v
        -> AbstractDynamics(s, v, a_fit=env_params_rand[0]) -> s' -> loss(s', s_goal)
        
      env_params_rand has a_fit randomized during training (DR).
      
    Deployment flow:
      s -> WM.encode -> h -> Policy(h, env_params_real, h_goal) -> v -> Execute -> a
      
      env_params_real comes from bridge (extracted from adapted WM).
      Policy adapts automatically WITHOUT retraining!
    """
    
    def __init__(self, wm, bridge, execute, policy, head, h_goal, 
                 env='pendulum', dr_range=(5.0, 25.0), lr=3e-3, device='cpu'):
        self.wm = wm.to(device)
        self.bridge = bridge
        self.execute = execute
        self.policy = policy.to(device)
        self.head = head.to(device) if head is not None else None
        self.h_goal = h_goal.to(device)
        self.dr_range = dr_range
        self.device = device
        self.env = env
        
        for p in wm.parameters(): p.requires_grad = False
        if head is not None:
            for p in head.parameters(): p.requires_grad = False
            head.eval()
        
        self.opt = torch.optim.Adam(policy.parameters(), lr=lr)
        
        # Abstract dynamics for DR training
        if env == 'pendulum':
            from control.cdpn import AbstractPendulumDynamics as APD
            S_T = torch.tensor([[0., 1., 0.]]).to(device)
            self.abstract = APD(bridge, S_T)
            self.S_T = S_T
        else:
            raise ValueError(f"Unknown env: {env}")
    
    def _encode(self, s):
        """Encode state into cognitive h-space."""
        with torch.no_grad():
            a = self.execute(torch.zeros(s.shape[0], 1, device=self.device), s)
            h = self.wm.layers[0](torch.cat([s, a], dim=-1))
        return h
    
    def train_epoch(self, s_dataset, H=5, batch_size=128):
        N = len(s_dataset); nb = max(1, N // batch_size); total = 0.0
        
        for _ in range(nb):
            # Sample random a_fit from DR range
            a_fit_rand = np.random.uniform(self.dr_range[0], self.dr_range[1], (batch_size, 1))
            env_rand = torch.tensor(np.concatenate([a_fit_rand, 
                np.full((batch_size, 2), [self.bridge.G_est, self.bridge.max_delta[0].item()])], axis=1),
                dtype=torch.float32, device=self.device)
            
            idx = torch.randint(0, N, (batch_size,), device=self.device)
            s_cur = s_dataset[idx]
            self.policy.train(); self.opt.zero_grad(); loss = 0.0
            
            for t in range(H):
                h_cur = self._encode(s_cur)
                v = self.policy(h_cur, env_rand, self.h_goal.expand(s_cur.shape[0], -1))
                
                # Abstract dynamics with randomized a_fit
                if self.env == 'pendulum':
                    sin_cur = s_cur[:, 1:2]; cos_cur = s_cur[:, 0:1]; td_cur = s_cur[:, 2:3] * 8.0
                    th = torch.atan2(sin_cur, cos_cur)
                    if v.dim() == 1: v = v.unsqueeze(1)

                    control = v[:, 0:1] * self.bridge.max_delta[0].item() * 8.0
                    td_next = td_cur + env_rand[:, 0:1] * sin_cur * self.abstract.DT + v[:, 0:1] * self.bridge.max_delta[0].item() * 8.0
                    th_next = th + td_next * self.abstract.DT
                    s_cur = torch.cat([torch.cos(th_next), torch.sin(th_next), td_next / 8.0], dim=-1)
                
                loss += (0.9 ** t) * (s_cur - self.S_T).pow(2).sum(dim=-1).mean()
            
            loss += 0.01 * v.pow(2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 10.0)
            self.opt.step()
            total += loss.item()
        
        return {'total': total / nb}
    
    def get_action(self, s):
        self.policy.eval()
        if isinstance(s, np.ndarray):
            s = torch.tensor(s, dtype=torch.float32, device=self.device)
        if s.dim() == 1:
            s = s.unsqueeze(0)
        
        with torch.no_grad():
            h = self._encode(s)
            env_params = make_env_params(self.bridge, s.shape[0], self.device)
            v = self.policy(h, env_params, self.h_goal.expand(s.shape[0], -1))
        
        a = self.execute(v * self.bridge.max_delta.unsqueeze(0), s)
        return a.squeeze().cpu().item()


