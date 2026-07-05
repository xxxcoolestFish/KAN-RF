"""Decision Network v3: KAN-adapted policy trained via KAN gradient signals.

Core idea: KAN provides gradients, not features.

  Training:  s → [π_θ] → a → [frozen KAN] → s'_pred → L(s'_pred, s*)
             gradient flows L → s'_pred → KAN → a → π_θ

  Deployment: s → [π_θ] → a   (pure forward, no KAN)

KAN's accurate Jacobian (∂s'/∂a, cos_sim ≈ 0.92 after CWS training) provides
high-quality training signal.  Root cause 3 is avoided because we only need
gradient DIRECTION, not inverse magnitude.

Key KAN-specific advantage: B-spline activation density ρ(s) provides a
built-in per-sample confidence weight — downweight states where KAN is
extrapolating.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# Policy Network
# ═══════════════════════════════════════════════════════════════════════════════

class KANPolicy(nn.Module):
    """Small MLP policy — the decision network.

    Input:  s ∈ R^d_s  (normalized state)
    Output: a ∈ [-1, 1]  (normalized action), optionally k ∈ [0, 1] (timescale)

    Architecture is deliberately simple.  KAN provides the physics knowledge
    through gradients; the policy only needs capacity to memorize the mapping.
    For pendulum: [3, 64, 64, 1] ≈ 4.5k params.

    When output_k=True, the network outputs (a, k_cont) where:
      a ∈ [-1, 1] via tanh
      k_cont ∈ [0, 1] via sigmoid → maps to discrete k = round(k_cont * 16)
    """

    def __init__(self, state_dim=3, action_dim=1, hidden=64, n_layers=2,
                 output_k=False):
        super().__init__()
        self.output_k = output_k
        out_dim = action_dim + 1 if output_k else action_dim
        layers = []
        in_dim = state_dim
        for _ in range(n_layers):
            layers.extend([nn.Linear(in_dim, hidden), nn.ReLU()])
            in_dim = hidden
        layers.append(nn.Linear(in_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, s):
        """s: (B, state_dim) → a: (B, action_dim) in [-1, 1], [optional k]."""
        out = self.net(s)
        if self.output_k:
            a = torch.tanh(out[:, :1])
            k_cont = torch.sigmoid(out[:, 1:2])
            return torch.cat([a, k_cont], dim=-1)
        return torch.tanh(out)


# ═══════════════════════════════════════════════════════════════════════════════
# Residual Physics Policy (PINN-inspired: physics prior + learned residual)
# ═══════════════════════════════════════════════════════════════════════════════

class ResidualPhysicsPolicy(nn.Module):
    """PINN-inspired policy: a(s) = a_physics(s) + δ_θ(s).

    The physics prior encodes the known energy-shaping control law:
      E = 0.5·θ̇² + G·sin(θ)
      a_physics = k_energy · (E - E_des) · θ̇   (swing-up)
                + k_stable · dθ                  (stabilize near upright)
                + k_damp · θ̇                     (damping)

    The residual MLP δ_θ(s) only needs to learn what the physics prior misses:
    friction, fine stabilization, model mismatch.  This means:
      - δ_θ is small → KAN gradient errors have less impact
      - a_physics provides a safe fallback when KAN is uncertain
      - Each term is interpretable (energy shaping vs. learned correction)

    Learnable parameters:
      - k_energy, k_stable, k_damp: scalar physics coefficients (initialized
        to sensible defaults, refined by KAN gradient)
      - residual_net: small MLP for δ_θ(s)

    Args:
        state_dim: state dimension (3 for pendulum)
        hidden: residual MLP hidden size
        n_layers: residual MLP depth
        G: gravitational constant (10.0 for Pendulum-v1)
        E_des: target energy (G for upright at rest)
        init_k_energy: initial energy shaping gain
        init_k_stable: initial stabilization gain
        init_k_damp: initial damping gain
    """

    def __init__(self, state_dim=3, action_dim=1, hidden=32, n_layers=2,
                 G=10.0, E_des=None,
                 init_k_energy=0.15, init_k_stable=-2.0, init_k_damp=-0.3):
        super().__init__()
        self.G = G
        self.E_des = E_des if E_des is not None else G

        # ── Learnable physics coefficients ──
        self.k_energy = nn.Parameter(torch.tensor(init_k_energy))
        self.k_stable = nn.Parameter(torch.tensor(init_k_stable))
        self.k_damp = nn.Parameter(torch.tensor(init_k_damp))

        # ── Residual MLP: δ_θ(s) → small correction ──
        # Deliberately smaller than standalone MLP — physics does most of the work
        layers = []
        in_dim = state_dim
        for _ in range(n_layers):
            layers.extend([nn.Linear(in_dim, hidden), nn.ReLU()])
            in_dim = hidden
        layers.append(nn.Linear(in_dim, action_dim))
        self.residual_net = nn.Sequential(*layers)

        # Scale factor for residual (initialized small → physics dominates early)
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def compute_physics_action(self, s):
        """Energy-shaping control law — the known physics prior.

        Args:
            s: (B, 3) normalized state [cosθ, sinθ, θ̇/8]

        Returns:
            a_physics: (B, 1) normalized action in [-1, 1]
            diagnostics: dict with E, delta_E, d_angle for analysis
        """
        cos, sin = s[:, 0:1], s[:, 1:2]
        thd_norm = s[:, 2:3]       # normalized θ̇ (÷8)
        thd = thd_norm * 8.0       # denormalize

        # Energy
        E = 0.5 * thd.pow(2) + self.G * sin      # (B, 1)
        delta_E = E - self.E_des                  # > 0: too much energy, < 0: need more

        # Angle from upright (for stabilization)
        angle = torch.atan2(sin, cos)             # (B, 1)
        d_angle = angle - (torch.pi / 2)          # deviation from upright
        # Normalize to [-π, π]
        d_angle = torch.atan2(torch.sin(d_angle), torch.cos(d_angle))

        # ── Physics control law ──
        # Swing-up: pump energy by applying torque aligned with velocity
        #   a_swing = k_energy * (E - E_des) * θ̇
        #   When E < E_des (need energy): torque with velocity → pump
        #   When E > E_des (too much): torque against velocity → brake
        a_swing = self.k_energy * delta_E * thd

        # Stabilize: LQR-like near upright
        a_stable = self.k_stable * d_angle

        # Damping: always oppose velocity (numerical stability)
        a_damp = self.k_damp * thd

        # Combine with smooth transition based on proximity to upright
        # Near upright (sin → 1): stabilize. Away (sin → -1): swing.
        uprightness = sin.clamp(0.1, 1.0)  # (B, 1), ∈ [0.1, 1.0]
        # swing_weight ∈ [0, 1]: 0 = upright (use stabilization), 1 = hanging (use swing)
        swing_weight = (1.0 - uprightness) / 0.9  # maps [0.1, 1.0] → [1.0, 0.0]
        swing_weight = swing_weight.clamp(0.0, 1.0)

        a_physics = swing_weight * a_swing + (1.0 - swing_weight) * (a_stable + a_damp)

        # Clamp to normalized action range
        a_physics = torch.tanh(a_physics)

        diagnostics = {
            'E': E,
            'delta_E': delta_E,
            'd_angle': d_angle,
            'swing_weight': swing_weight,
            'a_swing': a_swing,
            'a_stable': a_stable,
            'a_damp': a_damp,
        }
        return a_physics, diagnostics

    def forward(self, s, return_diag=False):
        """a(s) = tanh(a_physics(s) + residual_scale · δ_θ(s)).

        Args:
            s: (B, state_dim)
            return_diag: if True, also return diagnostics dict

        Returns:
            a: (B, action_dim) in [-1, 1]
            [optional] diag: dict with physics decomposition
        """
        a_physics, diag = self.compute_physics_action(s)
        delta = self.residual_scale * self.residual_net(s)
        a = torch.tanh(a_physics + delta)

        if return_diag:
            diag['a_physics'] = a_physics
            diag['delta_residual'] = delta
            diag['residual_scale'] = self.residual_scale
            return a, diag
        return a


# ═══════════════════════════════════════════════════════════════════════════════
# KAN Gradient Trainer
# ═══════════════════════════════════════════════════════════════════════════════

class KANEnergyTrainer:
    """Train π_θ using physics-informed energy-based loss through KAN.

    Unlike KANGradientTrainer which uses MSE(s_pred, s*) — impossible to
    minimize from the bottom of swing — this uses:

      L = -w_swing · (E_pred - E) + w_stable · MSE(s_pred, s*)

    where:
      E = 0.5·θ̇² + G·sin(θ)     (pendulum energy)
      E_pred via KAN(s, a)
      w_swing → 1 at bottom (swing-up: maximize energy gain)
      w_stable → 1 at top   (stabilize: minimize distance to upright)

    This is PINN philosophy: physics knowledge (energy is the right
    objective for swing-up) embedded in the loss function, not just
    the architecture.

    Args:
        kan: frozen KAN world model
        policy: π_θ network
        s_target: target state [0, 1, 0]
        G: gravitational constant (10.0 for Pendulum-v1)
        lr, lambda_ctrl, clip_grad: standard training params
    """

    def __init__(self, kan, policy, s_target, G=10.0, lr=1e-3,
                 lambda_ctrl=0.01, clip_grad=10.0, device='cpu',
                 multi_scale=False, k_norm=None):
        self.kan = kan
        self.policy = policy.to(device)
        self.s_target = s_target.to(device)
        self.device = device
        self.G = G
        self.lambda_ctrl = lambda_ctrl
        self.clip_grad = clip_grad
        self.multi_scale = multi_scale
        self.k_norm = k_norm

        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
        self.loss_history = []

        self.kan.eval()
        for p in self.kan.parameters():
            p.requires_grad = False

    def _compute_energy(self, s):
        """Compute pendulum energy from normalized state.

        s: (B, 3) [cosθ, sinθ, θ̇/8]
        Returns: E: (B, 1) — energy, E_des = G
        """
        sin = s[:, 1:2]
        thd_norm = s[:, 2:3]
        thd = thd_norm * 8.0
        E = 0.5 * thd.pow(2) + self.G * sin
        return E

    def _build_kan_input(self, s, policy_out):
        """Build KAN input. Same as KANGradientTrainer."""
        if self.multi_scale == 'policy':
            a = policy_out[:, :1]
            k_cont = policy_out[:, 1:2]
            return torch.cat([s, a, k_cont], dim=-1), a
        elif self.multi_scale == 'fixed' and self.k_norm is not None:
            a = policy_out
            k_batch = self.k_norm.expand(s.shape[0], -1).to(self.device)
            return torch.cat([s, a, k_batch], dim=-1), a
        else:
            a = policy_out
            return torch.cat([s, a], dim=-1), a

    def train_step(self, s_batch, weight_batch=None):
        """Single training step with energy-guided loss.

        The key insight: for swing-up states, maximize energy gain, not
        minimize distance to target (which is impossible in one step).
        For stabilize states, minimize distance as usual.
        """
        B = s_batch.shape[0]

        self.policy.train()
        self.optimizer.zero_grad()

        # Forward: s → policy → a → KAN → s'_pred
        policy_out = self.policy(s_batch)
        kan_input, a = self._build_kan_input(s_batch, policy_out)
        s_pred = self.kan(kan_input)

        # ── Physics-informed loss ──
        E_current = self._compute_energy(s_batch)       # (B, 1)
        E_pred = self._compute_energy(s_pred)           # (B, 1)
        E_des = self.G
        delta_E = E_current - E_des                      # >0: too much, <0: need more

        # Swing weight: 1 at bottom (need to swing), 0 at top (need to stabilize)
        sin = s_batch[:, 1:2]
        w_swing = ((1.0 - sin) / 2.0).clamp(0.0, 1.0)   # (B, 1)
        w_stable = ((1.0 + sin) / 2.0).clamp(0.0, 1.0)  # (B, 1)

        # Energy gain: positive when energy moves toward E_des
        # If E < E_des (need energy): reward increasing E
        # If E > E_des (too much): reward decreasing E
        energy_deficit = E_des - E_current                # >0: need energy
        energy_gain = (E_pred - E_current) * torch.sign(energy_deficit)
        # energy_gain > 0 means we moved in the right direction

        energy_loss = -energy_gain.mean()                 # minimize → maximize gain

        # Distance loss for stabilize states
        dist_loss = (s_pred - self.s_target.expand(B, -1)).pow(2).sum(dim=-1, keepdim=True)
        dist_loss = (w_stable * dist_loss).mean()

        # Blend
        pred_loss = energy_loss + dist_loss
        ctrl_loss = a.pow(2).mean()
        total_loss = pred_loss + self.lambda_ctrl * ctrl_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.clip_grad)
        self.optimizer.step()

        ld = {
            'total': total_loss.item(),
            'pred': pred_loss.item(),
            'energy': energy_loss.item(),
            'dist': dist_loss.item(),
            'ctrl': ctrl_loss.item(),
            'mean_E': E_current.mean().item(),
            'mean_w_swing': w_swing.mean().item(),
        }
        self.loss_history.append(ld)
        return ld

    def train_epoch(self, s_dataset, batch_size=256, n_batches=None,
                    weight_fn=None):
        N = s_dataset.shape[0]
        if n_batches is None:
            n_batches = max(1, N // batch_size)
        epoch_losses = []
        for _ in range(n_batches):
            idx = torch.randint(0, N, (batch_size,), device=self.device)
            s_batch = s_dataset[idx]
            weight_batch = weight_fn(s_batch) if weight_fn else None
            ld = self.train_step(s_batch, weight_batch)
            epoch_losses.append(ld)
        return {k: np.mean([l[k] for l in epoch_losses]) for k in epoch_losses[0]}

    @torch.no_grad()
    def get_action(self, s):
        self.policy.eval()
        if isinstance(s, np.ndarray):
            s = torch.tensor(s, dtype=torch.float32, device=self.device)
        if s.dim() == 1:
            s = s.unsqueeze(0)
        out = self.policy(s).squeeze().cpu()
        if self.multi_scale == 'policy':
            a = out[0].item()
            k = max(1, min(16, round(out[1].item() * 16)))
            return a, k
        return out.item()


class KANGradientTrainer:
    """Train π_θ using frozen KAN as a differentiable loss function.

    The KAN world model f(s, a) → s' evaluates "how good is this action?"
    by predicting the next state.  The gradient ∂L/∂θ flows through KAN's
    accurate Jacobian back to the policy.

    Supports both single-scale KAN [s, a] → s' and multi-scale KAN
    [s, a, k_norm] → s' (when k_norm is provided or policy outputs k).

    Args:
        kan: frozen KAN world model f(s, a, [k]) → s'
        policy: π_θ network to train
        s_target: target state (e.g., [0, 1, 0] for pendulum upright)
        lr: learning rate for policy optimizer
        lambda_ctrl: control penalty weight
        clip_grad: gradient clipping norm
        multi_scale: if True, policy outputs (a, k) and KAN takes k as input.
                     if 'fixed', use self.k_norm for all states.
    """

    def __init__(self, kan, policy, s_target, lr=1e-3,
                 lambda_ctrl=0.01, clip_grad=10.0, device='cpu',
                 multi_scale=False, k_norm=None):
        self.kan = kan
        self.policy = policy.to(device)
        self.s_target = s_target.to(device)
        self.device = device
        self.lambda_ctrl = lambda_ctrl
        self.clip_grad = clip_grad
        self.multi_scale = multi_scale
        self.k_norm = k_norm  # fixed k for 'fixed' mode, e.g., torch.tensor([[4/16]])

        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
        self.loss_history = []

        # Freeze KAN — we only use it for forward + gradient
        self.kan.eval()
        for p in self.kan.parameters():
            p.requires_grad = False

    def _build_kan_input(self, s, policy_out):
        """Build KAN input from state and policy output.

        Args:
            s: (B, state_dim)
            policy_out: (B, out_dim) — either just a, or (a, k_cont)

        Returns:
            kan_input: (B, kan_in_dim) — [s, a] or [s, a, k_norm]
            a: (B, 1) — extracted action
        """
        if self.multi_scale == 'policy':
            # Policy outputs (a, k_cont)
            a = policy_out[:, :1]
            k_cont = policy_out[:, 1:2]
            k_norm = k_cont  # k_cont ∈ [0, 1] maps directly to k/16
            return torch.cat([s, a, k_norm], dim=-1), a
        elif self.multi_scale == 'fixed' and self.k_norm is not None:
            # Fixed k, policy outputs only a
            a = policy_out
            k_batch = self.k_norm.expand(s.shape[0], -1).to(self.device)
            return torch.cat([s, a, k_batch], dim=-1), a
        else:
            # Single-scale: no k
            a = policy_out
            return torch.cat([s, a], dim=-1), a

    def train_step(self, s_batch, weight_batch=None):
        """Single training step.

        Args:
            s_batch: (B, state_dim) normalized states
            weight_batch: optional (B,) per-sample weights

        Returns:
            loss_dict with 'total', 'pred', 'ctrl' components
        """
        B = s_batch.shape[0]
        s_target_batch = self.s_target.expand(B, -1)

        self.policy.train()
        self.optimizer.zero_grad()

        # Forward: s → π_θ → (a, [k]) → KAN → s'_pred
        policy_out = self.policy(s_batch)              # (B, out_dim)
        kan_input, a = self._build_kan_input(s_batch, policy_out)
        s_pred = self.kan(kan_input)                   # (B, state_dim)

        # Loss: prediction error + control penalty
        per_sample_pred_loss = (s_pred - s_target_batch).pow(2).sum(dim=-1)
        if weight_batch is not None:
            pred_loss = (per_sample_pred_loss * weight_batch).mean()
        else:
            pred_loss = per_sample_pred_loss.mean()

        ctrl_loss = a.pow(2).mean()
        total_loss = pred_loss + self.lambda_ctrl * ctrl_loss

        # Backward: gradient flows through KAN (frozen) → a → π_θ
        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.clip_grad)
        self.optimizer.step()

        loss_dict = {
            'total': total_loss.item(),
            'pred': pred_loss.item(),
            'ctrl': ctrl_loss.item(),
        }
        self.loss_history.append(loss_dict)
        return loss_dict

    def train_epoch(self, s_dataset, batch_size=256, n_batches=None,
                    weight_fn=None):
        """One epoch over the state dataset.

        Args:
            s_dataset: (N, state_dim) tensor of training states
            batch_size: mini-batch size
            n_batches: number of batches per epoch (default: N/batch_size)
            weight_fn: optional fn(s_batch) → weight_batch for per-sample weighting

        Returns:
            mean loss dict
        """
        N = s_dataset.shape[0]
        if n_batches is None:
            n_batches = max(1, N // batch_size)

        epoch_losses = []
        for _ in range(n_batches):
            idx = torch.randint(0, N, (batch_size,), device=self.device)
            s_batch = s_dataset[idx]
            weight_batch = weight_fn(s_batch) if weight_fn else None
            ld = self.train_step(s_batch, weight_batch)
            epoch_losses.append(ld)

        return {k: np.mean([l[k] for l in epoch_losses]) for k in epoch_losses[0]}

    @torch.no_grad()
    def get_action(self, s):
        """Deployment: pure forward pass, no KAN involved.

        Args:
            s: (state_dim,) numpy or (1, state_dim) tensor
        Returns:
            If multi_scale='policy': (a_norm, k) tuple
            Else: a_norm scalar float in [-1, 1]
        """
        self.policy.eval()
        if isinstance(s, np.ndarray):
            s = torch.tensor(s, dtype=torch.float32, device=self.device)
        if s.dim() == 1:
            s = s.unsqueeze(0)
        out = self.policy(s).squeeze().cpu()
        if self.multi_scale == 'policy':
            a = out[0].item()
            k = max(1, min(16, round(out[1].item() * 16)))
            return a, k
        return out.item()


# ═══════════════════════════════════════════════════════════════════════════════
# KAN Activation Density Computer (for sample weighting)
# ═══════════════════════════════════════════════════════════════════════════════

class KANDensityWeight:
    """Compute per-state B-spline activation density for sample weighting.

    When ρ(s) is low, KAN is extrapolating (only SiLU baseline, no B-spline
    contribution).  These states should be downweighted during policy training
    because KAN's gradient signal is less reliable there.

    This is a KAN-specific advantage — MLP world models cannot provide this.
    """

    def __init__(self, kan, device='cpu'):
        self.kan = kan
        self.device = device
        self.kan.eval()
        for p in self.kan.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def compute(self, s_batch, a_batch=None):
        """Compute activation density for batch of states.

        Args:
            s_batch: (B, state_dim)
            a_batch: (B, 1) or None → uses a=0

        Returns:
            rho: (B,) ∈ [0, 1], mean activation density across all layers
        """
        B = s_batch.shape[0]
        if a_batch is None:
            a_batch = torch.zeros(B, 1, device=self.device)

        x = torch.cat([s_batch, a_batch], dim=-1)

        # Use KAN's internal activation recording
        _, B_list, _ = self.kan(x, return_activations=True)

        # Activation density per layer: fraction of active basis functions
        densities = []
        for B_mat in B_list:
            # B_mat: (B, in_dim, n_basis) — basis function values
            active = (B_mat > 1e-6).float().mean(dim=-1)   # (B, in_dim)
            densities.append(active.mean(dim=-1))           # (B,)

        rho = torch.stack(densities, dim=1).mean(dim=1)    # (B,)
        return rho

    def as_weights(self, s_batch, a_batch=None, min_weight=0.1):
        """Convert density to sample weights.

        High density → weight ≈ 1.0 (trust KAN)
        Low density  → weight ≈ min_weight (KAN extrapolating, less reliable)
        """
        rho = self.compute(s_batch, a_batch)
        # Map [0, 1] → [min_weight, 1.0]
        return rho * (1.0 - min_weight) + min_weight


# ═══════════════════════════════════════════════════════════════════════════════
# Multi-Step Extension
# ═══════════════════════════════════════════════════════════════════════════════

class KANMultiStepTrainer(KANGradientTrainer):
    """Train policy with H-step KAN rollout (backprop through time in model).

    L = Σₜ γᵗ·||sₜ - s*||²  where sₜ₊₁ = KAN(sₜ, π_θ(sₜ), [k])

    Gradient flows through H consecutive KAN forward passes.
    KAN's B-spline derivatives have bounded norm → gradient is naturally
    regularized against explosion.

    NOTE: For multi_scale='policy', the policy outputs (a, k).  The k value
    is used for the first step only; subsequent steps use the same k.
    This is because k represents the execution duration of the FIRST action.
    """

    def __init__(self, kan, policy, s_target, horizon=3, lr=1e-3,
                 lambda_ctrl=0.01, clip_grad=10.0, device='cpu',
                 multi_scale=False, k_norm=None):
        super().__init__(kan, policy, s_target, lr, lambda_ctrl, clip_grad,
                         device, multi_scale, k_norm)
        self.horizon = horizon

    def train_step(self, s_batch, weight_batch=None):
        """Multi-step rollout training step."""
        B = s_batch.shape[0]

        self.policy.train()
        self.optimizer.zero_grad()

        s = s_batch
        total_pred_loss = torch.tensor(0.0, device=self.device)
        total_ctrl_loss = torch.tensor(0.0, device=self.device)
        gamma = 0.9  # discount for future steps

        # First step: get k from policy if multi_scale='policy'
        policy_out = self.policy(s)
        kan_input, a = self._build_kan_input(s, policy_out)
        s_next = self.kan(kan_input)

        step_pred_loss = (s_next - self.s_target.expand(B, -1)).pow(2).sum(dim=-1)
        if weight_batch is not None:
            step_pred_loss = (step_pred_loss * weight_batch).mean()
        else:
            step_pred_loss = step_pred_loss.mean()
        total_pred_loss = total_pred_loss + step_pred_loss
        total_ctrl_loss = total_ctrl_loss + a.pow(2).mean()
        s = s_next

        for t in range(1, self.horizon):
            # For subsequent steps, use a=0 (let dynamics evolve naturally)
            # or re-query policy.  Using a=0 is more stable for rollout.
            a_zero = torch.zeros(B, 1, device=self.device)
            if self.multi_scale:
                # Reuse k from first step
                k_batch = kan_input[:, -1:]  # last dim is k_norm
                kan_input_t = torch.cat([s, a_zero, k_batch], dim=-1)
            else:
                kan_input_t = torch.cat([s, a_zero], dim=-1)
            s_next = self.kan(kan_input_t)

            step_pred_loss = (s_next - self.s_target.expand(B, -1)).pow(2).sum(dim=-1)
            if weight_batch is not None:
                step_pred_loss = (step_pred_loss * weight_batch).mean()
            else:
                step_pred_loss = step_pred_loss.mean()

            total_pred_loss = total_pred_loss + (gamma ** t) * step_pred_loss
            s = s_next

            if s_next.norm(dim=-1).max() < 0.01:
                break

        total_loss = total_pred_loss + self.lambda_ctrl * total_ctrl_loss
        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.clip_grad)
        self.optimizer.step()

        ld = {
            'total': total_loss.item(),
            'pred': total_pred_loss.item(),
            'ctrl': total_ctrl_loss.item(),
        }
        self.loss_history.append(ld)
        return ld


# ═══════════════════════════════════════════════════════════════════════════════
# Sequence Rollout Trainer (v3.1 proper)
# ═══════════════════════════════════════════════════════════════════════════════

class KANSequenceTrainer(KANEnergyTrainer):
    """Train π_θ with full H-step KAN rollout — policy queried at EVERY step.

    Core idea: π_θ learns to plan action SEQUENCES, not just single-step
    reactions.  At each rollout step t, the policy observes the model-predicted
    state s_t and outputs a new action a_t = π_θ(s_t).  This is backprop-through-
    time through the frozen KAN world model.

      s_0 → π_θ → a_0 → KAN → s_1 → π_θ → a_1 → KAN → s_2 → ...

    Three KAN-specific mechanisms exploited:
      1. Full B-spline parameters used at every rollout step (uses KAN knowledge)
      2. Multi-step Jacobian ∂s_H/∂a_0 ≈ Σ ∂s_t/∂a_0 accumulates signal → avoids
         root cause 3 (single-step amplification factor ~25×)
      3. Activation density ρ(s_t) downweights steps where rollout drifts OOD

    Args:
        horizon: rollout length (5–8 for pendulum)
        gamma: discount for future-step losses
        use_density_weight: if True, weight each step by KAN activation density
    """

    def __init__(self, kan, policy, s_target, G=10.0, horizon=5,
                 gamma=0.85, lr=1e-3, lambda_ctrl=0.01, clip_grad=10.0,
                 device='cpu', use_density_weight=True, multi_scale=False,
                 k_norm=None):
        super().__init__(kan, policy, s_target, G=G, lr=lr,
                         lambda_ctrl=lambda_ctrl, clip_grad=clip_grad,
                         device=device, multi_scale=multi_scale, k_norm=k_norm)
        self.horizon = horizon
        self.gamma = gamma
        self.use_density_weight = use_density_weight

    def _compute_step_density(self, s):
        """Compute activation density ρ(s) for a batch of states going through KAN.

        Returns ρ ∈ [0, 1] where 1 = well-covered training region.
        """
        # Need to pass through KAN to get activations.  Use a dummy action.
        B = s.shape[0]
        a_dummy = torch.zeros(B, 1, device=self.device)
        if self.multi_scale and self.multi_scale != 'policy':
            k_batch = self.k_norm.expand(B, -1).to(self.device)
            x = torch.cat([s, a_dummy, k_batch], dim=-1)
        else:
            x = torch.cat([s, a_dummy], dim=-1)

        with torch.no_grad():
            try:
                _, B_list, E_list = self.kan(x, return_activations=True)
                # ρ = fraction of active basis functions (averaged over layers)
                densities = []
                for B_mat in B_list:
                    active = (B_mat > 1e-6).float().mean(dim=-1)  # (B, in_dim)
                    densities.append(active.mean(dim=-1))          # (B,)
                rho = torch.stack(densities, dim=1).mean(dim=1)    # (B,)
            except Exception:
                rho = torch.ones(B, device=self.device)
        return rho

    def train_step(self, s_batch, weight_batch=None):
        """H-step rollout: policy queried at every step through KAN."""
        B = s_batch.shape[0]

        self.policy.train()
        self.optimizer.zero_grad()

        s = s_batch
        total_pred_loss = torch.tensor(0.0, device=self.device)
        total_ctrl_loss = torch.tensor(0.0, device=self.device)
        total_density = torch.tensor(0.0, device=self.device)

        E_current = self._compute_energy(s_batch)
        E_des = self.G

        for t in range(self.horizon):
            # ── Step density (OOD detection) ──
            if self.use_density_weight:
                rho_t = self._compute_step_density(s).clamp(0.1, 1.0)  # (B,)
            else:
                rho_t = torch.ones(B, device=self.device)

            # ── Policy forward: s_t → a_t ──
            policy_out = self.policy(s)
            kan_input, a = self._build_kan_input(s, policy_out)

            # ── KAN forward: (s_t, a_t) → s_{t+1} ──
            s_next = self.kan(kan_input)

            # ── Energy-guided loss for this step ──
            E_pred = self._compute_energy(s_next)
            energy_deficit = E_des - E_current
            energy_gain = (E_pred - E_current) * torch.sign(energy_deficit)

            sin = s[:, 1:2]
            w_swing = ((1.0 - sin) / 2.0).clamp(0.0, 1.0)
            w_stable = ((1.0 + sin) / 2.0).clamp(0.0, 1.0)

            energy_loss = -energy_gain.mean()
            dist_loss = (w_stable * (s_next - self.s_target.expand(B, -1)).pow(2).sum(dim=-1, keepdim=True)).mean()
            step_pred = energy_loss + dist_loss

            # Weight by density and discount
            step_weight = (self.gamma ** t) * rho_t.mean()
            total_pred_loss = total_pred_loss + step_pred * rho_t.mean()
            total_ctrl_loss = total_ctrl_loss + a.pow(2).mean() * (self.gamma ** t)
            total_density = total_density + rho_t.mean()

            # ── Advance state ──
            E_current = E_pred
            s = s_next

            # Early stop if all states near target
            if s_next.norm(dim=-1).max() < 0.1:
                break

        # Normalize by effective horizon (avoid bias toward shorter rollouts)
        pred_loss = total_pred_loss / max(total_density.item(), 0.01)
        ctrl_loss = total_ctrl_loss / self.horizon
        total_loss = pred_loss + self.lambda_ctrl * ctrl_loss * self.horizon

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.clip_grad)
        self.optimizer.step()

        ld = {
            'total': total_loss.item(),
            'pred': pred_loss.item(),
            'ctrl': ctrl_loss.item(),
            'horizon_eff': min(self.horizon, int(total_density.item() / max(B, 1))),
        }
        self.loss_history.append(ld)
        return ld

    def train_epoch(self, s_dataset, batch_size=256, n_batches=None,
                    weight_fn=None):
        return KANEnergyTrainer.train_epoch(
            self, s_dataset, batch_size=batch_size, n_batches=n_batches,
            weight_fn=weight_fn)
