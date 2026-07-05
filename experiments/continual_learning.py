"""Continual Learning Experiment: KAN vs MLP under physics parameter changes.

Tests KAN's core claim: B-spline local support + three-factor learning rate
enables rapid adaptation to changed dynamics without catastrophic forgetting.

Protocol:
  1. Generate training data from default Pendulum (g=10.0)
  2. Train both KAN and MLP world models to comparable accuracy
  3. Run a continuous trajectory with 4 gravity switches
  4. At each step: predict s', observe real s', compute L2 error, online-update
  5. Compare error recovery curves between KAN (three-factor) and MLP (SGD)

Usage:
  conda run -n pyt python experiments/continual_learning.py
  conda run -n pyt python experiments/continual_learning.py --steps 800 --no-train
"""
import torch
import torch.nn as nn
import numpy as np
import time, argparse, sys, os

# ── Add project root to path ──
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kanrf import KAN

# ═══════════════════════════════════════════════════════════════════════════════
# Configurable Pendulum Environment
# ═══════════════════════════════════════════════════════════════════════════════


class ConfigurablePendulum:
    """Thin wrapper around Pendulum-v1 that allows runtime gravity changes.

    Gymnasium's Pendulum-v1 stores gravity as env.unwrapped.g.
    We expose set_g() for clean runtime modification.
    """

    def __init__(self, g=10.0, seed=None):
        import gymnasium as gym
        self.env = gym.make("Pendulum-v1", g=g)
        self.g = g
        if seed is not None:
            self.env.reset(seed=seed)

    def set_g(self, new_g):
        """Change gravity at runtime."""
        self.g = new_g
        self.env.unwrapped.g = new_g

    def reset(self, seed=None):
        obs, info = self.env.reset(seed=seed)
        return np.array(obs, dtype=np.float32)

    def step(self, action):
        """Step with scalar or array action. Returns (obs, reward, terminated, truncated)."""
        if np.isscalar(action):
            action = np.array([action], dtype=np.float32)
        obs, reward, term, trunc, info = self.env.step(action)
        return np.array(obs, dtype=np.float32), reward, term, trunc

    def get_state(self):
        """Return raw (theta, thetadot) for state save/restore."""
        return self.env.unwrapped.state

    def set_state(self, state):
        self.env.unwrapped.state = state


# ═══════════════════════════════════════════════════════════════════════════════
# MLP World Model (baseline for comparison)
# ═══════════════════════════════════════════════════════════════════════════════


class MLPWorldModel(nn.Module):
    """Simple MLP world model: f(s, a) → s'.

    Architecture matched to KAN[4,12,3] ≈ 756 params.
    MLP[4, 32, 32, 3] ≈ 4*32+32 + 32*32+32 + 32*3+3 = 128+32+1024+32+96+3 = 1315 params
    Slightly larger to be fair to MLP baseline.
    """

    def __init__(self, state_dim=3, action_dim=1, hidden=32, n_layers=2):
        super().__init__()
        in_dim = state_dim + action_dim  # 4
        layers = []
        for _ in range(n_layers):
            layers.extend([nn.Linear(in_dim, hidden), nn.ReLU()])
            in_dim = hidden
        layers.append(nn.Linear(in_dim, state_dim))  # 3
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ═══════════════════════════════════════════════════════════════════════════════
# Data Generation
# ═══════════════════════════════════════════════════════════════════════════════


def generate_pendulum_data(env, n_transitions=20000):
    """Collect (s, a, s') transitions with random actions.
    Returns normalized x=(s_norm, a_norm), y=s_next_norm.
    """
    states, actions, next_states = [], [], []
    obs = env.reset()

    for _ in range(n_transitions):
        a = np.random.uniform(-2.0, 2.0)
        next_obs, _, term, trunc = env.step(a)

        states.append(obs.copy())
        actions.append(a)
        next_states.append(next_obs.copy())

        if term or trunc:
            obs = env.reset()
        else:
            obs = next_obs

    s = torch.tensor(np.array(states), dtype=torch.float32)
    a = torch.tensor(np.array(actions), dtype=torch.float32).unsqueeze(-1)
    s_next = torch.tensor(np.array(next_states), dtype=torch.float32)

    # Normalize
    s_norm = s.clone()
    s_norm[:, 2] /= 8.0
    a_norm = a / 2.0
    s_next_norm = s_next.clone()
    s_next_norm[:, 2] /= 8.0

    x = torch.cat([s_norm, a_norm], dim=-1)
    return x, s_next_norm


# ═══════════════════════════════════════════════════════════════════════════════
# Energy Controller (for meaningful trajectory generation)
# ═══════════════════════════════════════════════════════════════════════════════


class EnergyController:
    """Oracle energy-based swing-up + stabilize controller.

    Works under any gravity — the energy formula uses g dynamically.
    """

    def __init__(self, k_swing=1.5, k_stable=5.0, k_damp=1.0):
        self.k_swing = k_swing
        self.k_stable = k_stable
        self.k_damp = k_damp

    def get_action(self, obs, g=10.0):
        cos_th, sin_th, thd = obs
        E = 0.5 * thd * thd + g * sin_th
        E_des = g
        near_upright = abs(cos_th) < 0.5 and sin_th > 0 and abs(thd) < 3.0

        if near_upright:
            angle = np.arctan2(sin_th, cos_th)
            angle_err = angle - np.pi / 2
            angle_err = (angle_err + np.pi) % (2 * np.pi) - np.pi
            u = -self.k_stable * angle_err - self.k_damp * thd
        else:
            u = self.k_swing * (E - E_des) * thd
        return np.clip(u, -2.0, 2.0)


# ═══════════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════════


def train_kan_mse(x_train, y_train, x_val, y_val, epochs=2400, lr=1e-2):
    """Train KAN with pure MSE baseline (for ablation comparison)."""
    model = KAN([4, 12, 3], grid_size=5, spline_order=3)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=600, gamma=0.5)
    mse_fn = nn.MSELoss()

    best_val = float('inf')
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        pred = model(x_train)
        loss = mse_fn(pred, y_train)
        loss.backward()
        opt.step()
        scheduler.step()

        if epoch % 400 == 0:
            model.eval()
            with torch.no_grad():
                val_mse = mse_fn(model(x_val), y_val).item()
            if val_mse < best_val:
                best_val = val_mse
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            print(f"  KAN-MSE epoch {epoch:4d}  val_mse={val_mse:.6f}  best={best_val:.6f}")

    model.load_state_dict(best_state)
    model.eval()
    print(f"  KAN-MSE final val_mse={best_val:.8f}  params={sum(p.numel() for p in model.parameters())}")
    return model


def train_kan_cws(x_train, y_train, x_val, y_val, epochs=2400, lr=1e-2,
                   nu_cws=1.0):
    """Train KAN with CWS only — Jacobian matching, NO smoothness constraint.

    CWS: Controllability-Weighted Sobolev — matches ∂f/∂a to true Jacobian.

    Key design choice: we deliberately OMIT MOPS (P-spline penalty).
    MOPS enforces smoothness across adjacent control points, which:
      - Helps offline accuracy (when combined with CWS)
      - But IMPAIRS online adaptation (smoothness = control point coupling)

    CWS-only preserves:
      - High Jacobian accuracy (cos_sim ≈ 0.98)
      - Free control points (no cross-coupling → local adaptation works)

    From FITTING_DEPTH.md: CWS ν=1.0 achieves val_mse ≈ 0.00034,
    Jacobian cos_sim ≈ 0.98.

    Uses mini-batches (per-sample autograd Jacobian is memory-heavy).
    """
    from kanrf import jacobian_loss

    model = KAN([4, 12, 3], grid_size=5, spline_order=3)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=600, gamma=0.5)
    mse_fn = nn.MSELoss()

    # CWS weights: θ̇_dot is directly actuated → higher weight
    w_cws = torch.tensor([1.0, 1.0, 3.0])

    n_train = len(x_train)
    batch_size = 2048

    best_val = float('inf')
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()

        idx = torch.randperm(n_train)[:batch_size]
        x_batch = x_train[idx]
        y_batch = y_train[idx]
        s_batch = x_batch[:, :3]
        a_batch = x_batch[:, 3:4]

        opt.zero_grad()
        pred = model(x_batch)
        loss_mse = mse_fn(pred, y_batch)
        loss_cws = nu_cws * jacobian_loss(model, s_batch, a_batch, y_batch, w_cws)
        loss = loss_mse + loss_cws
        loss.backward()
        opt.step()
        scheduler.step()

        if epoch % 400 == 0:
            model.eval()
            with torch.no_grad():
                val_mse = mse_fn(model(x_val), y_val).item()
            if val_mse < best_val:
                best_val = val_mse
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            print(f"  KAN-CWS epoch {epoch:4d}  val_mse={val_mse:.6f}  best={best_val:.6f}  "
                  f"mse={loss_mse.item():.6f} cws={loss_cws.item():.6f}")

    model.load_state_dict(best_state)
    model.eval()
    print(f"  KAN-CWS final val_mse={best_val:.8f}  params={sum(p.numel() for p in model.parameters())}")
    return model


def train_mlp(x_train, y_train, x_val, y_val, epochs=1200, lr=1e-2):
    """Train MLP world model."""
    model = MLPWorldModel()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=400, gamma=0.5)
    mse_fn = nn.MSELoss()

    best_val = float('inf')
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        pred = model(x_train)
        loss = mse_fn(pred, y_train)
        loss.backward()
        opt.step()
        scheduler.step()

        if epoch % 200 == 0:
            model.eval()
            with torch.no_grad():
                val_mse = mse_fn(model(x_val), y_val).item()
            if val_mse < best_val:
                best_val = val_mse
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            print(f"  MLP  epoch {epoch:4d}  val_mse={val_mse:.6f}  best={best_val:.6f}")

    model.load_state_dict(best_state)
    model.eval()
    print(f"  MLP  final val_mse={best_val:.8f}  params={sum(p.numel() for p in model.parameters())}")
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# Online Updaters
# ═══════════════════════════════════════════════════════════════════════════════


class KANOnlineUpdater:
    """Three-factor online update for KAN (from control/online_learning_v2.py)."""

    def __init__(self, model, x_train, y_train, eta0=1e-3):
        from control.online_learning_v2 import ThreeFactorUpdater, compute_training_stats

        n_stats = min(len(x_train), 5000)
        idx = torch.randperm(len(x_train))[:n_stats]
        stats = compute_training_stats(model, x_train[idx], y_train[idx])
        self.updater = ThreeFactorUpdater(model, stats, eta0=eta0)
        self.model = model

    def update(self, s_norm, a_norm, s_true_norm):
        return self.updater.update(s_norm, a_norm, s_true_norm)


class KANConstantLRUpdater:
    """KAN with constant learning rate — ABLATION to isolate three-factor benefit.

    Uses the same SGD+Momentum+ReplayBuffer approach as MLP, but applied to
    KAN's parameters.  This lets us answer: is KAN's advantage from (a) the
    B-spline architecture, or (b) the three-factor learning rate?
    """

    def __init__(self, model, buffer_size=500, lr=1e-3, momentum=0.9):
        self.model = model
        self.lr = lr
        self.momentum = momentum
        self.buffer_size = buffer_size
        self.buffer_x = []
        self.buffer_y = []
        self.velocities = {name: torch.zeros_like(p)
                          for name, p in model.named_parameters()}

    def update(self, s_norm, a_norm, s_true_norm):
        x = torch.cat([s_norm, a_norm], dim=-1)
        self.buffer_x.append(x.detach().clone())
        self.buffer_y.append(s_true_norm.detach().clone())
        if len(self.buffer_x) > self.buffer_size:
            self.buffer_x.pop(0)
            self.buffer_y.pop(0)

        batch_size = min(32, len(self.buffer_x))
        idx = np.random.choice(len(self.buffer_x), batch_size, replace=False)
        x_batch = torch.cat([self.buffer_x[i] for i in idx], dim=0)
        y_batch = torch.cat([self.buffer_y[i] for i in idx], dim=0)

        self.model.train()
        pred = self.model(x_batch)
        loss = nn.functional.mse_loss(pred, y_batch)
        loss.backward()

        with torch.no_grad():
            for name, p in self.model.named_parameters():
                if p.grad is not None:
                    v = self.velocities[name]
                    v.mul_(self.momentum).add_(p.grad, alpha=self.lr)
                    p.sub_(v)
                    p.grad.zero_()

        self.model.eval()
        with torch.no_grad():
            error = (self.model(x) - s_true_norm).norm().item()
        return error, self.lr


class KANDiffusionUpdater(KANConstantLRUpdater):
    """KAN updater with control-point gradient diffusion along k-axis.

    For each edge (i,j), applies anisotropic diffusion to the raw gradient
    before momentum accumulation.  The diffusion preserves sharp features
    (large |c_{k+1}-c_k| → weak diffusion) while smoothing updates in flat
    regions (small |c_{k+1}-c_k| → strong diffusion).

    Args:
        model: KAN world model
        lam: diffusion strength (0 = no diffusion, same as KANConstantLRUpdater)
        beta: trend sensitivity (0=uniform, 1=linear, 2=strong protection)
        alpha: noise floor for weight computation
        T: number of diffusion iterations per update
        buffer_size, lr, momentum: same as KANConstantLRUpdater
    """

    def __init__(self, model, lam=0.5, beta=1, alpha=0.01, T=2,
                 buffer_size=500, lr=1e-3, momentum=0.9):
        super().__init__(model, buffer_size=buffer_size, lr=lr, momentum=momentum)
        self.lam = lam
        self.beta = beta
        self.alpha = alpha
        self.T = T

    def _diffuse_spline_grad(self):
        """Apply gradient diffusion to all spline_weight parameters in-place."""
        if self.lam <= 0:
            return

        eps = self.lam * self.lr / (1.0 + self.lam * self.lr)  # ∈ (0, 1)

        for layer in self.model.layers:
            c = layer.spline_weight.data  # (out_dim, in_dim, n_basis)
            g = layer.spline_weight.grad  # same shape
            if g is None:
                continue

            out_dim, in_dim, n = c.shape
            if n < 3:
                continue

            # --- Per-edge diffusion ---
            for i in range(out_dim):
                for j in range(in_dim):
                    c_ij = c[i, j, :]  # (n,)
                    g_ij = g[i, j, :]  # (n,)

                    # Step 1: trend-sensitive weights w_k for k=1..n-2
                    # w_k connects position k and k+1
                    dc = c_ij[1:] - c_ij[:-1]  # (n-1,), dc[k] = c[k+1] - c[k]
                    w = 1.0 / (self.alpha + dc.abs().pow(self.beta))  # (n-1,)
                    # w[k] controls diffusion between k and k+1

                    # Step 2: start from raw update
                    # (the actual update = -lr * g, we diffuse g directly)
                    dc_tilde = g_ij.clone()  # (n,), proxy for the update direction

                    # Step 3: T rounds of diffusion
                    for _ in range(self.T):
                        diff = dc_tilde[1:] - dc_tilde[:-1]  # (n-1,)
                        flux = w * diff                     # (n-1,)
                        dc_tilde[:-1] += eps * flux
                        dc_tilde[1:]  -= eps * flux

                    g[i, j, :] = dc_tilde

    def update(self, s_norm, a_norm, s_true_norm):
        """Same as KANConstantLRUpdater but with gradient diffusion."""
        x = torch.cat([s_norm, a_norm], dim=-1)
        self.buffer_x.append(x.detach().clone())
        self.buffer_y.append(s_true_norm.detach().clone())
        if len(self.buffer_x) > self.buffer_size:
            self.buffer_x.pop(0)
            self.buffer_y.pop(0)

        batch_size = min(32, len(self.buffer_x))
        idx = np.random.choice(len(self.buffer_x), batch_size, replace=False)
        x_batch = torch.cat([self.buffer_x[i] for i in idx], dim=0)
        y_batch = torch.cat([self.buffer_y[i] for i in idx], dim=0)

        self.model.train()
        pred = self.model(x_batch)
        loss = nn.functional.mse_loss(pred, y_batch)
        loss.backward()

        # ── Apply gradient diffusion BEFORE momentum/update ──
        self._diffuse_spline_grad()

        with torch.no_grad():
            for name, p in self.model.named_parameters():
                if p.grad is not None:
                    v = self.velocities[name]
                    v.mul_(self.momentum).add_(p.grad, alpha=self.lr)
                    p.sub_(v)
                    p.grad.zero_()

        self.model.eval()
        with torch.no_grad():
            error = (self.model(x) - s_true_norm).norm().item()
        return error, self.lr


class MLPOnlineUpdater:
    """Standard SGD with momentum for MLP online learning.

    Uses a small replay buffer to prevent catastrophic forgetting.
    This is a FAIR baseline — the standard MBRL approach for MLP adaptation.
    """

    def __init__(self, model, buffer_size=500, lr=1e-3, momentum=0.9):
        self.model = model
        self.lr = lr
        self.momentum = momentum
        self.buffer_size = buffer_size
        self.buffer_x = []
        self.buffer_y = []

        # Per-parameter momentum buffers
        self.velocities = {name: torch.zeros_like(p)
                          for name, p in model.named_parameters()}

    def update(self, s_norm, a_norm, s_true_norm):
        """One SGD step with replay buffer."""
        x = torch.cat([s_norm, a_norm], dim=-1)

        # Add to buffer
        self.buffer_x.append(x.detach().clone())
        self.buffer_y.append(s_true_norm.detach().clone())
        if len(self.buffer_x) > self.buffer_size:
            self.buffer_x.pop(0)
            self.buffer_y.pop(0)

        # Sample mini-batch from buffer
        batch_size = min(32, len(self.buffer_x))
        idx = np.random.choice(len(self.buffer_x), batch_size, replace=False)
        x_batch = torch.cat([self.buffer_x[i] for i in idx], dim=0)
        y_batch = torch.cat([self.buffer_y[i] for i in idx], dim=0)

        self.model.train()
        pred = self.model(x_batch)
        loss = nn.functional.mse_loss(pred, y_batch)
        loss.backward()

        # Manual SGD + momentum (no optimizer object for simplicity)
        with torch.no_grad():
            for name, p in self.model.named_parameters():
                if p.grad is not None:
                    v = self.velocities[name]
                    v.mul_(self.momentum).add_(p.grad, alpha=self.lr)
                    p.sub_(v)
                    p.grad.zero_()

        self.model.eval()
        with torch.no_grad():
            error = (self.model(x) - s_true_norm).norm().item()
        return error, self.lr


# ═══════════════════════════════════════════════════════════════════════════════
# Main Experiment
# ═══════════════════════════════════════════════════════════════════════════════


def normalize_state(obs):
    """obs[cos, sin, thd] → normalized [cos, sin, thd/8]."""
    s = obs.copy() if isinstance(obs, np.ndarray) else np.array(obs, dtype=np.float32)
    s[2] /= 8.0
    return s


def normalize_action(a):
    """a ∈ [-2, 2] → a_norm ∈ [-1, 1]."""
    return a / 2.0


def denormalize_action(a_norm):
    return a_norm * 2.0


def prediction_error(model, s_norm, a_norm, s_true_norm, device='cpu'):
    """Compute L2 prediction error in raw state space."""
    x = torch.cat([
        torch.tensor(s_norm, dtype=torch.float32).unsqueeze(0),
        torch.tensor([a_norm], dtype=torch.float32).unsqueeze(0)
    ], dim=-1).to(device)
    with torch.no_grad():
        pred = model(x).cpu().squeeze().numpy()
    # Error in raw space: denormalize theta_dot
    pred_raw = pred.copy()
    pred_raw[2] *= 8.0
    true_raw = s_true_norm.copy()
    true_raw[2] *= 8.0
    return np.linalg.norm(pred_raw - true_raw)


def run_experiment(model, updater, env, controller, device,
                   n_steps=800, gravity_schedule=None):
    """Run continual learning experiment.

    Args:
        model: world model (KAN or MLP)
        updater: online updater
        env: ConfigurablePendulum
        controller: EnergyController
        device: torch device
        n_steps: total steps
        gravity_schedule: list of (step_idx, gravity) to switch at

    Returns:
        dict with step_errors, gravity_changes, model_name
    """
    if gravity_schedule is None:
        gravity_schedule = [
            (0, 10.0),      # default
            (150, 18.0),    # heavy gravity (80% increase)
            (350, 10.0),    # back to default (test against forgetting)
            (550, 3.0),     # light gravity (70% decrease)
        ]

    model.eval()
    obs = env.reset(seed=42)
    errors = []
    raw_states = []  # for diagnostics
    current_g = 10.0

    # Apply initial gravity
    for step_idx, g_val in sorted(gravity_schedule):
        if 0 <= step_idx <= 0:
            env.set_g(g_val)
            current_g = g_val

    next_switch_idx = 1

    print(f"\n  Running {n_steps} steps...")
    print(f"  Gravity schedule: {gravity_schedule}")
    t_start = time.time()

    for step in range(n_steps):
        # Check for gravity switch
        while (next_switch_idx < len(gravity_schedule) and
               step >= gravity_schedule[next_switch_idx][0]):
            _, new_g = gravity_schedule[next_switch_idx]
            env.set_g(new_g)
            print(f"  Step {step}: gravity {current_g:.1f} → {new_g:.1f}")
            current_g = new_g
            next_switch_idx += 1

        # Get action from energy controller (uses current g)
        a = controller.get_action(obs, g=current_g)

        # Execute in environment
        obs_next, _, term, trunc = env.step(a)

        # Normalize
        s_norm = normalize_state(obs)
        a_norm = normalize_action(a)
        s_true_norm = normalize_state(obs_next)

        # Compute prediction error BEFORE update
        err = prediction_error(model, s_norm, a_norm, s_true_norm, device=device)

        # Online update
        s_norm_t = torch.tensor(s_norm, dtype=torch.float32).unsqueeze(0).to(device)
        a_norm_t = torch.tensor([[a_norm]], dtype=torch.float32).to(device)
        s_true_norm_t = torch.tensor(s_true_norm, dtype=torch.float32).unsqueeze(0).to(device)
        updater.update(s_norm_t, a_norm_t, s_true_norm_t)

        errors.append(err)
        raw_states.append(obs.copy())

        # Handle episode end
        if term or trunc:
            obs = env.reset()
        else:
            obs = obs_next

        if (step + 1) % 200 == 0:
            elapsed = time.time() - t_start
            recent_err = np.mean(errors[-100:])
            print(f"  Step {step+1:4d}/{n_steps}  recent_err={recent_err:.4f}  "
                  f"g={current_g:.1f}  [{elapsed:.0f}s]")

    elapsed = time.time() - t_start
    print(f"  Done in {elapsed:.0f}s  ({elapsed/n_steps*1000:.1f}ms/step)")

    return {
        'errors': np.array(errors),
        'gravity_changes': gravity_schedule,
        'raw_states': raw_states,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════════════════


def plot_results(results_dict, save_path=None):
    """Plot absolute AND relative prediction error over time."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n_models = len(results_dict)
    colors = {
        'KAN-CWS (no diffusion)': '#1565C0',
        'KAN-CWS (uniform diff β=0)': '#4CAF50',
        'KAN-CWS (trend diff β=1)': '#E91E63',
        'MLP (SGD+replay)': '#FF9800',
    }

    # ═══ Figure 1: Absolute error (separate subplots, shared x) ═══
    fig1, axes1 = plt.subplots(n_models, 1, figsize=(14, 3.5 * n_models), sharex=True)
    if n_models == 1:
        axes1 = [axes1]

    for ax, (name, result) in zip(axes1, results_dict.items()):
        errors = result['errors']
        steps = np.arange(len(errors))
        color = colors.get(name, '#333333')

        window = 20
        running_mean = np.convolve(errors, np.ones(window)/window, mode='same')
        running_mean[:window//2] = errors[:window//2]
        running_mean[-window//2:] = errors[-window//2:]

        ax.plot(steps, errors, alpha=0.12, color=color, linewidth=0.5)
        ax.plot(steps, running_mean, color=color, linewidth=2, label=name)

        for step_idx, g_val in result['gravity_changes']:
            if step_idx >= len(errors):
                continue
            ax.axvline(x=step_idx, color='red', linestyle='--', alpha=0.5, linewidth=1)
            ymax = ax.get_ylim()[1]
            ax.text(step_idx + 5, ymax * 0.92, f'g={g_val:.1f}',
                    fontsize=8, color='red', verticalalignment='top')

        ax.set_ylabel('L2 Error (raw)')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    axes1[-1].set_xlabel('Environment Step')
    fig1.suptitle('Continual Learning under Gravity Changes — Absolute Prediction Error',
                  fontsize=13, fontweight='bold')
    plt.tight_layout()

    abs_path = save_path.replace('.png', '_abs.png') if save_path else 'continual_learning_abs.png'
    fig1.savefig(abs_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {abs_path}")
    plt.close(fig1)

    # ═══ Figure 2: Relative error (all models on SAME axes for direct comparison) ═══
    fig2, (ax_rel, ax_summary) = plt.subplots(2, 1, figsize=(14, 8),
                                               gridspec_kw={'height_ratios': [3, 1]})

    for name, result in results_dict.items():
        errors = result['errors']
        steps = np.arange(len(errors))
        color = colors.get(name, '#333333')

        # Baseline: mean error in last 100 steps of first g=10.0 segment
        g_changes = result['gravity_changes']
        baseline_end = min(g_changes[1][0] if len(g_changes) > 1 else len(errors), len(errors))
        baseline_err = np.mean(errors[max(50, baseline_end-100):baseline_end])
        if baseline_err < 1e-6:
            baseline_err = 1.0  # fallback

        rel_errors = errors / baseline_err

        window = 20
        running_rel = np.convolve(rel_errors, np.ones(window)/window, mode='same')
        running_rel[:window//2] = rel_errors[:window//2]
        running_rel[-window//2:] = rel_errors[-window//2:]

        ax_rel.plot(steps, running_rel, color=color, linewidth=2, label=f'{name} (baseline={baseline_err:.4f})')

        for step_idx, g_val in result['gravity_changes']:
            if step_idx >= len(errors):
                continue
            ax_rel.axvline(x=step_idx, color='red', linestyle='--', alpha=0.4, linewidth=1)
            ax_rel.text(step_idx + 5, ax_rel.get_ylim()[1] * 0.92 if ax_rel.get_ylim()[1] > 0 else 5,
                        f'g={g_val:.1f}', fontsize=8, color='red', verticalalignment='top')

    ax_rel.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5, linewidth=1, label='baseline')
    ax_rel.set_ylabel('Error Relative to Baseline')
    ax_rel.legend(loc='upper left', fontsize=9)
    ax_rel.grid(True, alpha=0.3)
    ax_rel.set_ylim(bottom=0)
    ax_rel.set_title('Relative Prediction Error (normalized by each model\'s own g=10.0 baseline)')

    # ═══ Summary table ═══
    ax_summary.axis('off')
    headers = ['Model', 'g=10.0\nbaseline', 'g=18.0\nspike', 'g=18.0\nstable',
               'g=10.0\nreturn', 'g=3.0\nspike', 'g=3.0\nstable', 'Forget?']
    n_cols = len(headers)
    table_data = [headers]

    for name, result in results_dict.items():
        errors = result['errors']
        g_changes = result['gravity_changes'] + [(len(errors), None)]
        be = np.mean(errors[50:g_changes[1][0]])
        row = [name, f'{be:.3f}']

        # g=18.0 spike + stable
        if len(g_changes) > 2 and g_changes[1][0] < len(errors):
            start, end = g_changes[1][0], min(g_changes[2][0], len(errors))
            seg = errors[start:end]
            row.append(f"{np.mean(seg[:min(20,len(seg))]):.3f} ({np.mean(seg[:min(20,len(seg))])/(be+1e-8):.1f}x)")
            ss = min(30, len(seg))
            row.append(f"{np.mean(seg[ss:]):.3f} ({np.mean(seg[ss:])/(be+1e-8):.1f}x)" if ss < len(seg) else '-')
        else:
            row.extend(['-', '-'])

        # g=10.0 return
        if len(g_changes) > 3 and g_changes[2][0] < len(errors):
            start, end = g_changes[2][0] + 30, min(g_changes[3][0], len(errors))
            row.append(f"{np.mean(errors[start:end]):.3f} ({np.mean(errors[start:end])/(be+1e-8):.1f}x)" if end > start else '-')
        else:
            row.append('-')

        # g=3.0 spike + stable
        if len(g_changes) > 4 and g_changes[3][0] < len(errors):
            start, end = g_changes[3][0], min(g_changes[4][0], len(errors))
            seg = errors[start:end]
            row.append(f"{np.mean(seg[:min(20,len(seg))]):.3f} ({np.mean(seg[:min(20,len(seg))])/(be+1e-8):.1f}x)")
            ss = min(30, len(seg))
            row.append(f"{np.mean(seg[ss:]):.3f} ({np.mean(seg[ss:])/(be+1e-8):.1f}x)" if ss < len(seg) else '-')
        else:
            row.extend(['-', '-'])

        # Forgetting
        if len(g_changes) >= 4:
            rs, re = g_changes[2][0] + 30, min(g_changes[3][0], len(errors))
            ratio = np.mean(errors[rs:re])/(be+1e-8) if re > rs else 999
            row.append(f'{"✓" if ratio < 1.5 else "✗"} ({ratio:.1f}x)')
        else:
            row.append('-')

        while len(row) < n_cols:
            row.append('-')
        table_data.append(row)

    table = ax_summary.table(cellText=table_data, cellLoc='center', loc='center',
                             colWidths=[0.18, 0.1, 0.13, 0.13, 0.13, 0.13, 0.13, 0.07])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)

    plt.tight_layout()
    rel_path = save_path.replace('.png', '_rel.png') if save_path else 'continual_learning_rel.png'
    fig2.savefig(rel_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {rel_path}")
    plt.close(fig2)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description='Continual Learning: KAN vs MLP under physics parameter changes')
    parser.add_argument('--steps', type=int, default=800,
                       help='Total environment steps')
    parser.add_argument('--epochs', type=int, default=1200,
                       help='Pre-training epochs')
    parser.add_argument('--no-train', action='store_true',
                       help='Skip training, load cached models')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'mps', 'cuda'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default='continual_learning_results.png')
    parser.add_argument('--cache-dir', type=str, default='/tmp/kanrf_cl_exp')
    args = parser.parse_args()

    device = torch.device(args.device)
    if args.device == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
    elif args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.cache_dir, exist_ok=True)
    data_path = os.path.join(args.cache_dir, 'pendulum_data_cl.pt')
    kan_cws_path = os.path.join(args.cache_dir, 'kan_cws_cl.pt')
    kan_mse_path = os.path.join(args.cache_dir, 'kan_mse_cl.pt')
    mlp_path = os.path.join(args.cache_dir, 'mlp_wm_cl.pt')

    # ══════════════════════════════════════════════════════════════════
    # Phase 1: Generate training data (default g=10.0)
    # ══════════════════════════════════════════════════════════════════
    if args.no_train and os.path.exists(data_path):
        print("Loading cached training data...")
        x_train, y_train, x_val, y_val = torch.load(data_path, weights_only=True)
    else:
        print("=" * 60)
        print("Phase 1: Generating training data (g=10.0)")
        print("=" * 60)
        env_data = ConfigurablePendulum(g=10.0, seed=args.seed)
        x, y = generate_pendulum_data(env_data, n_transitions=20000)
        env_data.env.close()

        n_train = int(len(x) * 0.85)
        x_train, y_train = x[:n_train], y[:n_train]
        x_val, y_val = x[n_train:], y[n_train:]
        print(f"  Train: {x_train.shape}, Val: {x_val.shape}")

        torch.save((x_train, y_train, x_val, y_val), data_path)
        print(f"  Saved: {data_path}")

    # Move data to device
    x_train, y_train = x_train.to(device), y_train.to(device)
    x_val, y_val = x_val.to(device), y_val.to(device)

    # ══════════════════════════════════════════════════════════════════
    # Phase 2: Train world models
    # ══════════════════════════════════════════════════════════════════
    if args.no_train and os.path.exists(kan_cws_path) and os.path.exists(mlp_path):
        print("\nLoading cached models...")
        kan_cws_model = KAN([4, 12, 3], grid_size=5, spline_order=3)
        kan_cws_model.load_state_dict(torch.load(kan_cws_path, weights_only=True))
        kan_cws_model.to(device)
        kan_cws_model.eval()
        print(f"  KAN-CWS loaded: {sum(p.numel() for p in kan_cws_model.parameters())} params")

        kan_mse_model = KAN([4, 12, 3], grid_size=5, spline_order=3)
        if os.path.exists(kan_mse_path):
            kan_mse_model.load_state_dict(torch.load(kan_mse_path, weights_only=True))
            kan_mse_model.to(device)
            kan_mse_model.eval()
            print(f"  KAN-MSE loaded: {sum(p.numel() for p in kan_mse_model.parameters())} params")
        else:
            kan_mse_model = None

        mlp_model = MLPWorldModel()
        mlp_model.load_state_dict(torch.load(mlp_path, weights_only=True))
        mlp_model.to(device)
        mlp_model.eval()
        print(f"  MLP loaded: {sum(p.numel() for p in mlp_model.parameters())} params")
    else:
        print("\n" + "=" * 60)
        print("Phase 2: Training world models")
        print("=" * 60)

        print("\n[KAN-CWS] Training (Jacobian matching, no smoothness constraint)...")
        kan_cws_model = train_kan_cws(x_train, y_train, x_val, y_val, epochs=args.epochs)
        kan_cws_model.to(device)
        torch.save(kan_cws_model.state_dict(), kan_cws_path)

        print("\n[KAN-MSE] Training (pure MSE, for ablation)...")
        kan_mse_model = train_kan_mse(x_train, y_train, x_val, y_val, epochs=args.epochs)
        kan_mse_model.to(device)
        torch.save(kan_mse_model.state_dict(), kan_mse_path)

        print("\n[MLP] Training...")
        mlp_model = train_mlp(x_train, y_train, x_val, y_val, epochs=args.epochs)
        mlp_model.to(device)
        torch.save(mlp_model.state_dict(), mlp_path)

    # ══════════════════════════════════════════════════════════════════
    # Phase 3: Continual Learning Experiment
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("Phase 3: Continual Learning Experiment")
    print("=" * 60)

    # Gravity schedule: step → gravity value
    gravity_schedule = [
        (0, 10.0),       # default
        (200, 18.0),     # heavy gravity
        (400, 10.0),     # BACK to default (test: no catastrophic forgetting?)
        (600, 3.0),      # light gravity
    ]

    results = {}
    ctrl = EnergyController()

    def make_kan_cws():
        m = KAN([4, 12, 3], grid_size=5, spline_order=3)
        m.load_state_dict(kan_cws_model.state_dict())
        m.to(device)
        m.eval()
        return m

    # ── Experiment 1: Baseline (no diffusion) ──
    print("\n── Experiment 1: KAN-CWS + constant LR (baseline, no diffusion) ──")
    env1 = ConfigurablePendulum(g=10.0, seed=args.seed)
    m1 = make_kan_cws()
    up1 = KANConstantLRUpdater(m1, buffer_size=500, lr=1e-3, momentum=0.9)
    r1 = run_experiment(m1, up1, env1, ctrl, device, n_steps=args.steps,
                        gravity_schedule=gravity_schedule)
    env1.env.close()
    r1['model_name'] = 'KAN-CWS (no diffusion)'
    results['KAN-CWS (no diffusion)'] = r1

    # ── Experiment 2: Uniform diffusion (β=0) ──
    print("\n── Experiment 2: KAN-CWS + uniform diffusion (β=0, λ=0.5) ──")
    env2 = ConfigurablePendulum(g=10.0, seed=args.seed)
    m2 = make_kan_cws()
    up2 = KANDiffusionUpdater(m2, lam=0.5, beta=0, alpha=0.01, T=2,
                              buffer_size=500, lr=1e-3, momentum=0.9)
    r2 = run_experiment(m2, up2, env2, ctrl, device, n_steps=args.steps,
                        gravity_schedule=gravity_schedule)
    env2.env.close()
    r2['model_name'] = 'KAN-CWS (uniform diff β=0)'
    results['KAN-CWS (uniform diff β=0)'] = r2

    # ── Experiment 3: Trend-guided diffusion (β=1) ──
    print("\n── Experiment 3: KAN-CWS + trend diffusion (β=1, λ=0.5) ──")
    env3 = ConfigurablePendulum(g=10.0, seed=args.seed)
    m3 = make_kan_cws()
    up3 = KANDiffusionUpdater(m3, lam=0.5, beta=1, alpha=0.01, T=2,
                              buffer_size=500, lr=1e-3, momentum=0.9)
    r3 = run_experiment(m3, up3, env3, ctrl, device, n_steps=args.steps,
                        gravity_schedule=gravity_schedule)
    env3.env.close()
    r3['model_name'] = 'KAN-CWS (trend diff β=1)'
    results['KAN-CWS (trend diff β=1)'] = r3

    # ── Experiment 4: MLP + SGD + replay (reference) ──
    print("\n── Experiment 4: MLP (SGD + replay buffer) ──")
    env_mlp = ConfigurablePendulum(g=10.0, seed=args.seed)
    mlp_exp = MLPWorldModel()
    mlp_exp.load_state_dict(mlp_model.state_dict())
    mlp_exp.to(device)
    mlp_exp.eval()
    mlp_updater = MLPOnlineUpdater(mlp_exp, buffer_size=500, lr=1e-3, momentum=0.9)
    result_mlp = run_experiment(mlp_exp, mlp_updater, env_mlp, ctrl,
                                device, n_steps=args.steps,
                                gravity_schedule=gravity_schedule)
    env_mlp.env.close()
    result_mlp['model_name'] = 'MLP (SGD+replay)'
    results['MLP (SGD+replay)'] = result_mlp

    # ══════════════════════════════════════════════════════════════════
    # Phase 4: Analysis & Visualization
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("Phase 4: Analysis")
    print("=" * 60)

    for name, result in results.items():
        errors = result['errors']
        gravity_changes = result['gravity_changes'] + [(len(errors), None)]
        print(f"\n{name}:")
        print(f"  Overall mean error: {np.mean(errors):.4f}")
        print(f"  Overall std error:  {np.std(errors):.4f}")

        # Compute baseline error (last 100 steps of original g=10.0 segment)
        baseline_start = max(50, gravity_changes[0][0])
        baseline_end = min(gravity_changes[1][0], len(errors))
        baseline_err = np.mean(errors[baseline_start:baseline_end])
        print(f"  Baseline error (g=10.0): {baseline_err:.4f}")

        for i in range(len(gravity_changes) - 1):
            start = gravity_changes[i][0]
            end = min(gravity_changes[i+1][0], len(errors))
            g_val = gravity_changes[i][1]
            if start >= len(errors):
                continue
            seg = errors[start:end]
            # First 20 steps after change vs stable (after 30 steps)
            first_20 = np.mean(seg[:min(20, len(seg))])
            rest_start = min(30, len(seg))
            rest = np.mean(seg[rest_start:]) if rest_start < len(seg) else first_20
            rel_first = first_20 / (baseline_err + 1e-8)
            rel_rest = rest / (baseline_err + 1e-8)
            print(f"  g={g_val:.1f} [{start}:{end}]: "
                  f"first20={first_20:.4f} ({rel_first:.1f}x) "
                  f"stable={rest:.4f} ({rel_rest:.1f}x)")

        # Key metric: error when returning to g=10.0
        if len(gravity_changes) >= 3 and len(errors) > gravity_changes[2][0]:
            # Segment 3 is the return to g=10.0
            ret_start = gravity_changes[2][0]
            ret_end = min(gravity_changes[3][0] if len(gravity_changes) > 3 else len(errors), len(errors))
            return_g10 = np.mean(errors[ret_start+30:ret_end]) if ret_end > ret_start+30 else np.mean(errors[ret_start:ret_end])
            rel_return = return_g10 / (baseline_err + 1e-8)
            print(f"  g=10.0 original: {baseline_err:.4f}")
            print(f"  g=10.0 return:   {return_g10:.4f}  ({rel_return:.1f}x baseline)")
            if rel_return < 1.5:
                print(f"  ✓ No catastrophic forgetting!")
            else:
                print(f"  ✗ Catastrophic forgetting detected (>{1.5}x baseline)!")

    print("\nPlotting...")
    plot_results(results, save_path=args.output)

    # Also save raw data for later analysis
    keys = list(results.keys())
    np.savez(os.path.join(args.cache_dir, 'results.npz'),
             **{f'err_{k}': results[k]['errors'] for k in keys})
    print(f"Raw data saved: {os.path.join(args.cache_dir, 'results.npz')}")

    print("\nDone!")


if __name__ == '__main__':
    main()
