"""CartPole decision_v3: compare single-step vs sequence policy.

CartPole is a good test because:
  1. No easy energy-loss shortcut (unlike Pendulum)
  2. Needs multi-step coordination: swing up + center cart + stabilize
  3. Action has subtle effect → single-step Jacobian is small

State (4D): [x, x_dot, theta, theta_dot], all normalized to [-1, 1]
Action: scalar ∈ [-1, 1] → force ∈ [-10, 10]
World model: KAN([5, 12, 4])
Target: [0, 0, 0, 0] (centered, upright, stationary)
"""
import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
import time, argparse, sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kanrf import KAN

# ═══════════════════════════════════════════════════════════════════════════════
# CartPole Dynamics (analytical, for CWS Jacobian)
# ═══════════════════════════════════════════════════════════════════════════════
G = 9.8; MC = 1.0; MP = 0.1; L = 0.5; DT = 0.02
TOTAL_MASS = MC + MP; POLE_MASS_LEN = MP * L

# Normalization constants
X_SCALE = 2.5; XD_SCALE = 3.0; TH_SCALE = 0.3; THD_SCALE = 3.0
FORCE_MAX = 10.0


def cartpole_true_jacobian(s_next_norm):
    """Analytic Jacobian ds'_norm / da_norm for CartPole (semi-implicit Euler).

    Semi-implicit: v_new = v + a*dt, x_new = x + v_new*dt
    So d(x_new)/d(force) = d(x_dot_new)/d(force) * DT ≈ DT^2 * d(x_acc)/d(force)

    Returns J: (..., 4) in normalized units.
    """
    theta = s_next_norm[..., 2] * TH_SCALE  # denormalize
    costheta = torch.cos(theta)

    denom = L * (4.0/3.0 - MP * costheta**2 / TOTAL_MASS)
    # d(theta_acc)/d(force) = -costheta / (denom * TOTAL_MASS)
    # NEGATIVE: pushing cart right → reaction force tips pole LEFT
    d_theta_acc_df = -costheta / (denom * TOTAL_MASS)
    d_x_acc_df = 1.0 / TOTAL_MASS + POLE_MASS_LEN * costheta * d_theta_acc_df / TOTAL_MASS

    da_raw = FORCE_MAX  # d(force)/d(a_norm)

    # Semi-implicit Euler derivatives:
    # theta_dot_new = theta_dot + theta_acc*DT → d(theta_dot')/d(force) = DT * d_theta_acc
    # theta_new = theta + theta_dot_new*DT → d(theta')/d(force) = DT * d(theta_dot')/d(force) = DT^2 * d_theta_acc
    J_thd = DT * d_theta_acc_df * da_raw / THD_SCALE
    J_th = DT * J_thd * THD_SCALE / TH_SCALE  # = DT^2 * d_theta_acc_df * da_raw / TH_SCALE

    J_xd = DT * d_x_acc_df * da_raw / XD_SCALE
    J_x = DT * J_xd * XD_SCALE / X_SCALE  # = DT^2 * d_x_acc_df * da_raw / X_SCALE

    return torch.stack([J_x, J_xd, J_th, J_thd], dim=-1)


# ═══════════════════════════════════════════════════════════════════════════════
# Data Generation
# ═══════════════════════════════════════════════════════════════════════════════

def generate_data(n=20000):
    """Use analytical CartPole dynamics (supports continuous actions)."""
    torch.manual_seed(42)

    # Sample random states in CartPole valid range
    states_raw = torch.rand(n, 4)
    states_raw[:, 0] = states_raw[:, 0] * 4.8 - 2.4      # x ∈ [-2.4, 2.4]
    states_raw[:, 1] = states_raw[:, 1] * 6.0 - 3.0      # x_dot ∈ [-3, 3]
    states_raw[:, 2] = states_raw[:, 2] * 0.5 - 0.25     # theta ∈ [-0.25, 0.25]
    states_raw[:, 3] = states_raw[:, 3] * 6.0 - 3.0      # theta_dot ∈ [-3, 3]
    actions_raw = torch.rand(n, 1) * 2 - 1                 # a_norm ∈ [-1, 1]

    # Use analytical dynamics
    s_next_raw = step_cartpole_cont(states_raw, actions_raw.squeeze())

    # Normalize
    s_norm = states_raw.clone()
    s_norm[:, 0] /= X_SCALE; s_norm[:, 1] /= XD_SCALE
    s_norm[:, 2] /= TH_SCALE; s_norm[:, 3] /= THD_SCALE

    s_next_norm = s_next_raw.clone()
    s_next_norm[:, 0] /= X_SCALE; s_next_norm[:, 1] /= XD_SCALE
    s_next_norm[:, 2] /= TH_SCALE; s_next_norm[:, 3] /= THD_SCALE

    x = torch.cat([s_norm, actions_raw], dim=-1)  # (N, 5)
    y = s_next_norm

    n_train = int(n * 0.85)
    return (x[:n_train], y[:n_train], x[n_train:], y[n_train:]), x.shape[1], y.shape[1]


def step_cartpole_cont(state, a_norm):
    """Single CartPole step — semi-implicit Euler (velocity first, then position).

    a_norm: (B,) ∈ [-1, 1] → force = a_norm * FORCE_MAX
    state: (B, 4) in raw units: [x, x_dot, theta, theta_dot]
    """
    force = a_norm * FORCE_MAX  # (B,)
    x, x_dot, theta, theta_dot = (state[:, i] for i in range(4))

    costheta = torch.cos(theta)
    sintheta = torch.sin(theta)

    temp = (force + POLE_MASS_LEN * theta_dot**2 * sintheta) / TOTAL_MASS
    denom = L * (4.0/3.0 - MP * costheta**2 / TOTAL_MASS)
    theta_acc = (G * sintheta - costheta * temp) / (denom + 1e-8)
    x_acc = temp - POLE_MASS_LEN * theta_acc * costheta / TOTAL_MASS

    # Semi-implicit Euler: update velocity first, then position using NEW velocity
    x_dot_new = x_dot + x_acc * DT
    theta_dot_new = theta_dot + theta_acc * DT
    x_new = x + x_dot_new * DT
    theta_new = theta + theta_dot_new * DT

    return torch.stack([x_new, x_dot_new, theta_new, theta_dot_new], dim=-1)


# ═══════════════════════════════════════════════════════════════════════════════
# KAN Training (CWS only)
# ═══════════════════════════════════════════════════════════════════════════════

def jacobian_loss_cartpole(model, s_batch, a_batch, y_batch, w):
    a = a_batch.clone().detach().requires_grad_(True)
    s_pred = model(torch.cat([s_batch, a], dim=-1))
    J_model = []
    for dim in range(4):
        g = torch.autograd.grad(s_pred[:, dim].sum(), a, retain_graph=True, create_graph=True)[0]
        J_model.append(g)
    J_model = torch.cat(J_model, dim=-1)
    J_true = cartpole_true_jacobian(y_batch)
    err = (J_model - J_true) ** 2
    return (err * w.unsqueeze(0)).mean()


def train_kan_cws(x_train, y_train, x_val, y_val, epochs=1200, nu=1.0):
    model = KAN([5, 12, 4], grid_size=5, spline_order=3)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=400, gamma=0.5)
    mse_fn = nn.MSELoss()
    w_cws = torch.tensor([0.5, 1.0, 2.0, 3.0])  # pole angle/vel more important

    n_train = len(x_train); batch_size = 2048
    best_val = float('inf'); best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        idx = torch.randperm(n_train)[:batch_size]
        xb, yb = x_train[idx], y_train[idx]
        sb = xb[:, :4]; ab = xb[:, 4:5]

        opt.zero_grad()
        loss = mse_fn(model(xb), yb)
        loss_cws = nu * jacobian_loss_cartpole(model, sb, ab, yb, w_cws)
        (loss + loss_cws).backward()
        opt.step(); scheduler.step()

        if epoch % 200 == 0:
            model.eval()
            with torch.no_grad():
                val = mse_fn(model(x_val), y_val).item()
            if val < best_val:
                best_val = val
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            print(f"  KAN epoch {epoch:4d}  val={val:.6f}  best={best_val:.6f}  mse={loss.item():.6f} cws={loss_cws.item():.6f}")

    model.load_state_dict(best_state)
    model.eval()
    print(f"  KAN final val_mse={best_val:.8f}  params={sum(p.numel() for p in model.parameters())}")
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# MLP Policy
# ═══════════════════════════════════════════════════════════════════════════════

class CartPolePolicy(nn.Module):
    def __init__(self, state_dim=4, hidden=64, n_layers=2):
        super().__init__()
        layers = []
        d = state_dim
        for _ in range(n_layers):
            layers.extend([nn.Linear(d, hidden), nn.ReLU()])
            d = hidden
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, s):
        return torch.tanh(self.net(s))  # a ∈ [-1, 1]


# ═══════════════════════════════════════════════════════════════════════════════
# Trainers
# ═══════════════════════════════════════════════════════════════════════════════

def compute_energy(s, pole_only=True):
    """Pole kinetic + potential energy (normalized space)."""
    th = s[:, 2] * TH_SCALE; thd = s[:, 3] * THD_SCALE
    E = 0.5 * MP * (L * thd)**2 + MP * G * L * (1 - torch.cos(th))
    return E / (MP * G * L + 1e-8)  # normalize to ~[0, 2]


class CartPoleSingleStepTrainer:
    """Train policy with single-step KAN evaluation."""
    def __init__(self, kan, policy, s_target, lr=1e-3, lambda_ctrl=0.01,
                 clip_grad=10.0, device='cpu', w_pole=5.0, w_cart=1.0):
        self.kan = kan; self.policy = policy.to(device)
        self.s_target = s_target.to(device); self.device = device
        self.lambda_ctrl = lambda_ctrl; self.clip_grad = clip_grad
        self.w_pole = w_pole; self.w_cart = w_cart

        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
        self.loss_history = []
        self.kan.eval()
        for p in self.kan.parameters(): p.requires_grad = False

    def train_step(self, s_batch):
        B = s_batch.shape[0]; self.policy.train(); self.optimizer.zero_grad()

        a = self.policy(s_batch)
        s_pred = self.kan(torch.cat([s_batch, a], dim=-1))

        # Direct pole-centering loss: minimize |theta'| + 0.1 * |x'|
        # This tells the policy: "your job is to keep the pole upright and cart centered"
        th_pred = s_pred[:, 2]  # normalized pole angle
        x_pred = s_pred[:, 0]   # normalized cart position
        pole_loss = th_pred.pow(2).mean()        # minimize pole angle squared
        cart_loss = 0.1 * x_pred.pow(2).mean()   # secondary: center cart
        pred_loss = pole_loss + cart_loss
        ctrl_loss = a.pow(2).mean()
        total = pred_loss + self.lambda_ctrl * ctrl_loss

        total.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.clip_grad)
        self.optimizer.step()

        self.loss_history.append({
            'total': total.item(), 'pred': pred_loss.item(), 'ctrl': ctrl_loss.item()})
        return self.loss_history[-1]

    def train_epoch(self, s_dataset, batch_size=256, n_batches=None):
        N = s_dataset.shape[0]
        if n_batches is None: n_batches = max(1, N // batch_size)
        losses = []
        for _ in range(n_batches):
            idx = torch.randint(0, N, (batch_size,), device=self.device)
            losses.append(self.train_step(s_dataset[idx]))
        return {k: np.mean([l[k] for l in losses]) for k in losses[0]}


class CartPoleSequenceTrainer(CartPoleSingleStepTrainer):
    """Train policy with H-step KAN rollout — policy queried at every step."""
    def __init__(self, *args, horizon=5, gamma=0.85, **kwargs):
        super().__init__(*args, **kwargs)
        self.horizon = horizon; self.gamma = gamma

    def train_step(self, s_batch):
        B = s_batch.shape[0]; self.policy.train(); self.optimizer.zero_grad()

        s = s_batch
        total_pred = torch.tensor(0.0, device=self.device)
        total_ctrl = torch.tensor(0.0, device=self.device)

        for t in range(self.horizon):
            a = self.policy(s)
            s = self.kan(torch.cat([s, a], dim=-1))
            step_loss = s[:, 2].pow(2).mean() + 0.1 * s[:, 0].pow(2).mean()
            total_pred = total_pred + (self.gamma ** t) * step_loss
            total_ctrl = total_ctrl + a.pow(2).mean() * (self.gamma ** t)

        total = total_pred / self.horizon + self.lambda_ctrl * total_ctrl

        total.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.clip_grad)
        self.optimizer.step()

        self.loss_history.append({
            'total': total.item(), 'pred': total_pred.item(), 'ctrl': total_ctrl.item()})
        return self.loss_history[-1]


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_continuous(policy, device, n_trials=10, max_steps=500, label=''):
    """Evaluate with continuous-force CartPole simulation (analytical dynamics)."""
    torch.manual_seed(42); np.random.seed(42)
    all_steps = []
    for trial in range(n_trials):
        seed = 42 + trial * 100
        torch.manual_seed(seed)
        # Random initial state near upright
        th = np.random.uniform(-0.05, 0.05)
        s_raw = torch.tensor([[0.0, 0.0, th, 0.0]], dtype=torch.float32)

        for step in range(max_steps):
            s_norm = s_raw.clone()
            s_norm[:, 0] /= X_SCALE; s_norm[:, 1] /= XD_SCALE
            s_norm[:, 2] /= TH_SCALE; s_norm[:, 3] /= THD_SCALE

            with torch.no_grad():
                a_norm = policy(s_norm).item()
            force = a_norm * FORCE_MAX

            s_raw = step_cartpole_cont(s_raw, torch.tensor([a_norm]))

            # Check termination
            theta = s_raw[0, 2].item()
            x = s_raw[0, 0].item()
            if abs(theta) > 0.21 or abs(x) > 2.4:
                break

        all_steps.append(step + 1)

    mean_s = np.mean(all_steps)
    success = sum(1 for s in all_steps if s >= max_steps) / n_trials
    print(f"  {label}: survived {mean_s:.0f}±{np.std(all_steps):.0f} steps  "
          f"success={success*100:.0f}%")
    return mean_s, all_steps


class HeuristicPolicy:
    """Push in the direction the pole is leaning. PD controller."""
    def __call__(self, s_norm):
        th = s_norm[0, 2].item() * TH_SCALE   # denormalize
        thd = s_norm[0, 3].item() * THD_SCALE
        x = s_norm[0, 0].item() * X_SCALE
        xd = s_norm[0, 1].item() * XD_SCALE
        # PD: push toward pole lean direction + center cart
        a_raw = 50.0 * th + 10.0 * thd + 2.0 * x + 5.0 * xd
        return torch.tensor([[np.clip(a_raw / FORCE_MAX, -1.0, 1.0)]])


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--kan-epochs', type=int, default=1200)
    parser.add_argument('--horizon', type=int, default=5)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--no-train', action='store_true')
    parser.add_argument('--cache-dir', type=str, default='/tmp/kanrf_cl_cp')
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.cache_dir, exist_ok=True)

    # ── Phase 1: Data ──
    data_path = os.path.join(args.cache_dir, 'cartpole_data.pt')
    kan_path = os.path.join(args.cache_dir, 'cartpole_kan_cws.pt')
    p1_path = os.path.join(args.cache_dir, 'cartpole_policy_single.pt')
    p2_path = os.path.join(args.cache_dir, 'cartpole_policy_seq.pt')

    state_dim = 4  # CartPole: [x, x_dot, theta, theta_dot]

    if not args.no_train or not os.path.exists(data_path):
        print("=" * 60 + "\nPhase 1: Generating CartPole data\n" + "=" * 60)
        (x_tr, y_tr, x_val, y_val), _, _ = generate_data(20000)
        torch.save((x_tr, y_tr, x_val, y_val), data_path)
    else:
        print("Loading cached data...")
        x_tr, y_tr, x_val, y_val = torch.load(data_path, weights_only=True)

    x_tr, y_tr = x_tr.to(device), y_tr.to(device)
    x_val, y_val = x_val.to(device), y_val.to(device)
    print(f"  Train: {x_tr.shape}, Val: {x_val.shape}")

    # ── Phase 2: Train CWS KAN ──
    if not args.no_train or not os.path.exists(kan_path):
        print("\n" + "=" * 60 + "\nPhase 2: Training CartPole CWS KAN\n" + "=" * 60)
        kan = train_kan_cws(x_tr, y_tr, x_val, y_val, epochs=args.kan_epochs)
        kan.to(device); torch.save(kan.state_dict(), kan_path)
    else:
        kan = KAN([5, 12, 4], grid_size=5, spline_order=3)
        kan.load_state_dict(torch.load(kan_path, weights_only=True))
        kan.to(device); kan.eval()
        print(f"  KAN loaded: {sum(p.numel() for p in kan.parameters())} params")

    s_target = torch.zeros(1, 4, device=device)

    # ── Phase 3: Train policies ──
    n_states = 30000
    # Use CartPole's actual state range (near-vertical pole for realistic training)
    # Sample from rollout trajectories + uniform near-center
    env_tmp = gym.make('CartPole-v1')
    rollout_states = []
    obs, _ = env_tmp.reset(seed=42)
    for _ in range(20000):
        a = np.random.choice([0, 1])
        obs, _, term, trunc, _ = env_tmp.step(a)
        rollout_states.append(obs.copy())
        if term or trunc:
            obs, _ = env_tmp.reset()
    env_tmp.close()
    rollout = np.array(rollout_states, dtype=np.float32)

    # Also add uniform samples around interesting regions
    n_uniform = n_states - len(rollout)
    s_uniform = np.stack([
        np.random.uniform(-2.5, 2.5, n_uniform) / X_SCALE,
        np.random.uniform(-3.0, 3.0, n_uniform) / XD_SCALE,
        np.random.uniform(-0.4, 0.4, n_uniform) / TH_SCALE,  # wider than env but still near upright
        np.random.uniform(-3.0, 3.0, n_uniform) / THD_SCALE,
    ], axis=1)

    # Mix
    s_rollout_norm = rollout.copy()
    s_rollout_norm[:, 0] /= X_SCALE; s_rollout_norm[:, 1] /= XD_SCALE
    s_rollout_norm[:, 2] /= TH_SCALE; s_rollout_norm[:, 3] /= THD_SCALE

    s_all = np.vstack([s_rollout_norm, s_uniform])
    np.random.shuffle(s_all)
    s_dataset = torch.from_numpy(s_all[:n_states].astype(np.float32)).to(device)
    print(f"  Training states: {s_dataset.shape}")

    print("\n" + "=" * 60 + "\nPhase 3: Training policies\n" + "=" * 60)

    # 3a: Single-step
    print("\n[Single-step training]")
    if not args.no_train or not os.path.exists(p1_path):
        p1 = CartPolePolicy(state_dim=state_dim)
        t1 = CartPoleSingleStepTrainer(kan, p1, s_target, device=device)
        for ep in range(1, args.epochs + 1):
            ld = t1.train_epoch(s_dataset)
            if ep % 40 == 0: print(f"  Epoch {ep:3d}  loss={ld['total']:.4f}")
        torch.save(p1.state_dict(), p1_path)
    else:
        p1 = CartPolePolicy(); p1.load_state_dict(torch.load(p1_path))

    # 3b: Sequence
    print("\n[Sequence training H={}]".format(args.horizon))
    if not args.no_train or not os.path.exists(p2_path):
        p2 = CartPolePolicy(state_dim=state_dim)
        t2 = CartPoleSequenceTrainer(kan, p2, s_target, device=device,
                                      horizon=args.horizon)
        for ep in range(1, args.epochs + 1):
            ld = t2.train_epoch(s_dataset)
            if ep % 40 == 0: print(f"  Epoch {ep:3d}  loss={ld['total']:.4f}")
        torch.save(p2.state_dict(), p2_path)
    else:
        p2 = CartPolePolicy(); p2.load_state_dict(torch.load(p2_path))

    # ── Phase 4: Evaluate ──
    print("\n" + "=" * 60 + "\nPhase 4: Evaluation (continuous-force CartPole)\n" + "=" * 60)
    p1.to(device); p1.eval(); p2.to(device); p2.eval()

    r_heuristic = evaluate_continuous(HeuristicPolicy(), device, n_trials=20,
                                       label='Heuristic PD')
    r_single = evaluate_continuous(p1, device, n_trials=20, label='Single-step')
    r_seq = evaluate_continuous(p2, device, n_trials=20,
                                 label='Sequence H={}'.format(args.horizon))

    print("\nDone!")


if __name__ == '__main__':
    main()
