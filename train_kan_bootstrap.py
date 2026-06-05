"""Iterative bootstrap training: collect data from energy-guided trials,
augment training set with top-region samples, retrain KAN on GPU.

Usage:
  python train_kan_bootstrap.py --device mps --n-collect 20 --n-epochs 2000
"""
import sys, time, argparse, os
import torch
import gymnasium as gym
import numpy as np
from kanrf import KAN
from online_learning_v2 import ThreeFactorUpdater, compute_training_stats

G = 10.0; PI_2 = np.pi / 2; E_DES = G


def _normalize_state(s, device=None):
    t = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
    t[0, 2] /= 8.0
    return t.to(device) if device is not None else t


def _normalize_action(a, device=None):
    t = torch.tensor([[a / 2.0]], dtype=torch.float32)
    return t.to(device) if device is not None else t


# ─── Energy-guided action (single-step, fast) ──────────────────────────────

def find_action_energy(model, s_norm, s_target_norm, n_iters=50, lr=0.05,
                       lambda_ctrl=0.001, device=None):
    """Single-step energy-guided action optimization."""
    s_raw = s_norm.clone(); s_raw[0, 2] *= 8.0
    sin_th = s_raw[0, 1].item()
    thd = s_raw[0, 2].item()
    near_upright = abs(s_raw[0, 0].item()) < 0.5 and sin_th > 0 and abs(thd) < 3.0
    w_E = 1.0
    w_pos = 3.0 if near_upright else 0.0

    a_n = torch.zeros(1, 1, device=device)
    torch.nn.init.uniform_(a_n, a=-0.3, b=0.3)
    a_n.requires_grad_(True)
    opt = torch.optim.Adam([a_n], lr=lr)

    for _ in range(n_iters):
        opt.zero_grad()
        x = torch.cat([s_norm, a_n], dim=-1)
        s_pred = model(x)
        sin_pred = s_pred[0, 1]; thd_pred = s_pred[0, 2] * 8.0
        E_pred = 0.5 * thd_pred * thd_pred + G * sin_pred
        loss_E = w_E * (E_pred - E_DES) ** 2
        loss_pos = torch.tensor(0.0, device=device)
        if w_pos > 0:
            loss_pos = w_pos * ((s_pred[0, :2] - s_target_norm[0, :2]) ** 2).sum()
        loss = loss_E + loss_pos + lambda_ctrl * (a_n ** 2).sum()
        loss.backward()
        opt.step()
        with torch.no_grad():
            a_n.clamp_(-1.0, 1.0)

    with torch.no_grad():
        s_pred = model(torch.cat([s_norm, a_n], dim=-1))
    return a_n.detach().item() * 2.0


# ─── Phase 1: Data collection ──────────────────────────────────────────────

def collect_trials(model, env, s_goal, device, n_trials=20, total_steps=60,
                   n_iters=50, collect_all=False):
    """Run energy-guided trials, collect transitions from successful ones.

    Returns list of (s_normalized, a_normalized, s_next_normalized) tuples.
    All in normalized space (ready for KAN training).
    """
    model.eval()
    all_transitions = []
    success_count = 0

    for t in range(n_trials):
        obs, _ = env.reset()
        trial_transitions = []
        trial_success = False

        for step in range(total_steps):
            s_norm = _normalize_state(obs, device=device)
            s_target_norm = torch.tensor([[0.0, 1.0, 0.0]], device=device)

            a = find_action_energy(model, s_norm, s_target_norm,
                                   n_iters=n_iters, lr=0.05, device=device)
            obs_next, _, term, trunc, _ = env.step([a])

            a_norm = _normalize_action(a, device=device)
            s_next_norm = _normalize_state(obs_next, device=device)

            x_vec = torch.cat([s_norm, a_norm], dim=-1).cpu().squeeze(0)  # (4,) = [cos,sin,thd/8,torque/2]
            trial_transitions.append((x_vec, s_next_norm.cpu().squeeze(0)))  # (4,), (3,)

            obs = obs_next

            if abs(np.arctan2(obs[1], obs[0]) - PI_2) < 0.2:
                trial_success = True
                break
            if term or trunc:
                break

        if trial_success or collect_all:
            all_transitions.extend(trial_transitions)
            success_count += 1

    print(f"  Collected: {len(all_transitions)} transitions "
          f"from {success_count}/{n_trials} {'successful' if not collect_all else 'all'} trials")
    return all_transitions, success_count


# ─── Phase 2: Augment dataset ──────────────────────────────────────────────

def augment_dataset(original_x, original_y, new_transitions, device):
    """Merge new transitions into training data.

    original_x:  (N,4) normalized [cos, sin, thd/8, torque/2]
    original_y:  (N,3) normalized [cos', sin', thd'/8']
    new_transitions: list of (s_norm(4,), s_next_norm(3,))
    """
    if not new_transitions:
        print("  No new transitions to add.")
        return original_x, original_y

    new_x = torch.stack([t[0] for t in new_transitions])  # (M, 4)
    new_y = torch.stack([t[1] for t in new_transitions])  # (M, 3)

    x_aug = torch.cat([original_x, new_x], dim=0)
    y_aug = torch.cat([original_y, new_y], dim=0)

    print(f"  Augmented: {len(original_x)} → {len(x_aug)} "
          f"(+{len(new_x)} new, {len(new_x)/len(original_x)*100:.1f}%)")
    return x_aug, y_aug


# ─── Phase 3: Retrain ──────────────────────────────────────────────────────

def retrain_kan(model, x_train, y_train, device, n_epochs=2000, lr=1e-2):
    """Train KAN on MPS with full-batch gradient descent."""
    x = x_train.to(device)
    y = y_train.to(device)
    model = model.to(device)
    model.train()

    n_val = min(int(len(x) * 0.15), 5000)
    x_val, y_val = x[-n_val:], y[-n_val:]
    x_tr, y_tr = x[:-n_val], y[:-n_val]

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=500, gamma=0.5)
    loss_fn = torch.nn.MSELoss()

    best_val = float('inf')
    t0 = time.time()

    for epoch in range(1, n_epochs + 1):
        opt.zero_grad()
        loss = loss_fn(model(x_tr), y_tr)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        opt.step()
        scheduler.step()

        if epoch % 300 == 0 or epoch == 1 or epoch == n_epochs:
            model.eval()
            with torch.no_grad():
                pred = model(x_val)
                vloss = loss_fn(pred, y_val).item()
                dim_mse = ((pred - y_val) ** 2).mean(dim=0)
            model.train()

            if vloss < best_val:
                best_val = vloss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            elapsed = time.time() - t0
            print(f"  Epoch {epoch:5d}/{n_epochs}  lr={scheduler.get_last_lr()[0]:.4f}  "
                  f"train={loss.item():.6f}  val={vloss:.6f}  "
                  f"dim=[cos:{dim_mse[0]:.5f} sin:{dim_mse[1]:.5f} thd:{dim_mse[2]:.5f}]  "
                  f"[{elapsed:.0f}s]")

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        final_pred = model(x_val)
        final_val = loss_fn(final_pred, y_val).item()
        dim_mse = ((final_pred - y_val) ** 2).mean(dim=0)

    print(f"\n  Best val MSE: {best_val:.6f}  |  Final: {final_val:.6f}")
    print(f"  Per-dim: cosθ={dim_mse[0]:.6f} sinθ={dim_mse[1]:.6f} θ̇/8={dim_mse[2]:.6f}")

    # Check top-region accuracy
    with torch.no_grad():
        pred = model(x_val)
        err = (pred - y_val).norm(dim=-1)
        top_mask = (y_val[:, 1] > 0.5) & (y_val[:, 0].abs() < 0.5) & (y_val[:, 2].abs() < 0.3)
        if top_mask.any():
            top_rmse = err[top_mask].mean().item()
            bot_rmse = err[~top_mask].mean().item()
            print(f"  Top RMSE: {top_rmse:.4f}  |  Bottom RMSE: {bot_rmse:.4f}")

    return model.cpu()


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='kan_pendulum_model_v4.pt')
    parser.add_argument('--data', type=str, default='pendulum_data_v4.pt')
    parser.add_argument('--out-model', type=str, default='kan_pendulum_model_v5.pt')
    parser.add_argument('--out-data', type=str, default='pendulum_data_v5.pt')
    parser.add_argument('--device', type=str, default='mps',
                       choices=['cpu', 'mps', 'cuda'])
    parser.add_argument('--n-collect', type=int, default=20)
    parser.add_argument('--n-iters', type=int, default=50)
    parser.add_argument('--n-epochs', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--collect-all', action='store_true', default=False,
                       help='Collect from ALL trials, not just successful ones')
    parser.add_argument('--iterations', type=int, default=1,
                       help='Number of bootstrap iterations (collect→augment→retrain)')
    args = parser.parse_args()

    torch.manual_seed(42); np.random.seed(42)

    # Device
    if args.device == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
    elif args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}")

    # Load base data
    data = torch.load(args.data, weights_only=True)
    if isinstance(data, tuple) and len(data) == 2:
        x_base, y_base = data
    else:
        print("Unknown data format"); sys.exit(1)

    x_current, y_current = x_base, y_base
    model_path = args.model
    env = gym.make("Pendulum-v1")
    s_goal = torch.tensor([[0.0, 1.0, 0.0]])

    for iteration in range(args.iterations):
        print(f"\n{'=' * 70}")
        print(f"BOOTSTRAP ITERATION {iteration + 1}/{args.iterations}")
        print(f"{'=' * 70}")

        # Load model
        model = KAN([4, 12, 3], grid_size=5, spline_order=3)
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model = model.to(device)

        # Phase 1: Collect data
        print(f"\n[Phase 1] Collecting data ({args.n_collect} trials)...")
        t0 = time.time()
        transitions, n_success = collect_trials(
            model, env, s_goal, device,
            n_trials=args.n_collect, n_iters=args.n_iters,
            collect_all=args.collect_all)
        print(f"  Time: {time.time() - t0:.0f}s")

        # Phase 2: Augment
        print(f"\n[Phase 2] Augmenting dataset...")
        x_current, y_current = augment_dataset(x_current, y_current, transitions, device)

        # Save augmented data
        torch.save((x_current, y_current), args.out_data)

        # Phase 3: Retrain
        print(f"\n[Phase 3] Retraining ({args.n_epochs} epochs on {device})...")
        model = retrain_kan(model, x_current, y_current, device,
                           n_epochs=args.n_epochs, lr=args.lr)

        # Save model
        torch.save(model.state_dict(), args.out_model)
        print(f"  Saved: {args.out_model}  |  Data: {args.out_data}")
        model_path = args.out_model

    env.close()
    print(f"\nDone. Final model: {args.out_model}")


if __name__ == "__main__":
    main()
