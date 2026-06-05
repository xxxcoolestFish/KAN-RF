"""KAN Native Curiosity-Driven Exploration.

Core idea: B-spline activation density rho is a FREE uncertainty signal.
Low rho = untrained region. Use this to actively explore.

Experiment structure:
  1. Train KAN on random data, compute rho, visualize uncertainty map
  2. Single-round: target high-uncertainty states, explore, collect data, retrain
  3. (Optional) Iterative: repeat the explore-retrain cycle

Usage:
  python explore_kan.py                      # full experiment
  python explore_kan.py --map-only           # only visualize uncertainty map
  python explore_kan.py --no-train           # skip initial training (use existing model)
"""
import sys, time, argparse, os
import torch
import gymnasium as gym
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from kanrf import KAN
from kanrf import compute_per_step_uncertainty

G = 10.0
PI_2 = np.pi / 2

# ─── Uncertainty Map ────────────────────────────────────────────────────

def compute_training_density(model, x_train):
    """Compute per-basis-function activation density from training data.

    Returns:
        density: list of (in_dim, n_basis) tensors, one per layer
    """
    model.eval()
    density = []
    with torch.no_grad():
        _, B_list, _ = model(x_train, return_activations=True)
        for B in B_list:
            active = (B > 1e-8).float()       # (N, in_dim, n_basis)
            freq = active.mean(dim=0)          # (in_dim, n_basis)
            density.append(freq)
    return density


def state_uncertainty_rho(model, s_norm, density):
    """Compute uncertainty using training density rho directly.

    For each activated B-spline basis function, check its training frequency rho.
    Low rho = rarely activated during training = model hasn't learned this region.

    U(s) = 1 - average(rho of activated basis functions, weighted by activation)

    This directly answers: "has the model seen data in this region during training?"

    Args:
        s_norm: (1, 3) normalized state [cos, sin, thd/8]
        density: list of (in_dim, n_basis) — training activation frequency

    Returns: scalar uncertainty in [0, 1]
    """
    model.eval()
    with torch.no_grad():
        x = torch.cat([s_norm, torch.zeros(1, 1)], dim=-1)
        _, B_list, _ = model(x, return_activations=True)

        unc_per_layer = []
        for l, B in enumerate(B_list):
            # B: (1, in_dim, n_basis)
            # density[l]: (in_dim, n_basis)
            active = B.squeeze(0)  # (in_dim, n_basis)
            rho = density[l]       # (in_dim, n_basis)

            # Weighted average: higher activation → more weight
            weights = active.abs() + 1e-8
            weighted_rho = (weights * rho).sum() / weights.sum()
            unc_per_layer.append(1.0 - weighted_rho.item())

    return float(np.mean(unc_per_layer))


def build_uncertainty_map(model, density, grid_n=25):
    """Build uncertainty map over state space.

    Grid: cos in [-1,1], sin in [-1,1], thd in [-8,8].
    Returns: (cos_grid, sin_grid, thd_grid, unc_3d) where unc_3d is (G, G, G).
    """
    cos_vals = np.linspace(-1, 1, grid_n)
    sin_vals = np.linspace(-1, 1, grid_n)
    # thd: use non-uniform grid — higher density near 0 (important region)
    thd_vals = np.linspace(-8, 8, grid_n)

    unc_3d = np.zeros((grid_n, grid_n, grid_n))
    total = grid_n ** 3

    for i, cos_t in enumerate(cos_vals):
        for j, sin_t in enumerate(sin_vals):
            r = np.sqrt(cos_t**2 + sin_t**2)
            if abs(r - 1.0) > 0.01:
                unc_3d[i, j, :] = np.nan
                continue
            for k, thd in enumerate(thd_vals):
                s_norm = torch.tensor([[cos_t, sin_t, thd / 8.0]], dtype=torch.float32)
                unc_3d[i, j, k] = state_uncertainty_rho(model, s_norm, density)

        if (i * grid_n + j) % 50 == 0:
            print(f"    uncertainty map: {i*grid_n+j}/{grid_n**2} slices done")

    return cos_vals, sin_vals, thd_vals, unc_3d


def plot_uncertainty_map(cos_v, sin_v, thd_v, unc_3d, save_path):
    """Visualize uncertainty map.

    Left: slice at thd=0 (cos-sin plane).
    Right: slice at different sin values (thd-cos plane).
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. thd=0 slice: cos-sin plane
    k0 = np.argmin(abs(thd_v))
    ax = axes[0, 0]
    im = ax.pcolormesh(cos_v, sin_v, unc_3d[:, :, k0].T,
                       shading='auto', cmap='YlOrRd', vmin=0, vmax=1)
    ax.set_xlabel('cos(theta)'); ax.set_ylabel('sin(theta)')
    ax.set_title(f'rho-based Uncertainty: thd={thd_v[k0]:.1f} slice')
    ax.axhline(0, color='gray', ls='--', alpha=0.3)
    ax.axvline(0, color='gray', ls='--', alpha=0.3)
    ax.plot(0, 1, 'g*', markersize=15, label='Goal (upright)')
    ax.plot(-1, 0, 'rx', markersize=10, label='Bottom')
    ax.legend()
    plt.colorbar(im, ax=ax, label='Uncertainty')

    # 2. sin=1.0 slice: thd-cos plane (near upright)
    j_high = np.argmin(abs(sin_v - 0.9))
    ax = axes[0, 1]
    valid = ~np.isnan(unc_3d[:, j_high, :])
    if valid.any():
        im = ax.pcolormesh(cos_v, thd_v, unc_3d[:, j_high, :].T,
                           shading='auto', cmap='YlOrRd', vmin=0, vmax=1)
        ax.set_xlabel('cos(theta)'); ax.set_ylabel('theta_dot')
        ax.set_title(f'Uncertainty map: sin={sin_v[j_high]:.2f} slice')
        ax.axhline(0, color='gray', ls='--', alpha=0.3)
        ax.plot(0, 0, 'g*', markersize=15, label='Goal')
        ax.legend()
        plt.colorbar(im, ax=ax, label='Uncertainty')

    # 3. sin=-0.9 slice: thd-cos plane (near bottom)
    j_low = np.argmin(abs(sin_v + 0.9))
    ax = axes[1, 0]
    valid = ~np.isnan(unc_3d[:, j_low, :])
    if valid.any():
        im = ax.pcolormesh(cos_v, thd_v, unc_3d[:, j_low, :].T,
                           shading='auto', cmap='YlOrRd', vmin=0, vmax=1)
        ax.set_xlabel('cos(theta)'); ax.set_ylabel('theta_dot')
        ax.set_title(f'Uncertainty map: sin={sin_v[j_low]:.2f} slice')
        ax.axhline(0, color='gray', ls='--', alpha=0.3)
        plt.colorbar(im, ax=ax, label='Uncertainty')

    # 4. Histogram of uncertainty values
    ax = axes[1, 1]
    flat = unc_3d[~np.isnan(unc_3d)]
    ax.hist(flat, bins=50, color='orangered', alpha=0.7, edgecolor='black')
    ax.axvline(np.mean(flat), color='blue', ls='--', label=f'mean={np.mean(flat):.3f}')
    ax.axvline(np.percentile(flat, 90), color='red', ls='--',
               label=f'P90={np.percentile(flat, 90):.3f}')
    ax.set_xlabel('Uncertainty'); ax.set_ylabel('Count')
    ax.set_title('Uncertainty distribution over state space')
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()
    print(f"  Uncertainty map saved to: {save_path}")


def find_exploration_targets(cos_v, sin_v, thd_v, unc_3d, n_targets=3):
    """Find states with highest uncertainty as exploration targets.

    Returns list of (unc, [cos, sin, thd_raw]).
    """
    valid_mask = ~np.isnan(unc_3d)
    flat_idx = np.argsort(unc_3d[valid_mask])[::-1]  # descending
    valid_indices = np.argwhere(valid_mask)

    targets = []
    for idx in flat_idx[:n_targets * 10]:  # oversample to filter nearby targets
        i, j, k = valid_indices[idx]
        unc = unc_3d[i, j, k]
        state = np.array([cos_v[i], sin_v[j], thd_v[k]], dtype=np.float32)

        # Avoid targets too close to each other
        too_close = False
        for _, prev_state in targets:
            if np.linalg.norm(state - prev_state) < 2.0:
                too_close = True
                break
        if not too_close:
            targets.append((unc, state))
        if len(targets) >= n_targets:
            break

    return targets


# ─── Exploration Episode ────────────────────────────────────────────────

def explore_toward_target(model, env, s_target, horizon=10, n_shoot_iters=200,
                          execute_steps=15, noise_std=0.3):
    """Run one exploration episode targeting s_target.

    1. Shooting through KAN to plan H-step trajectory toward target
    2. Execute first `execute_steps` actions with noise
    3. Collect all (s, a, s') transitions

    Args:
        model: KAN world model
        env: gym environment (already reset to some state)
        s_target: (3,) numpy array, target state [cos, sin, thd_raw]
        horizon: shooting horizon
        n_shoot_iters: shooting optimization iterations
        execute_steps: how many steps to execute (<= horizon)
        noise_std: action noise std for exploration

    Returns:
        transitions: list of (s, a, s') numpy arrays
        reached: final state
    """
    obs0, _ = env.reset()
    s0 = torch.tensor(obs0, dtype=torch.float32).unsqueeze(0)
    s_target_t = torch.tensor(s_target, dtype=torch.float32).unsqueeze(0)

    # Normalize
    s0_norm = s0.clone(); s0_norm[:, 2] /= 8.0
    s_target_norm = s_target_t.clone(); s_target_norm[:, 2] /= 8.0

    # Shooting through KAN toward target
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    horizon_eff = min(horizon, execute_steps + 5)
    a_norm = torch.zeros(horizon_eff, 1)
    torch.nn.init.uniform_(a_norm, a=-0.3, b=0.3)
    a_norm.requires_grad_(True)
    opt = torch.optim.Adam([a_norm], lr=0.1)

    for _ in range(n_shoot_iters):
        opt.zero_grad()
        s = s0_norm.clone()
        for h in range(horizon_eff):
            x = torch.cat([s, a_norm[h:h + 1]], dim=-1)
            s = model(x)
            norm = (s[:, :2] ** 2).sum(dim=-1, keepdim=True).sqrt().clamp(min=1e-6)
            s = torch.cat([s[:, :2] / norm, s[:, 2:]], dim=-1)
        loss = ((s - s_target_norm) ** 2).sum() + 0.001 * (a_norm ** 2).sum()
        loss.backward()
        opt.step()
        with torch.no_grad():
            a_norm.clamp_(-1.0, 1.0)

    for p in model.parameters():
        p.requires_grad = True

    # Execute with noise
    transitions = []
    obs = obs0
    actions_planned = (a_norm.detach().numpy().flatten() * 2.0)[:execute_steps]

    for a in actions_planned:
        a_noisy = np.clip(a + np.random.normal(0, noise_std), -2.0, 2.0)
        obs_next, _, term, trunc, _ = env.step([a_noisy])
        transitions.append((obs.copy(), a_noisy, obs_next.copy()))
        obs = obs_next
        if term or trunc:
            break

    return transitions, obs


# ─── Data Collection Helpers ────────────────────────────────────────────

def collect_random_data(env, n_transitions):
    """Collect (s,a,s') with uniform random actions."""
    states, actions, next_states = [], [], []
    obs, _ = env.reset()
    for _ in range(n_transitions):
        a = np.random.uniform(-2.0, 2.0)
        next_obs, _, term, trunc, _ = env.step([a])
        states.append(obs.copy()); actions.append(a); next_states.append(next_obs.copy())
        obs = next_obs
        if term or trunc:
            obs, _ = env.reset()
    return states, actions, next_states


def to_tensors(states, actions, next_states):
    s = torch.tensor(np.array(states), dtype=torch.float32)
    a = torch.tensor(np.array(actions), dtype=torch.float32).unsqueeze(-1)
    sn = torch.tensor(np.array(next_states), dtype=torch.float32)
    s_norm = s.clone(); s_norm[:, 2] /= 8.0
    a_norm = a / 2.0
    sn_norm = sn.clone(); sn_norm[:, 2] /= 8.0
    return torch.cat([s_norm, a_norm], dim=-1), sn_norm


def train_kan(x_train, y_train, x_val, y_val, epochs=800, lr=1e-2, label=""):
    """Train KAN world model."""
    model = KAN([4, 12, 3], grid_size=5, spline_order=3)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=200, gamma=0.5)

    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(model(x_train), y_train)
        loss.backward()
        opt.step()
        scheduler.step()

        if epoch % 200 == 0:
            model.eval()
            with torch.no_grad():
                vloss = torch.nn.functional.mse_loss(model(x_val), y_val)
            print(f"  [{label}] Epoch {epoch:4d}  train={loss.item():.6f}  val={vloss.item():.6f}")

    model.eval()
    with torch.no_grad():
        pred = model(x_val)
        final_mse = torch.nn.functional.mse_loss(pred, y_val).item()
        dim_mse = ((pred - y_val) ** 2).mean(dim=0)
    print(f"  [{label}] Final val MSE: {final_mse:.6f}  "
          f"dim=[{dim_mse[0]:.6f},{dim_mse[1]:.6f},{dim_mse[2]:.6f}]")
    return model


# ─── Main Experiment ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--map-only', action='store_true',
                       help='Only compute and visualize uncertainty map')
    parser.add_argument('--no-train', action='store_true',
                       help='Skip initial training (load existing model)')
    parser.add_argument('--n-random', type=int, default=5000,
                       help='Number of random transitions for initial training')
    parser.add_argument('--n-explore-episodes', type=int, default=10,
                       help='Number of exploration episodes')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)

    print(f"{'='*70}")
    print(f"KAN Native Curiosity-Driven Exploration")
    print(f"{'='*70}")

    # ── Step 1: Initial Data + Training ──
    env = gym.make("Pendulum-v1")

    if args.no_train:
        print("\n[1/5] Loading existing model...")
        model = KAN([4, 12, 3], grid_size=5, spline_order=3)
        model.load_state_dict(torch.load("kan_pendulum_model.pt", weights_only=True))
        x_train, y_train = torch.load("pendulum_data.pt", weights_only=True)
        print(f"  Loaded kan_pendulum_model.pt + pendulum_data.pt ({len(x_train)} samples)")
    else:
        print(f"\n[1/5] Collecting {args.n_random} random transitions...")
        s_rand, a_rand, ns_rand = collect_random_data(env, args.n_random)
        x_all, y_all = to_tensors(s_rand, a_rand, ns_rand)
        n_train = int(len(x_all) * 0.85)
        x_train, y_train = x_all[:n_train], y_all[:n_train]
        x_val, y_val = x_all[n_train:], y_all[n_train:]
        print(f"  Train: {len(x_train)}, Val: {len(x_val)}")

        print("\n[2/5] Training initial KAN world model...")
        model = train_kan(x_train, y_train, x_val, y_val, epochs=800, label="init")
        torch.save(model.state_dict(), "kan_explore_init.pt")

    # ── Step 2: Compute Uncertainty Map ──
    print(f"\n[{'3' if args.no_train else '3'}/5] Computing training density rho...")
    density = compute_training_density(model, x_train)
    for li, d in enumerate(density):
        print(f"  Layer {li}: rho range [{d.min():.4f}, {d.max():.4f}], "
              f"mean={d.mean():.4f}")
        # Identify least-trained basis functions
        flat = d.flatten()
        low_idx = flat.argsort()[:5]
        for idx in low_idx:
            j = idx // d.shape[1]
            k = idx % d.shape[1]
            print(f"    Low rho: Layer{li} dim{j} basis{k} = {d[j,k]:.4f}")

    print("\n[4/5] Building uncertainty map...")
    cos_v, sin_v, thd_v, unc_3d = build_uncertainty_map(model, density, grid_n=20)
    plot_uncertainty_map(cos_v, sin_v, thd_v, unc_3d, "explore_uncertainty_map.png")

    if args.map_only:
        env.close()
        print("\nDone. Uncertainty map saved.")
        return

    # ── Step 3: Find Exploration Targets ──
    print("\n[5/5] Finding exploration targets...")
    targets = find_exploration_targets(cos_v, sin_v, thd_v, unc_3d, n_targets=3)
    for i, (unc, state) in enumerate(targets):
        print(f"  Target {i+1}: unc={unc:.4f}  "
              f"s=[{state[0]:+.2f},{state[1]:+.2f},{state[2]:+.2f}]")

    # ── Step 4: Explore ──
    print(f"\n{'='*70}")
    print(f"Exploration Phase")
    print(f"{'='*70}")

    all_explore_states, all_explore_actions, all_explore_next = [], [], []

    for ep in range(args.n_explore_episodes):
        target_idx = ep % len(targets)
        _, s_target = targets[target_idx]
        print(f"\n  Episode {ep+1}/{args.n_explore_episodes} → "
              f"target=[{s_target[0]:+.2f},{s_target[1]:+.2f},{s_target[2]:+.2f}]")

        transitions, final_obs = explore_toward_target(
            model, env, s_target, horizon=15, execute_steps=10)
        for s, a, ns in transitions:
            all_explore_states.append(s)
            all_explore_actions.append(a)
            all_explore_next.append(ns)

        angle_f = np.arctan2(final_obs[1], final_obs[0])
        print(f"    Collected {len(transitions)} transitions, "
              f"final: [{final_obs[0]:+.2f},{final_obs[1]:+.2f},{final_obs[2]:+.2f}]  "
              f"|Δθ|={abs(angle_f - PI_2):.3f}rad")

    # ── Step 5: Augment + Retrain ──
    print(f"\n{'='*70}")
    print(f"Retraining with augmented data")
    print(f"{'='*70}")

    x_explore, y_explore = to_tensors(all_explore_states, all_explore_actions, all_explore_next)
    print(f"  Exploration data: {len(x_explore)} transitions")

    # Analyze what regions we reached
    explore_sin = np.array([s[1] for s in all_explore_states])
    near_top = (explore_sin > 0.8).mean()
    print(f"  Fraction of exploration data near upright (sin>0.8): {near_top:.1%}")

    # Augment training set
    x_aug = torch.cat([x_train, x_explore], dim=0)
    y_aug = torch.cat([y_train, y_explore], dim=0)
    perm = torch.randperm(len(x_aug))
    x_aug, y_aug = x_aug[perm], y_aug[perm]

    n_train2 = int(len(x_aug) * 0.85)
    x_train2, y_train2 = x_aug[:n_train2], y_aug[:n_train2]
    x_val2, y_val2 = x_aug[n_train2:], y_aug[n_train2:]

    print(f"  Augmented train: {len(x_train2)}, Val: {len(x_val2)}")
    print(f"  (Original: {len(x_train)} + {len(x_explore)} explore)")

    model2 = train_kan(x_train2, y_train2, x_val2, y_val2, epochs=800, label="augmented")
    torch.save(model2.state_dict(), "kan_explore_augmented.pt")

    # ── Step 6: Compare ──
    print(f"\n{'='*70}")
    print(f"Comparison: Before vs After Exploration")
    print(f"{'='*70}")

    # Test on data near upright (sin>0.8) — the critical sparse region
    test_mask = y_val2[:, 1] > 0.8  # sin > 0.8
    if test_mask.sum() > 0:
        x_test_top = x_val2[test_mask]
        y_test_top = y_val2[test_mask]

        with torch.no_grad():
            pred_before = model(x_test_top)
            pred_after = model2(x_test_top)
            mse_before = torch.nn.functional.mse_loss(pred_before, y_test_top).item()
            mse_after = torch.nn.functional.mse_loss(pred_after, y_test_top).item()

        print(f"  Test samples near upright (sin>0.8): {test_mask.sum().item()}")
        print(f"  MSE before exploration:  {mse_before:.6f}")
        print(f"  MSE after exploration:   {mse_after:.6f}")
        if mse_after < mse_before:
            print(f"  Improvement: {(1 - mse_after/mse_before)*100:.1f}%")
    else:
        print(f"  No test samples near upright in val set (unusual)")

    # Overall comparison
    with torch.no_grad():
        pred_all_before = model(x_val2)
        pred_all_after = model2(x_val2)
        mse_all_before = torch.nn.functional.mse_loss(pred_all_before, y_val2).item()
        mse_all_after = torch.nn.functional.mse_loss(pred_all_after, y_val2).item()

    print(f"\n  Overall val MSE before: {mse_all_before:.6f}")
    print(f"  Overall val MSE after:  {mse_all_after:.6f}")

    # Also plot uncertainty map AFTER exploration
    print(f"\n  Recomputing uncertainty map after exploration...")
    density2 = compute_training_density(model2, x_aug)
    _, _, _, unc_3d_after = build_uncertainty_map(model2, density2, grid_n=20)
    plot_uncertainty_map(cos_v, sin_v, thd_v, unc_3d_after, "explore_uncertainty_map_after.png")

    env.close()
    print(f"\nDone. Key outputs:")
    print(f"  explore_uncertainty_map.png      — uncertainty before exploration")
    print(f"  explore_uncertainty_map_after.png — uncertainty after exploration")
    print(f"  kan_explore_init.pt              — model before exploration")
    print(f"  kan_explore_augmented.pt         — model after exploration")


if __name__ == "__main__":
    main()
