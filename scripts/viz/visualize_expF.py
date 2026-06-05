"""Generate GIF for each trial from exp_F (energy-guided + smart burst, FULL).

Usage:
  python visualize_expF.py --model kan_pendulum_model_v6.pt --trials 10
"""
import sys, argparse, os, io
import torch
import gymnasium as gym
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from kanrf import KAN
from control.online_learning_v2 import ThreeFactorUpdater, compute_training_stats

G = 10.0; PI_2 = np.pi / 2; E_DES = G


def _normalize_state(s, device=None):
    t = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
    t[0, 2] /= 8.0
    return t.to(device) if device is not None else t


def _normalize_action(a, device=None):
    t = torch.tensor([[a / 2.0]], dtype=torch.float32)
    return t.to(device) if device is not None else t


def deviation(s_real, s_pred_norm, device=None):
    s_n = _normalize_state(s_real, device=device)
    return (s_n - s_pred_norm.to(s_n.device)).norm().item()


def find_action_energy(model, s_norm, s_target_norm, n_iters=50, lr=0.05,
                       lambda_ctrl=0.001, device=None):
    """Single-step energy-guided action optimization."""
    s_raw = s_norm.clone(); s_raw[0, 2] *= 8.0
    sin_th = s_raw[0, 1].item(); thd = s_raw[0, 2].item()
    near_upright = abs(s_raw[0, 0].item()) < 0.5 and sin_th > 0 and abs(thd) < 3.0
    w_E, w_pos = 1.0, (3.0 if near_upright else 0.0)

    a_n = torch.zeros(1, 1, device=device)
    torch.nn.init.uniform_(a_n, a=-0.3, b=0.3)
    a_n.requires_grad_(True)
    opt = torch.optim.Adam([a_n], lr=lr)

    for _ in range(n_iters):
        opt.zero_grad()
        x = torch.cat([s_norm, a_n], dim=-1)
        s_pred = model(x)
        E_pred = 0.5 * (s_pred[0, 2] * 8.0) ** 2 + G * s_pred[0, 1]
        loss_E = w_E * (E_pred - E_DES) ** 2
        loss_pos = torch.tensor(0.0, device=device)
        if w_pos > 0:
            loss_pos = w_pos * ((s_pred[0, :2] - s_target_norm[0, :2]) ** 2).sum()
        loss = loss_E + loss_pos + lambda_ctrl * (a_n ** 2).sum()
        loss.backward()
        opt.step()
        with torch.no_grad(): a_n.clamp_(-1.0, 1.0)

    with torch.no_grad():
        s_pred = model(torch.cat([s_norm, a_n], dim=-1))
    return a_n.detach().item() * 2.0, s_pred


def draw_pendulum(ax, s, a, step, info=""):
    """Draw pendulum frame."""
    ax.clear()
    ax.set_xlim(-1.8, 1.8); ax.set_ylim(-1.8, 1.8)
    ax.set_aspect('equal'); ax.axis('off')
    ax.set_facecolor('#1a1a2e')

    cos_th, sin_th, thd = s
    pivot = np.array([0.0, 0.0])
    bob = np.array([cos_th, sin_th])
    angle_err = abs(np.arctan2(sin_th, cos_th) - PI_2)

    ax.add_patch(patches.Circle(pivot, 0.06, color='#888888', zorder=3))
    if angle_err < 0.2: color = '#06d6a0'
    elif abs(thd) > 4: color = '#ff6b35'
    else: color = '#00b4d8'
    ax.plot([pivot[0], bob[0]], [pivot[1], bob[1]], color=color, linewidth=3, zorder=2)
    ax.add_patch(patches.Circle(bob, 0.12, color=color, ec='white', linewidth=1, zorder=3))
    ax.plot([0.0, 0.0], [0.0, 1.0], 'o', color='white', markersize=6, alpha=0.3)

    if abs(thd) > 0.01:
        v_dir = np.array([-sin_th, cos_th]) * np.sign(thd)
        v_len = min(abs(thd) * 0.15, 0.6)
        ax.arrow(bob[0], bob[1], v_dir[0]*v_len, v_dir[1]*v_len,
                head_width=0.06, head_length=0.08, fc='#ffff00', ec='#ffff00', alpha=0.7)
    if abs(a) > 0.05:
        t_dir = np.array([-sin_th, cos_th]) * np.sign(a)
        t_len = min(abs(a) * 0.3, 0.4)
        t_start = bob - t_dir * 0.14
        ax.arrow(t_start[0], t_start[1], t_dir[0]*t_len, t_dir[1]*t_len,
                head_width=0.08, head_length=0.1, fc='#ff0000', ec='#ff0000', alpha=0.9, width=0.03)

    E = 0.5 * thd * thd + G * sin_th
    ax.text(0.02, 0.98, f'Step {step}  {info}', transform=ax.transAxes,
            fontsize=11, color='white', va='top', fontweight='bold', fontfamily='monospace')
    ax.text(0.02, 0.92, f'|dth|={angle_err:.3f}rad  E={E:+.2f}  a={a:+.3f}',
            transform=ax.transAxes, fontsize=9, color='#ccc', va='top', fontfamily='monospace')


def capture_frame(obs, a, step, info=""):
    fig, ax = plt.subplots(figsize=(6, 6))
    draw_pendulum(ax, obs, a, step, info)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=72, bbox_inches='tight', pad_inches=0.1, facecolor='#1a1a2e')
    buf.seek(0)
    frame = Image.open(buf).copy()
    plt.close(fig); buf.close()
    return frame


# ─── Smart Burst (from exp_F) ──────────────────────────────────────────────

def smart_burst(model, env, updater, s_target_norm, obs_start, device,
                n_steps=5, burst_eta_mult=20, n_extra_passes=5):
    """Active model verification at failure point. Returns (final_obs, n_updates)."""
    original_eta0 = updater.eta0
    updater.eta0 = original_eta0 * burst_eta_mult

    transitions = []; obs = obs_start; n_updates = 0

    for step in range(n_steps):
        s_norm = _normalize_state(obs, device=device)
        a, s_pred = find_action_energy(model, s_norm, s_target_norm, n_iters=50, device=device)
        obs_next, _, _, _, _ = env.step([a])
        a_norm = _normalize_action(a, device=device)
        s_next_norm = _normalize_state(obs_next, device=device)
        transitions.append((s_norm, a_norm, s_next_norm, s_pred))
        updater.update(s_norm, a_norm, s_next_norm)
        n_updates += 1
        obs = obs_next

    for _ in range(n_extra_passes):
        for (s_norm, a_norm, s_next_norm, _) in transitions:
            updater.update(s_norm, a_norm, s_next_norm)
            n_updates += 1

    updater.eta0 = original_eta0
    return obs, n_updates


# ─── Full trial matching exp_F ─────────────────────────────────────────────

def run_trial_full(model, updater, stats, env, s_target_norm, seed, device,
                   total_steps=60, n_iters=50, dev_threshold_hard=None):
    """Full exp_F trial: energy-guided + smart burst + online learning."""
    if dev_threshold_hard is None:
        dev_threshold_hard = 2.5 * stats['sigma_train']

    obs, _ = env.reset(seed=seed)
    frames = []; step_count = 0; burst_state = False
    angles = []

    while step_count < total_steps:
        s_norm = _normalize_state(obs, device=device)

        # Energy-guided action
        a, s_pred = find_action_energy(model, s_norm, s_target_norm, n_iters=n_iters, device=device)

        # Draw BEFORE execute
        info = "BURST" if burst_state else ""
        frames.append(capture_frame(obs, a, step_count, info))
        burst_state = False

        s_before = obs.copy()
        obs_next, _, term, trunc, _ = env.step([a])
        step_count += 1

        # Deviation check
        dev = deviation(obs_next, s_pred, device=device)

        # Online update
        a_norm = _normalize_action(a, device=device)
        s_next_norm = _normalize_state(obs_next, device=device)
        updater.update(s_norm, a_norm, s_next_norm)

        # Smart burst if deviation > threshold
        if dev > dev_threshold_hard:
            obs, _ = smart_burst(model, env, updater, s_target_norm, obs_next, device,
                                n_steps=5, burst_eta_mult=20, n_extra_passes=5)
            burst_state = True  # next frame will show BURST label
        else:
            obs = obs_next

        angles.append(abs(np.arctan2(obs[1], obs[0]) - PI_2))
        if angles[-1] < 0.2 or term or trunc:
            break

    return frames, angles, step_count


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='kan_pendulum_model_v6.pt')
    parser.add_argument('--data', type=str, default='pendulum_data_v4.pt')
    parser.add_argument('--trials', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out-dir', type=str, default='.')
    parser.add_argument('--fps', type=int, default=10)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(args.device)

    # Load model
    ckpt = torch.load(args.model, weights_only=True)
    layer_dims = [4]
    for key in sorted(ckpt.keys()):
        if 'base_weight' in key: layer_dims.append(ckpt[key].shape[0])
    model = KAN(layer_dims, grid_size=5, spline_order=3)
    model.load_state_dict(ckpt)
    model = model.to(device)

    # Stats & updater
    data = torch.load(args.data, weights_only=True)
    x_train, y_train = data if isinstance(data, tuple) else (data[0], data[1])
    n_stats = min(len(x_train), 5000)
    idx = torch.randperm(len(x_train))[:n_stats]
    stats = compute_training_stats(model, x_train[idx].to(device), y_train[idx].to(device))
    updater = ThreeFactorUpdater(model, stats, eta0=1e-3)
    dev_threshold_hard = 2.5 * stats['sigma_train']

    env = gym.make("Pendulum-v1")
    s_target_norm = torch.tensor([[0.0, 1.0, 0.0]], device=device)
    print(f"sigma={stats['sigma_train']:.4f}  thresh={dev_threshold_hard:.4f}")

    print(f"\n{'Trial':>5s}  {'seed':>5s}  {'|dth0|':>8s}  {'|dth_f|':>9s}  {'frames':>6s}  {'result':>6s}  {'output'}")
    print(f"{'─'*5}  {'─'*5}  {'─'*8}  {'─'*9}  {'─'*6}  {'─'*6}  {'─'*30}")

    for t in range(args.trials):
        trial_seed = args.seed + t * 100
        obs0, _ = env.reset(seed=trial_seed)
        init_err = abs(np.arctan2(obs0[1], obs0[0]) - PI_2)

        frames, angles, steps = run_trial_full(
            model, updater, stats, env, s_target_norm, trial_seed, device,
            total_steps=60, n_iters=50, dev_threshold_hard=dev_threshold_hard)

        success = angles[-1] < 0.2 if angles else False
        fname = f'trial_{t+1:02d}_expF_burst.gif'
        out_path = os.path.join(args.out_dir, fname)
        frames[0].save(out_path, save_all=True, append_images=frames[1:],
                       duration=int(1000/args.fps), loop=0, optimize=True)
        print(f"  {t+1:4d}  {trial_seed:5d}  {init_err:8.3f}  {angles[-1]:9.4f}  "
              f"{len(frames):6d}  {'Y' if success else 'N':>6s}  {fname}")

    env.close()
    print(f"\nDone.")


if __name__ == "__main__":
    main()
