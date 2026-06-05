"""Render pendulum control trial as GIF animation.

Usage:
  python visualize_trial.py --model kan_pendulum_model.pt --output trial.gif
"""
import sys, argparse, os, io
import torch
import gymnasium as gym
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import animation
from PIL import Image
from kanrf import KAN
from strategy_v2 import compute_gap, desired_velocity, strategy_mode
from execute_v2 import execute_v2


def draw_pendulum(ax, s, a, v_des, mode, step, s_goal=np.array([0, 1, 0])):
    """Draw one frame of the pendulum.

    ax: matplotlib axis
    s: (3,) [cosθ, sinθ, θ̇] — current state
    a: scalar torque
    v_des: (3,) desired velocity
    mode: strategy mode string
    step: step number
    """
    ax.clear()
    ax.set_xlim(-1.8, 1.8)
    ax.set_ylim(-1.8, 1.8)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_facecolor('#1a1a2e')

    cos_th, sin_th, thd = s

    # Pendulum pivot (top center)
    pivot = np.array([0.0, 0.0])
    # Pendulum bob position: angle measured such that (cos, sin) = tip position
    bob = np.array([cos_th, sin_th])

    # Draw pivot
    ax.add_patch(patches.Circle(pivot, 0.06, color='#888888', zorder=3))

    # Draw rod
    color_map = {'swing_up': '#ff6b35', 'brake': '#00b4d8', 'stabilize': '#06d6a0'}
    color = color_map.get(mode, '#ffffff')
    lw = 3 if abs(a) > 1.0 else 2
    ax.plot([pivot[0], bob[0]], [pivot[1], bob[1]], color=color, linewidth=lw, zorder=2)

    # Draw bob
    bob_size = 0.12 + min(abs(thd) / 10, 0.1)
    ax.add_patch(patches.Circle(bob, bob_size, color=color, ec='white', linewidth=1, zorder=3))

    # Draw target (upright)
    target = np.array([0.0, 1.0])
    ax.plot(target[0], target[1], 'o', color='#ffffff', markersize=6, alpha=0.3, zorder=1)
    ax.plot([pivot[0], target[0]], [pivot[1], target[1]], '--', color='#ffffff',
            linewidth=0.5, alpha=0.15, zorder=1)

    # Draw velocity as arrow from bob
    if abs(thd) > 0.01:
        v_scale = 0.15
        # Tangential direction: perpendicular to (cos, sin)
        v_dir = np.array([-sin_th, cos_th]) * np.sign(thd)
        v_len = min(abs(thd) * v_scale, 0.6)
        v_end = bob + v_dir * v_len
        ax.arrow(bob[0], bob[1], v_dir[0]*v_len, v_dir[1]*v_len,
                head_width=0.06, head_length=0.08, fc='#ffff00', ec='#ffff00',
                alpha=0.7, zorder=4)

    # Draw torque direction
    if abs(a) > 0.05:
        torque_scale = 0.3
        torque_dir = np.array([-sin_th, cos_th]) * np.sign(a)
        t_len = min(abs(a) * torque_scale, 0.4)
        t_start = bob - torque_dir * (bob_size + 0.02)
        ax.arrow(t_start[0], t_start[1], torque_dir[0]*t_len, torque_dir[1]*t_len,
                head_width=0.08, head_length=0.1, fc='#ff0000', ec='#ff0000',
                alpha=0.9, zorder=5, width=0.03)

    # Info text
    angle = np.arctan2(sin_th, cos_th)
    angle_err = abs(angle - np.pi/2)
    E = 0.5 * thd * thd + 10 * sin_th

    ax.text(0.02, 0.98, f'Step {step}  |  {mode}',
            transform=ax.transAxes, fontsize=11, color='white',
            va='top', fontweight='bold', fontfamily='monospace')
    ax.text(0.02, 0.92, f'|Δθ|={angle_err:.3f}rad  E={E:+.2f}  a={a:+.3f}',
            transform=ax.transAxes, fontsize=9, color='#cccccc',
            va='top', fontfamily='monospace')
    ax.text(0.02, 0.86, f'θ̇={thd:+.2f} rad/s',
            transform=ax.transAxes, fontsize=9, color='#999999',
            va='top', fontfamily='monospace')

    # Legend
    ax.text(0.98, 0.98, '● target', transform=ax.transAxes, fontsize=8,
            color='white', va='top', ha='right', alpha=0.4)
    ax.text(0.98, 0.94, '→ velocity', transform=ax.transAxes, fontsize=8,
            color='#ffff00', va='top', ha='right', alpha=0.7)
    ax.text(0.98, 0.90, '⇒ torque', transform=ax.transAxes, fontsize=8,
            color='#ff0000', va='top', ha='right', alpha=0.9)


def run_and_capture(model, env, s_goal, total_steps=60, verbose=1):
    """Run one trial, capture frames."""
    obs, _ = env.reset()
    frames = []

    for step in range(total_steps):
        s_now = obs

        # Strategy
        gap = compute_gap(s_now)
        mode = strategy_mode(gap)
        v_des = desired_velocity(gap, mode)

        s_tensor = torch.tensor(s_now, dtype=torch.float32).unsqueeze(0)
        v_des_tensor = torch.tensor(v_des, dtype=torch.float32).unsqueeze(0)

        # Execution
        a, _, _, _ = execute_v2(model, s_tensor, v_des_tensor, n_iter=15)

        # Draw
        fig, ax = plt.subplots(figsize=(6, 6))
        draw_pendulum(ax, s_now, a, v_des, mode, step)
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=72, bbox_inches='tight', pad_inches=0.1,
                   facecolor='#1a1a2e')
        buf.seek(0)
        frames.append(Image.open(buf).copy())
        plt.close(fig)
        buf.close()

        # Step env
        obs_next, _, terminated, truncated, _ = env.step([a])
        obs = obs_next
        if terminated or truncated:
            break

    return frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='kan_pendulum_model.pt')
    parser.add_argument('--output', type=str, default='trial.gif')
    parser.add_argument('--steps', type=int, default=60)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--fps', type=int, default=10)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = KAN([4, 12, 3], grid_size=5, spline_order=3)
    model.load_state_dict(torch.load(args.model, weights_only=True))
    env = gym.make("Pendulum-v1")
    s_goal = np.array([0.0, 1.0, 0.0])

    print(f"Running trial ({args.steps} steps)...")
    frames = run_and_capture(model, env, s_goal, total_steps=args.steps)

    # Save GIF
    frames[0].save(
        args.output,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / args.fps),
        loop=0,
        optimize=True,
    )
    print(f"Saved: {args.output} ({len(frames)} frames, {args.fps} fps)")
    env.close()


if __name__ == "__main__":
    main()
