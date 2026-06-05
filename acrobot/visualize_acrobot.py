"""Visualize Acrobot: MPC episodes, state trajectories, model predictions."""
import torch, numpy as np, gymnasium as gym, sys, os, imageio
from kanrf import KAN
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

wm = KAN([10, 24, 6], grid_size=5, spline_order=3)
wm.load_state_dict(torch.load('acrobot_wm.pt', weights_only=True))
wm.eval()

a_oh = torch.zeros(1, 3)

def render_episode(seed, path, max_steps=500):
    """Render one Acrobot episode as GIF."""
    env = gym.make('Acrobot-v1', render_mode='rgb_array')
    obs, _ = env.reset(seed=seed)
    frames = [env.render()]
    record = {'t1': [], 't2': [], 'd1': [], 'd2': [], 'h': [], 'a': [],
              'pred_err': []}

    for step in range(min(max_steps, 200)):
        sn = torch.tensor([[obs[0],obs[1],obs[2],obs[3],obs[4]/6.0,obs[5]/8.0]], dtype=torch.float32)

        # MPC: try all 3 actions
        wm.eval()
        for p in wm.parameters(): p.requires_grad = False
        bs, ba = -float('inf'), 0
        for a in [0, 1, 2]:
            a_oh.zero_()
            a_oh[0, a] = 1.0
            with torch.no_grad():
                pred = wm(torch.cat([sn, a_oh, torch.tensor([[1/8.0]])], dim=-1))
            t1 = torch.atan2(pred[0, 1], pred[0, 0])
            t2 = torch.atan2(pred[0, 3], pred[0, 2])
            h = (-torch.cos(t1) - torch.cos(t1 + t2)).item()
            if h > bs: bs, ba = h, a

        # Prediction error
        a_oh.zero_()
        a_oh[0, ba] = 1.0
        with torch.no_grad():
            pred = wm(torch.cat([sn, a_oh, torch.tensor([[1/8.0]])], dim=-1))

        obs_before = obs.copy()
        obs, _, term, _, _ = env.step(ba)

        err = np.linalg.norm([pred[0, d].item() - obs[d] / ([1, 1, 1, 1, 6.0, 8.0][d])
                              for d in range(6)])

        t1_r = np.arctan2(obs[1], obs[0])
        t2_r = np.arctan2(obs[3], obs[2])
        record['t1'].append(np.rad2deg(t1_r))
        record['t2'].append(np.rad2deg(t2_r))
        record['d1'].append(obs[4])
        record['d2'].append(obs[5])
        record['h'].append(-np.cos(t1_r) - np.cos(t1_r + t2_r))
        record['a'].append(ba)
        record['pred_err'].append(err)

        if step % 5 == 0:
            frames.append(env.render())
        if term:
            frames.append(env.render())
            break

    env.close()
    if len(frames) > 150:
        stride = len(frames) // 120
        frames = frames[::stride]
    imageio.mimsave(path, frames, fps=15, loop=0)
    h_max = max(record['h'])
    goal_reached = h_max > 1.0
    print(f'  {path}: {len(frames)} frames, max_height={h_max:.2f} (goal=1.0) {"GOAL!" if goal_reached else ""}')
    return record


def plot_state_trajectories(records, labels, path):
    """Side-by-side state trajectory plots."""
    n = len(records)
    fig, axes = plt.subplots(3, n, figsize=(5*n, 12))

    for i, (rec, label) in enumerate(zip(records, labels)):
        t = np.arange(len(rec['t1']))

        ax = axes[0, i] if n > 1 else axes[0]
        ax.plot(t, rec['t1'], 'b-', alpha=0.7, linewidth=0.6, label='θ1')
        ax.plot(t, rec['t2'], 'r-', alpha=0.7, linewidth=0.6, label='θ2')
        ax.axhline(y=0, color='gray', alpha=0.3)
        ax.set_ylabel('Angle (deg)')
        ax.set_title(f'{label}: Joint Angles')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        ax = axes[1, i] if n > 1 else axes[1]
        ax.plot(t, rec['d1'], 'purple', alpha=0.7, linewidth=0.6, label='dθ1')
        ax.plot(t, rec['d2'], 'orange', alpha=0.7, linewidth=0.6, label='dθ2')
        ax.set_ylabel('Ang Vel (rad/s)')
        ax.set_title(f'{label}: Velocities')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        ax = axes[2, i] if n > 1 else axes[2]
        ax.plot(t, rec['h'], 'green', linewidth=0.8)
        ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Goal')
        # Color regions by action
        for j in range(len(t) - 1):
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
            ax.axvspan(j, j + 1, alpha=0.1, color=colors[rec['a'][j]])
        ax.set_ylabel('Tip Height')
        ax.set_xlabel('Step')
        ax.set_title(f'{label}: Goal Progress (colored by action)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'  State trajectories: {path}')


def plot_prediction_errors(records, labels, path):
    """Prediction error distributions and time series."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Time series
    for rec, label in zip(records, labels):
        axes[0].plot(rec['pred_err'], alpha=0.7, linewidth=0.6, label=label)
    axes[0].set_ylabel('Prediction Error (L2 norm)')
    axes[0].set_xlabel('Step')
    axes[0].set_title('Model Prediction Error over Time')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Histogram
    all_errs = np.concatenate([rec['pred_err'] for rec in records])
    axes[1].hist(all_errs, bins=40, color='steelblue', edgecolor='white', alpha=0.8)
    axes[1].axvline(x=np.mean(all_errs), color='red', linestyle='--', label=f'mean={np.mean(all_errs):.3f}')
    axes[1].axvline(x=np.percentile(all_errs, 90), color='orange', linestyle='--',
                    label=f'P90={np.percentile(all_errs, 90):.3f}')
    axes[1].set_xlabel('Prediction Error')
    axes[1].set_title('Error Distribution')
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'  Prediction errors: {path}')


def plot_goal_landscape(wm, path):
    """Show what the world model predicts for a grid of states."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for a in [0, 1, 2]:
        # Grid over theta1, theta2 (fix velocities at 0)
        N = 40
        t1_vals = np.linspace(-np.pi, np.pi, N)
        t2_vals = np.linspace(-np.pi, np.pi, N)
        H = np.zeros((N, N))
        a_oh.zero_()
        a_oh[0, a] = 1.0

        for i, t1 in enumerate(t1_vals):
            for j, t2 in enumerate(t2_vals):
                sn = torch.tensor([[np.cos(t1), np.sin(t1), np.cos(t2), np.sin(t2),
                                    0.0, 0.0]], dtype=torch.float32)
                with torch.no_grad():
                    pred = wm(torch.cat([sn, a_oh, torch.tensor([[1 / 8.0]])], dim=-1))
                pt1 = torch.atan2(pred[0, 1], pred[0, 0]).item()
                pt2 = torch.atan2(pred[0, 3], pred[0, 2]).item()
                H[i, j] = -np.cos(pt1) - np.cos(pt1 + pt2)

        ax = axes[0, a]
        im = ax.pcolormesh(np.rad2deg(t2_vals), np.rad2deg(t1_vals), H,
                           cmap='RdYlGn', vmin=-2, vmax=2)
        ax.contour(np.rad2deg(t2_vals), np.rad2deg(t1_vals), H, levels=[1.0],
                   colors='black', linewidths=2)
        ax.set_xlabel('θ2 (deg)')
        ax.set_ylabel('θ1 (deg)')
        ax.set_title(f'Action {a}: Predicted Tip Height After 1 Step')
        plt.colorbar(im, ax=ax, label='Height')

    # Also show: which action gives highest predicted height at each state?
    for a in [0, 1, 2]:
        ax = axes[1, a]
        H_all = np.zeros((N, N, 3))
        for aa in [0, 1, 2]:
            a_oh.zero_()
            a_oh[0, aa] = 1.0
            for i, t1 in enumerate(t1_vals):
                for j, t2 in enumerate(t2_vals):
                    sn = torch.tensor([[np.cos(t1), np.sin(t1), np.cos(t2), np.sin(t2),
                                        0.0, 0.0]], dtype=torch.float32)
                    with torch.no_grad():
                        pred = wm(torch.cat([sn, a_oh, torch.tensor([[1 / 8.0]])], dim=-1))
                    pt1 = torch.atan2(pred[0, 1], pred[0, 0]).item()
                    pt2 = torch.atan2(pred[0, 3], pred[0, 2]).item()
                    H_all[i, j, aa] = -np.cos(pt1) - np.cos(pt1 + pt2)

        best_a = np.argmax(H_all, axis=2)
        ax.pcolormesh(np.rad2deg(t2_vals), np.rad2deg(t1_vals), best_a,
                       cmap='Set1', vmin=0, vmax=2)
        ax.set_xlabel('θ2 (deg)')
        ax.set_ylabel('θ1 (deg)')
        ax.set_title(f'Best Action Map ({["Red=0","Blue=1","Green=2"][a]})')

    plt.suptitle('Acrobot World Model: Predicted Goal Progress by Action', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f'  Goal landscape: {path}')


# === MAIN ===
print('Rendering Acrobot MPC episodes...')
records = []

# 1. Different random seeds (all fail)
for seed, label in [(0, 'Seed 0'), (10, 'Seed 10'), (42, 'Seed 42')]:
    rec = render_episode(seed, f'acrobot_mpc_seed{seed}.gif')
    rec['seed'] = seed
    rec['label'] = label
    records.append(rec)

# 2. Random action baseline
env = gym.make('Acrobot-v1', render_mode='rgb_array')
obs, _ = env.reset(seed=99)
frames = [env.render()]
for step in range(200):
    obs, _, term, _, _ = env.step(env.action_space.sample())
    if step % 5 == 0: frames.append(env.render())
    if term: break
env.close()
if len(frames) > 150:
    frames = frames[::len(frames)//120]
imageio.mimsave('acrobot_random.gif', frames, fps=15, loop=0)
print(f'  acrobot_random.gif: {len(frames)} frames (random baseline)')

# 3. State trajectories
plot_state_trajectories(records, [r['label'] for r in records],
                        'acrobot_trajectories.png')

# 4. Prediction errors
plot_prediction_errors(records, [r['label'] for r in records],
                       'acrobot_errors.png')

# 5. Goal landscape map
plot_goal_landscape(wm, 'acrobot_goal_landscape.png')

# 6. Summary text
print('\n=== FINAL STATE SUMMARY ===')
for rec in records:
    h = rec['h']
    goal_count = sum(1 for x in h if x > 1.0)
    print(f"  {rec['label']}: max_height={max(h):.3f}  "
          f"goal_steps={goal_count}/{len(h)}  "
          f"mean_err={np.mean(rec['pred_err']):.3f}")

print('\nOpen with: open acrobot_*.gif acrobot_*.png')
