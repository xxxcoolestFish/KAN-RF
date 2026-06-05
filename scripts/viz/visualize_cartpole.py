"""Visualize CartPole MPC: success and failure trajectories.

Shows state evolution, chosen actions, and model prediction vs reality.
"""
import torch, numpy as np, gymnasium as gym, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from kanrf import KAN


def run_and_record(seed, max_steps=500):
    wm = KAN([7, 20, 4], grid_size=5, spline_order=3)
    wm.load_state_dict(torch.load('kan_cartpole.pt', weights_only=True))
    wm.eval()

    env = gym.make('CartPole-v1')
    obs, _ = env.reset(seed=seed)
    a_oh = torch.zeros(1, 2)
    kn = torch.tensor([[4 / 16.0]])

    record = {
        'cart_pos': [], 'cart_vel': [], 'pole_angle': [], 'pole_ang_vel': [],
        'actions': [], 'model_scores': [],
        'pred_errors': [],
    }

    for step in range(max_steps):
        sn = torch.tensor(
            [[obs[0]/2.5, obs[1]/3.0, obs[2]/0.3, obs[3]/3.0]],
            dtype=torch.float32)

        # MPC: try both actions
        scores = {}
        for a in [0, 1]:
            a_oh.zero_()
            a_oh[0, a] = 1.0
            x = torch.cat([sn, a_oh, kn], dim=-1)
            with torch.no_grad():
                pred = wm(x)
            scores[a] = abs(pred[0, 2].item()) * 0.5 + abs(pred[0, 0].item()) * 0.2

        best_a = 0 if scores[0] < scores[1] else 1

        # Record prediction for the chosen action
        a_oh.zero_()
        a_oh[0, best_a] = 1.0
        x = torch.cat([sn, a_oh, kn], dim=-1)
        with torch.no_grad():
            pred = wm(x)
        pred_real = [p.item() for p in pred[0]]

        s_before = obs.copy()
        obs, _, term, trunc, _ = env.step(best_a)

        # Prediction error
        err = np.linalg.norm([
            pred_real[0] - obs[0]/2.5,
            pred_real[1] - obs[1]/3.0,
            pred_real[2] - obs[2]/0.3,
            pred_real[3] - obs[3]/3.0,
        ])

        record['cart_pos'].append(obs[0])
        record['cart_vel'].append(obs[1])
        record['pole_angle'].append(np.rad2deg(obs[2]))
        record['pole_ang_vel'].append(obs[3])
        record['actions'].append(best_a)
        record['model_scores'].append((scores[0], scores[1]))
        record['pred_errors'].append(err)

        if term or trunc:
            break

    env.close()
    return record


def plot_episode(record, seed, fig_path):
    steps = len(record['actions'])
    t = np.arange(steps)

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))

    # Row 1: cart position + velocity
    ax = axes[0, 0]
    ax.plot(t, record['cart_pos'], 'b-', linewidth=0.8)
    ax.axhline(y=2.4, color='r', linestyle='--', alpha=0.3, label='boundary')
    ax.axhline(y=-2.4, color='r', linestyle='--', alpha=0.3)
    ax.set_ylabel('Cart Position (m)')
    ax.set_title(f'Cart Position | Seed={seed} | {steps} steps')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(t, record['cart_vel'], 'purple', linewidth=0.8)
    ax.set_ylabel('Cart Velocity (m/s)')
    ax.set_title('Cart Velocity')
    ax.grid(True, alpha=0.3)

    # Row 2: pole angle + angular velocity
    ax = axes[1, 0]
    ax.plot(t, record['pole_angle'], 'r-', linewidth=0.8)
    ax.axhline(y=12, color='gray', linestyle='--', alpha=0.5, label='12 deg')
    ax.axhline(y=-12, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=0, color='green', linestyle='-', alpha=0.2)
    ax.set_ylabel('Pole Angle (deg)')
    ax.set_title('Pole Angle')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(t, record['pole_ang_vel'], 'orange', linewidth=0.8)
    ax.set_ylabel('Pole Ang Vel (rad/s)')
    ax.set_title('Pole Angular Velocity')
    ax.grid(True, alpha=0.3)

    # Row 3: actions + prediction error
    ax = axes[2, 0]
    s0 = [s[0] for s in record['model_scores']]
    s1 = [s[1] for s in record['model_scores']]
    ax.plot(t, s0, 'blue', linewidth=0.6, alpha=0.7, label='action 0 score')
    ax.plot(t, s1, 'red', linewidth=0.6, alpha=0.7, label='action 1 score')
    chosen = np.array(record['actions'])
    for i in range(len(t)):
        color = 'blue' if chosen[i] == 0 else 'red'
        ax.axvline(x=i, color=color, alpha=0.1, linewidth=1)
    ax.set_ylabel('Model Score (lower=better)')
    ax.set_title('MPC Action Scores + Chosen Actions')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[2, 1]
    ax.plot(t, record['pred_errors'], 'green', linewidth=0.8)
    ax.axhline(y=0.08, color='orange', linestyle='--', alpha=0.5, label='error threshold')
    ax.set_ylabel('Prediction Error (norm)')
    ax.set_xlabel('Step')
    ax.set_title('World Model Prediction Error')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(fig_path, dpi=120)
    plt.close()
    print(f'  Saved: {fig_path}')


def main():
    print('CartPole MPC Visualization')
    print('=' * 50)
    torch.manual_seed(42)
    np.random.seed(42)

    # Success: seed=1, 500 steps
    rec = run_and_record(1, max_steps=500)
    plot_episode(rec, 1, 'cartpole_success_seed1.png')

    # Late failure: seed=0, 406 steps
    rec = run_and_record(0, max_steps=500)
    plot_episode(rec, 0, 'cartpole_failure_seed0.png')


if __name__ == '__main__':
    main()
