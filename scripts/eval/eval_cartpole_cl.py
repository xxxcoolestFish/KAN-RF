"""CartPole: World model MPC + continuous learning.

At each step:
  1. Try both actions through frozen world model
  2. Pick the one predicting more balanced state
  3. Execute, observe real next state
  4. Feed to ContinuousLearner
  5. After episode, fine-tune world model

This requires zero system-specific knowledge — just the world model + MPC.
"""
import torch, numpy as np, time, argparse, copy
import gymnasium as gym
from kanrf import KAN
from control.continuous_learner import ContinuousLearner


def load_wm(path='kan_cartpole.pt'):
    model = KAN([7, 20, 4], grid_size=5, spline_order=3)
    model.load_state_dict(torch.load(path, weights_only=True))
    return model


def mpc_action(wm, s_norm, k=4):
    """Try both actions through world model, pick best."""
    a_onehot = torch.zeros(1, 2)
    kn = torch.tensor([[k / 16.0]])
    best_a, best_score = 0, float('inf')

    for a in [0, 1]:
        a_onehot.zero_()
        a_onehot[0, a] = 1.0
        x = torch.cat([s_norm, a_onehot, kn], dim=-1)
        with torch.no_grad():
            pred = wm(x)
        score = abs(pred[0, 2]) * 0.5 + abs(pred[0, 0]) * 0.2  # pole_angle + cart_pos
        if score < best_score:
            best_score, best_a = score, a

    return best_a, best_score


def run_episode(wm, learner, env):
    obs, _ = env.reset()
    total_reward = 0
    step_count = 0

    for _ in range(500):
        s_norm = torch.tensor(
            [[obs[0]/2.5, obs[1]/3.0, obs[2]/0.3, obs[3]/3.0]],
            dtype=torch.float32)

        wm.eval()
        for p in wm.parameters():
            p.requires_grad = False

        a, _ = mpc_action(wm, s_norm, k=4)
        s_before = obs.copy()

        obs, reward, term, trunc, _ = env.step(a)
        total_reward += reward
        step_count += 1

        # Record transition — only if model is WRONG (large prediction error)
        s_n = torch.tensor(
            [[s_before[0]/2.5, s_before[1]/3.0, s_before[2]/0.3, s_before[3]/3.0]],
            dtype=torch.float32)
        a_onehot = torch.zeros(1, 2)
        a_onehot[0, a] = 1.0
        k_n = torch.tensor([[4/16.0]])
        s_next_n = torch.tensor(
            [[obs[0]/2.5, obs[1]/3.0, obs[2]/0.3, obs[3]/3.0]],
            dtype=torch.float32)

        x_full = torch.cat([s_n, a_onehot, k_n], dim=-1)
        with torch.no_grad():
            err = (wm(x_full) - s_next_n).norm().item()
        learner.error_log.append(err)

        # Only add to buffer if model was surprised (error > threshold)
        if err > learner.error_threshold:
            learner.buffer_x.append(x_full.detach())
            learner.buffer_y.append(s_next_n.detach())

        if term or trunc:
            break

    return total_reward >= 475, total_reward, step_count


def main(model_path='kan_cartpole.pt', n_episodes=15, lr=1e-4, ft_epochs=20):
    torch.manual_seed(42)
    np.random.seed(42)

    wm = load_wm(model_path)
    wm_orig = copy.deepcopy(wm)

    # Load original training data as replay buffer to prevent forgetting
    x_orig, y_orig = torch.load('cartpole_data_ms.pt', weights_only=True)
    n_replay = 5000
    idx = torch.randperm(len(x_orig))[:n_replay]

    learner = ContinuousLearner(wm, lr=lr, error_threshold=0.08)
    learner.replay_x = x_orig[idx]
    learner.replay_y = y_orig[idx]

    env = gym.make('CartPole-v1')

    print(f'CartPole MPC + Continuous Learning (error-gated + replay)')
    print(f'  World model: [7,20,4]  |  lr={lr}  |  ft_epochs={ft_epochs}')
    print(f'  Replay buffer: {n_replay} original transitions')
    print()

    # Training episodes
    rewards = []
    for ep in range(1, n_episodes + 1):
        ok, reward, steps = run_episode(wm, learner, env)
        loss = learner.fine_tune(epochs=ft_epochs)
        rewards.append(reward)

        s = learner.summary()
        ok_str = 'OK' if ok else 'FAIL'
        print(f'  Ep {ep:2d}: {ok_str:>4s}  reward={reward:3.0f}  steps={steps:3d}  '
              f'err={s["mean_error"]:.4f}  buffer={s["buffer_size"]}  '
              f'loss={loss:.5f}')

    # Final test: 20 episodes, no further fine-tuning
    sep = '=' * 50
    print(f'\n{sep}')
    print('FINAL TEST: 20 episodes, no further learning')
    print(sep)

    def test_model(model, label, n=20):
        buf = ContinuousLearner(model, lr=0)  # lr=0 = no updates
        ok = 0; rwds = []
        for _ in range(n):
            is_ok, r, steps = run_episode(model, buf, env)
            if is_ok: ok += 1
            rwds.append(r)
        print(f'{label}: {ok}/{n} (avg reward: {np.mean(rwds):.0f})')
        return ok

    test_model(wm, 'After continuous learning')
    test_model(wm_orig, 'Original model')

    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='kan_cartpole.pt')
    parser.add_argument('--episodes', type=int, default=15)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--ft-epochs', type=int, default=20)
    args = parser.parse_args()
    main(model_path=args.model, n_episodes=args.episodes,
         lr=args.lr, ft_epochs=args.ft_epochs)
