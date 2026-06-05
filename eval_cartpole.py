"""Evaluate CartPole control pipeline.

Methods:
  1. Decision network: (state) -> (action, k)
  2. World model MPC: try both actions through world model, pick best
  3. Random baseline
"""
import torch, numpy as np, time, argparse
import gymnasium as gym
from kanrf import KAN
from cartpole_decision import CartPoleDecisionNet

MAX_K = 8


def test_decision_net(dn, n_episodes=10, render=False):
    """Test decision network: (state) -> (action, k), execute for k steps."""
    env = gym.make('CartPole-v1', render_mode='human' if render else None)
    rewards = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0
        for step in range(500):
            sn = torch.tensor([[obs[0]/2.5, obs[1]/3.0, obs[2]/0.3, obs[3]/3.0]],
                              dtype=torch.float32)
            with torch.no_grad():
                a_logit, k_cont = dn(sn)
                a = 1 if torch.sigmoid(a_logit).item() > 0.5 else 0
                k = max(1, min(MAX_K, round(k_cont.item() * 16)))

            for _ in range(k):
                obs, reward, term, trunc, _ = env.step(a)
                total_reward += reward
                if term or trunc:
                    break
            if term or trunc:
                break
        rewards.append(total_reward)

    env.close()
    success = sum(1 for r in rewards if r >= 500)
    print(f'Decision net: {success}/{n_episodes} (reward: {np.mean(rewards):.0f})')
    return success


def test_world_model_mpc(wm, n_episodes=10):
    """World model MPC: try both actions, pick the one with better predicted state."""
    env = gym.make('CartPole-v1')
    rewards = []
    a_onehot = torch.zeros(1, 2)

    for ep in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0
        for step in range(500):
            sn = torch.tensor([[obs[0]/2.5, obs[1]/3.0, obs[2]/0.3, obs[3]/3.0]],
                              dtype=torch.float32)

            # Try both actions through world model
            best_a, best_score = None, float('inf')
            for a in [0, 1]:
                a_onehot.zero_()
                a_onehot[0, a] = 1.0
                kn = torch.tensor([[4/16.0]])  # fixed k=4 for MPC
                x = torch.cat([sn, a_onehot, kn], dim=-1)
                with torch.no_grad():
                    pred = wm(x)
                score = abs(pred[0, 2].item()) * 0.5 + abs(pred[0, 0].item()) * 0.2
                if score < best_score:
                    best_score, best_a = score, a

            obs, reward, term, trunc, _ = env.step(best_a)
            total_reward += reward
            if term or trunc:
                break
        rewards.append(total_reward)

    env.close()
    success = sum(1 for r in rewards if r >= 500)
    print(f'World model MPC: {success}/{n_episodes} (reward: {np.mean(rewards):.0f})')
    return success


def test_random(n_episodes=10):
    """Random baseline."""
    env = gym.make('CartPole-v1')
    rewards = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0
        for _ in range(500):
            obs, reward, term, trunc, _ = env.step(env.action_space.sample())
            total_reward += reward
            if term or trunc:
                break
        rewards.append(total_reward)
    env.close()
    success = sum(1 for r in rewards if r >= 500)
    print(f'Random: {success}/{n_episodes} (reward: {np.mean(rewards):.0f})')
    return success


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dn', type=str, default='kan_cartpole_dn.pt')
    parser.add_argument('--wm', type=str, default='kan_cartpole.pt')
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)

    print(f'CartPole-v1 Evaluation ({args.episodes} episodes)')
    print('=' * 50)

    # Random baseline
    test_random(n_episodes=args.episodes)

    # World model MPC
    wm = KAN([7, 20, 4], grid_size=5, spline_order=3)
    wm.load_state_dict(torch.load(args.wm, weights_only=True))
    wm.eval()
    test_world_model_mpc(wm, n_episodes=args.episodes)

    # Decision network
    dn = CartPoleDecisionNet()
    dn.load_state_dict(torch.load(args.dn, weights_only=True))
    dn.eval()
    test_decision_net(dn, n_episodes=args.episodes, render=args.render)


if __name__ == '__main__':
    main()
