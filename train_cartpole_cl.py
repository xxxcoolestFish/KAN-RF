"""CartPole: decision network learns k from environment feedback.

Like Pendulum Approach 1: periodically try alternative (a, k) pairs from the
same state, compare real outcomes, record corrections, fine-tune decision net.
"""
import torch, numpy as np, gymnasium as gym, time, argparse, sys
from kanrf import KAN


def run_and_learn(dn, wm, env, explore_prob=0.08, max_steps=500):
    """One episode: DN chooses k (prediction horizon), MPC selects action.
    k is learned from environment feedback."""
    obs, _ = env.reset()
    corrections = []
    a_oh = torch.zeros(1, 2)

    for step in range(max_steps):
        sn = torch.tensor(
            [[obs[0]/2.5, obs[1]/3.0, obs[2]/0.3, obs[3]/3.0]],
            dtype=torch.float32)

        # DN chooses prediction horizon k
        with torch.no_grad():
            out = dn(sn)
            k_dn = max(1, min(16, round(out[0, 1].item() * 16)))

        # Explore: try different k values, compare real outcomes
        if np.random.random() < explore_prob:
            # MPC action (with k=2 for baseline)
            bs, ba = float('inf'), 0
            for a in [0, 1]:
                a_oh.zero_()
                a_oh[0, a] = 1.0
                with torch.no_grad():
                    pred = wm(torch.cat([sn, a_oh, torch.tensor([[2/16.0]])], dim=-1))
                s = abs(pred[0, 2]) * 0.5 + abs(pred[0, 0]) * 0.2
                if s < bs: bs, ba = s, a

            saved_state = env.unwrapped.state
            best_k, best_reward = k_dn, -float('inf')
            for k_try in [2, 4, 8, 16]:
                r = _evaluate_action(env, saved_state, ba, k_try, max_steps - step)
                if r > best_reward:
                    best_reward, best_k = r, k_try
            env.unwrapped.state = saved_state

            if best_k != k_dn:
                corrections.append({
                    'state': sn.squeeze(0).numpy().copy(),
                    'k_cont': best_k / 16.0,
                })
            k_use = best_k
        else:
            k_use = k_dn

        # MPC scores actions using chosen k as lookahead horizon
        wm.eval()
        for p in wm.parameters():
            p.requires_grad = False
        kn = torch.tensor([[k_use / 16.0]])
        bs, ba = float('inf'), 0
        for a in [0, 1]:
            a_oh.zero_()
            a_oh[0, a] = 1.0
            with torch.no_grad():
                pred = wm(torch.cat([sn, a_oh, kn], dim=-1))
            score = abs(pred[0, 2]) * 0.5 + abs(pred[0, 0]) * 0.2
            if score < bs: bs, ba = score, a

        # Execute 1 step (high-frequency replanning at 50Hz)
        obs, _, term, trunc, _ = env.step(ba)
        if term or trunc:
            break

    return step >= 475, corrections, step + 1


def _evaluate_action(env, state, a, k, remaining_steps):
    """Execute (a) for k steps or until termination.  Return avg reward per step."""
    env.unwrapped.state = state
    total_reward = 0
    steps = 0
    for _ in range(min(k, remaining_steps)):
        obs, reward, term, trunc, _ = env.step(a)
        total_reward += reward
        steps += 1
        if term or trunc:
            break
    env.unwrapped.state = state  # restore
    return total_reward / max(steps, 1)


def fine_tune_dn(dn, corrections, epochs=200, lr=1e-3):
    """Fine-tune decision network on corrected k labels only."""
    if len(corrections) < 4:
        return 0.0
    s_data = torch.tensor(np.array([c['state'] for c in corrections]))
    k_data = torch.tensor([[c['k_cont']] for c in corrections])

    dn.train()
    opt = torch.optim.Adam(dn.parameters(), lr=lr)
    mse = torch.nn.MSELoss()

    for _ in range(epochs):
        opt.zero_grad()
        out = dn(s_data)
        loss = mse(out[:, 1:2], k_data)  # only train k output
        loss.backward()
        opt.step()

    dn.eval()
    return loss.item()


def test(dn, wm, n_episodes=100):
    """Test: MPC scores actions using k from decision network (lookahead horizon).
    Execution is always 1 step (50Hz replanning for CartPole's fast dynamics).
    Decision network's k controls prediction horizon, not execution horizon."""
    env = gym.make('CartPole-v1')
    a_oh = torch.zeros(1, 2)
    ok = 0; k_used = []
    for seed in range(n_episodes):
        obs, _ = env.reset(seed=seed)
        for step in range(500):
            sn = torch.tensor(
                [[obs[0]/2.5, obs[1]/3.0, obs[2]/0.3, obs[3]/3.0]],
                dtype=torch.float32)
            # DN chooses prediction horizon k
            with torch.no_grad():
                k_dn = dn(sn)[0, 1].item()
            k = max(1, min(16, round(k_dn * 16)))
            k_used.append(k)
            # MPC scores actions using DN's k as lookahead
            kn = torch.tensor([[k/16.0]])
            bs, ba = float('inf'), 0
            for a in [0, 1]:
                a_oh.zero_()
                a_oh[0, a] = 1.0
                with torch.no_grad():
                    pred = wm(torch.cat([sn, a_oh, kn], dim=-1))
                score = abs(pred[0, 2]) * 0.5 + abs(pred[0, 0]) * 0.2
                if score < bs: bs, ba = score, a
            # Always execute 1 step (high-frequency replanning)
            obs, _, term, trunc, _ = env.step(ba)
            if term or trunc:
                break
        if step >= 475:
            ok += 1
    env.close()
    return ok, np.mean(k_used) if k_used else 0


def main(n_episodes=30, lr=1e-3):
    torch.manual_seed(42)
    np.random.seed(42)

    # Start from the k={2,16} decision net (best baseline)
    dn = KAN([4, 12, 2], grid_size=5, spline_order=3)
    dn.load_state_dict(
        torch.load('/Users/zhuangxinyu/KAN/KAN-RF/kan_cartpole_dn_v3.pt',
                   weights_only=True))
    dn.eval()

    # Load world model (for MPC action selection)
    wm = KAN([7, 20, 4], grid_size=5, spline_order=3)
    wm.load_state_dict(
        torch.load('/Users/zhuangxinyu/KAN/KAN-RF/kan_cartpole.pt', weights_only=True))
    wm.eval()

    env = gym.make('CartPole-v1')
    all_corrections = []
    ok_hist = []

    print(f'CartPole: MPC for action + Decision Network learns k from env feedback')
    print(f'  Episodes: {n_episodes}  |  lr={lr}\n')

    for ep in range(1, n_episodes + 1):
        ok, corrections, steps = run_and_learn(dn, wm, env, explore_prob=0.08)

        all_corrections.extend(corrections)
        total_corr = len(all_corrections)

        # Fine-tune after each episode
        loss = fine_tune_dn(dn, all_corrections, epochs=100, lr=lr)

        # Test every 5 episodes
        test_ok, test_k = test(dn, wm, n_episodes=100) if ep % 5 == 0 else (None, None)
        ok_hist.append(test_ok)

        test_str = f'  test_100={test_ok} k_avg={test_k:.1f}' if test_ok is not None else ''
        print(f'  Ep {ep:2d}: {"OK" if ok else "FAIL"}  steps={steps:3d}  '
              f'corr_ep={len(corrections)}  total_corr={total_corr}  '
              f'loss={loss:.4f}{test_str}')

    # Final test
    print(f'\nFinal test (500 seeds): ', end='', flush=True)
    final_ok, final_k = test(dn, wm, n_episodes=500)
    print(f'{final_ok}/500 ({final_ok/5:.0f}%)  avg_k={final_k:.1f}')

    torch.save(dn.state_dict(), '/Users/zhuangxinyu/KAN/KAN-RF/kan_cartpole_dn_cl.pt')
    print('Saved: kan_cartpole_dn_cl.pt')
    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()
    main(n_episodes=args.episodes, lr=args.lr)
