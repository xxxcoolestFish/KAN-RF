"""Acrobot: MPC + decision network for k + environment feedback learning.

World model: KAN([10,24,6]) — f(s, a_onehot, k) -> s'
Decision net: KAN([6,12,1]) — state -> k_cont
MPC: try all 3 actions through world model, pick the one closest to goal.
Exploration: save state, try k={2,4,8}, compare real outcomes.
"""
import torch, numpy as np, gymnasium as gym, time, argparse, sys, os
from kanrf import KAN

K_VALS = [1, 2, 4, 8]
MAX_EPISODE = 500


def goal_score_from_obs(obs):
    """Lower = closer to goal. Goal: -cos(t1) - cos(t1+t2) > 1.0"""
    cos1, sin1, cos2, sin2 = obs[0], obs[1], obs[2], obs[3]
    t1 = np.arctan2(sin1, cos1)
    t2 = np.arctan2(sin2, cos2)
    tip_height = -np.cos(t1) - np.cos(t1 + t2)
    return max(0.0, 1.0 - tip_height)


def goal_score_from_state(s_norm):
    """From normalized 6D state prediction."""
    cos1, sin1 = s_norm[:, 0], s_norm[:, 1]
    cos2, sin2 = s_norm[:, 2], s_norm[:, 3]
    t1 = torch.atan2(sin1, cos1)
    t2 = torch.atan2(cos2, sin2)
    tip_height = -torch.cos(t1) - torch.cos(t1 + t2)
    score = torch.clamp(1.0 - tip_height, min=0.0)
    return score


def load_wm(path='acrobot_wm.pt'):
    m = KAN([10, 24, 6], grid_size=5, spline_order=3)
    m.load_state_dict(torch.load(path, weights_only=True))
    return m.eval()


def evaluate_action(env, state, a, k, max_steps):
    """Execute action for k steps, return avg progress per step."""
    env.unwrapped.state = state
    score_before = goal_score_from_obs(env.unwrapped._get_ob())
    steps = 0
    for _ in range(min(k, max_steps)):
        obs, _, term, _, _ = env.step(a)
        steps += 1
        if term: break
    score_after = goal_score_from_obs(obs)
    progress = score_before - score_after  # positive = improved (lower score = better)
    env.unwrapped.state = state
    return progress / max(steps, 1)


def run_episode(wm, dn, env, explore_prob=0.08):
    """MPC action + DN k.  Returns (success, corrections, steps, k_used)."""
    obs, _ = env.reset()
    corrections = []
    k_used = []
    a_oh = torch.zeros(1, 3)

    for step in range(MAX_EPISODE):
        sn = torch.tensor(
            [[obs[0], obs[1], obs[2], obs[3], obs[4]/6.0, obs[5]/8.0]],
            dtype=torch.float32)

        # DN predicts k
        with torch.no_grad():
            k_dn = max(1, min(8, round(dn(sn)[0, 0].item() * 16)))

        # Explore: try different k
        if np.random.random() < explore_prob:
            # Pick best action via MPC (k=2 for accuracy)
            bs, ba = float('inf'), 0
            for a in [0, 1, 2]:
                a_oh.zero_(); a_oh[0, a] = 1.0
                with torch.no_grad():
                    pred = wm(torch.cat([sn, a_oh, torch.tensor([[2/8.0]])], dim=-1))
                s = goal_score_from_state(pred).item()
                if s < bs: bs, ba = s, a

            saved = env.unwrapped.state.copy()
            best_k, best_progress = k_dn, -float('inf')
            for kt in K_VALS:
                prog = evaluate_action(env, saved, ba, kt, MAX_EPISODE - step)
                if prog > best_progress:
                    best_progress, best_k = prog, kt

            if best_k != k_dn:
                corrections.append({'state': sn.squeeze(0).numpy().copy(),
                                    'k_cont': best_k / 16.0})
            k_use = best_k
        else:
            k_use = k_dn

        k_used.append(k_use)

        # MPC with chosen k
        wm.eval()
        for p in wm.parameters(): p.requires_grad = False
        kn = torch.tensor([[k_use / 8.0]])
        bs, ba = float('inf'), 0
        for a in [0, 1, 2]:
            a_oh.zero_(); a_oh[0, a] = 1.0
            with torch.no_grad():
                pred = wm(torch.cat([sn, a_oh, kn], dim=-1))
            s = goal_score_from_state(pred).item()
            if s < bs: bs, ba = s, a

        obs, _, term, _, _ = env.step(ba)
        if term: return True, corrections, step + 1, k_used

    return False, corrections, MAX_EPISODE, k_used


def fine_tune_dn(dn, corrections, epochs=200, lr=1e-3):
    if len(corrections) < 4: return 0.0
    s_data = torch.tensor(np.array([c['state'] for c in corrections]))
    k_data = torch.tensor([[c['k_cont']] for c in corrections])
    dn.train(); opt = torch.optim.Adam(dn.parameters(), lr=lr)
    mse = torch.nn.MSELoss()
    for _ in range(epochs):
        opt.zero_grad(); loss = mse(dn(s_data), k_data); loss.backward(); opt.step()
    dn.eval(); return loss.item()


def test(wm, dn, n_episodes=100):
    env = gym.make('Acrobot-v1'); a_oh = torch.zeros(1, 3); ok = 0; k_hist = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        for step in range(MAX_EPISODE):
            sn = torch.tensor([[obs[0],obs[1],obs[2],obs[3],obs[4]/6.0,obs[5]/8.0]],
                              dtype=torch.float32)
            with torch.no_grad(): k_dn = dn(sn)[0,0].item()
            k = max(1, min(8, round(k_dn * 16))); k_hist.append(k)
            kn = torch.tensor([[k/8.0]])
            bs, ba = float('inf'), 0
            for a in [0,1,2]:
                a_oh.zero_(); a_oh[0,a]=1.0
                with torch.no_grad():
                    pred = wm(torch.cat([sn, a_oh, kn], dim=-1))
                s = goal_score_from_state(pred).item()
                if s < bs: bs, ba = s, a
            obs, _, term, _, _ = env.step(ba)
            if term: ok += 1; break
    env.close()
    return ok, np.mean(k_hist)


def test_random(n_episodes=100):
    env = gym.make('Acrobot-v1'); ok = 0
    for _ in range(n_episodes):
        obs,_=env.reset()
        for step in range(MAX_EPISODE):
            obs,_,term,_,_=env.step(env.action_space.sample())
            if term: ok+=1; break
    env.close(); return ok


def main(n_episodes=30, lr=1e-3):
    torch.manual_seed(42); np.random.seed(42)
    wm = load_wm()
    dn = KAN([6, 12, 1], grid_size=5, spline_order=3)
    # Pre-train DN on world model labels
    print('Pre-training DN on world model labels...')
    states = torch.rand(2000, 6) * 2 - 1; a_oh = torch.zeros(1, 3)
    labels = []
    for i in range(2000):
        sn = states[i:i+1]; best_k, best_score = 1, float('inf')
        for k in K_VALS:
            kn = torch.tensor([[k/8.0]])
            for a in [0,1,2]:
                a_oh.zero_(); a_oh[0,a]=1.0
                with torch.no_grad():
                    pred = wm(torch.cat([sn, a_oh, kn], dim=-1))
                s = goal_score_from_state(pred).item()
                if s < best_score: best_score, best_k = s, k
        labels.append(best_k/16.0)
    k_lbl = torch.tensor([[l] for l in labels])
    # Train DN
    dn.train(); opt=torch.optim.Adam(dn.parameters(),lr=1e-2)
    mse=torch.nn.MSELoss()
    for _ in range(500):
        opt.zero_grad(); loss=mse(dn(states),k_lbl); loss.backward(); opt.step()
    dn.eval()
    print(f'  DN pre-trained on {len(labels)} labels')

    # Show k distribution in labels
    for kv in K_VALS:
        c = sum(1 for l in labels if round(l*16) == kv)
        print(f'    k={kv}: {c}/{len(labels)} ({c/len(labels)*100:.0f}%)')

    env = gym.make('Acrobot-v1'); all_corrections = []

    print(f'\nAcrobot CL: {n_episodes} episodes, lr={lr}')
    for ep in range(1, n_episodes+1):
        ok, corrections, steps, k_hist = run_episode(wm, dn, env)
        all_corrections.extend(corrections)
        loss = fine_tune_dn(dn, all_corrections, lr=lr)

        if ep % 10 == 0:
            test_ok, _ = test(wm, dn, n_episodes=50)
            print(f'  Ep {ep:2d}: {"OK" if ok else "FAIL"} steps={steps:3d} '
                  f'corr={len(corrections)} total_corr={len(all_corrections)} '
                  f'loss={loss:.4f} k_avg={np.mean(k_hist):.1f} test_50={test_ok}')

    # Final
    rnd = test_random(200)
    print(f'\nRandom baseline: {rnd}/200')
    ok_cl, avg_k = test(wm, dn, n_episodes=200)
    print(f'MPC+DN: {ok_cl}/200  avg_k={avg_k:.1f}')
    torch.save(dn.state_dict(), 'acrobot_dn.pt')
    print('Saved: acrobot_dn.pt')
    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()
    main(n_episodes=args.episodes, lr=args.lr)
