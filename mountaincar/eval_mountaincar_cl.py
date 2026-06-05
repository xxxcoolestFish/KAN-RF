"""MountainCar: MPC + decision network for k + environment feedback learning.

Goal: reach position >= 0.5 (flag).  Reward is sparse (-1 per step).
MPC: pick action that predicts maximum position after k steps.
"""
import torch, numpy as np, gymnasium as gym, time, argparse, sys, os
from kanrf import KAN

K_VALS = [1, 2, 4]
MAX_STEPS = 200
G = 0.0025


def energy_score(pred):
    """Higher = better. kinetic + potential energy."""
    vr = pred[:, 1] * 0.07  # denormalize velocity
    pr = pred[:, 0] * 0.6   # denormalize position
    return 0.5 * vr * vr + G * torch.clamp(pr, min=0.0)


def load_wm(path='mountaincar_wm.pt'):
    m = KAN([6, 16, 2], grid_size=5, spline_order=3)
    m.load_state_dict(torch.load(path, weights_only=True))
    return m.eval()


def evaluate_k(env, state_tuple, a, k):
    """Execute action for k steps, return final position."""
    pos, vel = state_tuple
    env.unwrapped.state = (pos, vel)
    for _ in range(k):
        obs, _, term, _, _ = env.step(a)
        if term:
            break
    env.unwrapped.state = (pos, vel)  # restore
    return obs[0]  # final position (higher = better)


def run_episode(wm, dn, env, explore_prob=0.08):
    obs, _ = env.reset()
    corrections = []
    k_used = []
    a_oh = torch.zeros(1, 3)

    for step in range(MAX_STEPS):
        p_n, v_n = obs[0] / 0.6, obs[1] / 0.07
        sn = torch.tensor([[p_n, v_n]], dtype=torch.float32)

        # DN predicts k
        with torch.no_grad():
            k_dn = max(1, min(4, round(dn(sn)[0, 0].item() * 16)))

        # Explore
        if np.random.random() < explore_prob:
            # Best action via MPC (k=1)
            bs, ba = -float('inf'), 0
            for a in [0, 1, 2]:
                a_oh.zero_(); a_oh[0, a] = 1.0
                with torch.no_grad():
                    pred = wm(torch.cat([sn, a_oh, torch.tensor([[1/4.0]])], dim=-1))
                e = energy_score(pred).item()
                if e > bs: bs, ba = e, a

            saved = (obs[0], obs[1])
            best_k, best_pos = k_dn, -float('inf')
            for kt in K_VALS:
                final_pos = evaluate_k(env, saved, ba, kt)
                if final_pos > best_pos:
                    best_pos, best_k = final_pos, kt

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
        kn = torch.tensor([[k_use / 4.0]])
        bs, ba = -float('inf'), 0
        for a in [0, 1, 2]:
            a_oh.zero_(); a_oh[0, a] = 1.0
            with torch.no_grad():
                pred = wm(torch.cat([sn, a_oh, kn], dim=-1))
            pos_pred = pred[0, 0].item() * 0.6
            if pos_pred > bs: bs, ba = pos_pred, a

        obs, _, term, _, _ = env.step(ba)
        if term: return True, corrections, step + 1, k_used

    return False, corrections, MAX_STEPS, k_used


def fine_tune_dn(dn, corrections, epochs=200, lr=1e-3):
    if len(corrections) < 4: return 0.0
    s_data = torch.tensor(np.array([c['state'] for c in corrections]))
    k_data = torch.tensor([[c['k_cont']] for c in corrections])
    dn.train(); opt = torch.optim.Adam(dn.parameters(), lr=lr)
    mse = torch.nn.MSELoss()
    for _ in range(epochs):
        opt.zero_grad(); loss = mse(dn(s_data), k_data); loss.backward(); opt.step()
    dn.eval(); return loss.item()


def test(wm, dn, n_episodes=200):
    env = gym.make('MountainCar-v0', max_episode_steps=200)
    a_oh = torch.zeros(1, 3); ok = 0; k_hist = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        for step in range(MAX_STEPS):
            p_n, v_n = obs[0] / 0.6, obs[1] / 0.07
            sn = torch.tensor([[p_n, v_n]], dtype=torch.float32)
            with torch.no_grad():
                k_dn = dn(sn)[0, 0].item()
            k = max(1, min(4, round(k_dn * 16))); k_hist.append(k)
            kn = torch.tensor([[k / 4.0]])
            bs, ba = -float('inf'), 0
            for a in [0, 1, 2]:
                a_oh.zero_(); a_oh[0, a] = 1.0
                with torch.no_grad():
                    pred = wm(torch.cat([sn, a_oh, kn], dim=-1))
                e = energy_score(pred).item()
                if e > bs: bs, ba = e, a
            obs, _, term, _, _ = env.step(ba)
            if term: ok += 1; break
    env.close()
    return ok, np.mean(k_hist) if k_hist else 0


def test_random(n_episodes=200):
    env = gym.make('MountainCar-v0', max_episode_steps=200); ok = 0
    for _ in range(n_episodes):
        obs, _ = env.reset()
        for _ in range(MAX_STEPS):
            obs, _, term, _, _ = env.step(env.action_space.sample())
            if term: ok += 1; break
    env.close(); return ok


def main(n_episodes=30, lr=1e-3):
    torch.manual_seed(42); np.random.seed(42)
    wm = load_wm()
    dn = KAN([2, 8, 1], grid_size=5, spline_order=3)

    # Pre-train DN on world model labels
    print('Pre-training DN...')
    states = torch.rand(2000, 2) * 2 - 1; a_oh = torch.zeros(1, 3)
    labels = []
    for i in range(2000):
        sn = states[i:i+1]; best_k, best_pos = 1, -float('inf')
        for k in K_VALS:
            kn = torch.tensor([[k/4.0]])
            for a in [0, 1, 2]:
                a_oh.zero_(); a_oh[0,a]=1.0
                with torch.no_grad():
                    pred = wm(torch.cat([sn, a_oh, kn], dim=-1))
                e = energy_score(pred).item()
                if e > best_pos: best_pos, best_k = e, k
        labels.append(best_k/16.0)
    k_lbl = torch.tensor([[l] for l in labels])
    dn.train(); opt=torch.optim.Adam(dn.parameters(),lr=1e-2)
    mse=torch.nn.MSELoss()
    for _ in range(500):
        opt.zero_grad(); loss=mse(dn(states),k_lbl); loss.backward(); opt.step()
    dn.eval()
    for kv in K_VALS:
        c = sum(1 for l in labels if round(l*16)==kv)
        print(f'  k={kv}: {c}/{len(labels)}')

    env = gym.make('MountainCar-v0', max_episode_steps=200)
    all_corrections = []

    print(f'\nMountainCar CL: {n_episodes} episodes')
    for ep in range(1, n_episodes+1):
        ok, corrections, steps, k_hist = run_episode(wm, dn, env)
        all_corrections.extend(corrections)
        loss = fine_tune_dn(dn, all_corrections, lr=lr)

        if ep % 10 == 0:
            test_ok, _ = test(wm, dn, n_episodes=100)
            print(f'  Ep {ep:2d}: {"OK" if ok else "FAIL"} steps={steps:3d} '
                  f'corr={len(corrections)} total_corr={len(all_corrections)} '
                  f'k_avg={np.mean(k_hist):.1f} test_100={test_ok}')

    rnd = test_random(200)
    print(f'\nRandom: {rnd}/200')
    ok_cl, avg_k = test(wm, dn, n_episodes=200)
    print(f'MPC+DN: {ok_cl}/200  avg_k={avg_k:.1f}')
    torch.save(dn.state_dict(), 'mountaincar_dn.pt')
    print('Saved: mountaincar_dn.pt')
    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()
    main(n_episodes=args.episodes, lr=args.lr)
