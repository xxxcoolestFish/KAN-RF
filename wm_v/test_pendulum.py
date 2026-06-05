"""Test World Model + Value Network on Pendulum-v1.

Continuous action via inverse optimization through world model,
scored by V(s').  Reward: cos(angle_error), bounded in [-1, 1].

Usage:
  python test_pendulum.py                          # cold-start V
  python test_pendulum.py --pretrain v_pretrained.pt  # pre-trained V
"""
import torch, numpy as np, gymnasium as gym, sys, os, time
from kanrf import KAN
from wm_v_core import MLPValue, ReplayBuffer

PI_2 = np.pi / 2


def inverse_optimize(s_norm, wm, V, target, gamma=0.97, n_iters=30):
    """min_a -[r(s,a,s') + gamma*V(s')] where s' = f(s,a)."""
    a = torch.zeros(1, 1, device=s_norm.device)
    torch.nn.init.uniform_(a, -0.3, 0.3)
    a.requires_grad_(True)
    opt = torch.optim.Adam([a], lr=0.05)

    for _ in range(n_iters):
        opt.zero_grad()
        x = torch.cat([s_norm, a], dim=-1)
        sp = wm(x)[:, :3]
        r = -((sp - target) ** 2).sum()
        with torch.no_grad():
            v = V(sp)
        loss = -(r + gamma * v)
        loss.backward()
        opt.step()
        with torch.no_grad():
            a.clamp_(-1.0, 1.0)

    return a.detach().item()


def render_td_state(V, device):
    """Print V(s) for key states to monitor training progress."""
    test = torch.tensor([
        [-1., 0., 0.],       # bottom, v=0
        [0., 1., 0.],        # upright
        [-0.5, 0.866, 0.],   # mid-up
        [-1., 0., 0.5],      # bottom, v>0
        [0.5, 0.866, 0.],    # near upright
    ], device=device)
    with torch.no_grad():
        vv = V(test).squeeze(1).tolist()
    return (f'V_bot0={vv[0]:.2f} V_up={vv[1]:.2f} '
            f'V_mid={vv[2]:.2f} V_bot+={vv[3]:.2f} V_near={vv[4]:.2f}')


def main(n_episodes=30, gamma=0.97, lr=3e-4, pretrain_path=None, device='mps'):
    if device == 'mps' and not torch.backends.mps.is_available():
        device = 'cpu'
    torch.manual_seed(42)
    np.random.seed(42)
    t0 = time.time()

    # Load frozen world model
    wm = KAN([4, 12, 3], grid_size=5, spline_order=3)
    wm_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                           'kan_pendulum_model_v4.pt')
    wm.load_state_dict(torch.load(wm_path, weights_only=True))
    wm.eval().to(device)
    for p in wm.parameters():
        p.requires_grad = False

    # Initialize or load V(s)
    V = MLPValue(3).to(device)
    if pretrain_path:
        pretrain_full = os.path.join(os.path.dirname(__file__), pretrain_path)
        V.load_state_dict(torch.load(pretrain_full, weights_only=True))
        print(f'Loaded pretrained V: {pretrain_path}')
    else:
        print('Cold-start V(s)')

    opt = torch.optim.Adam(V.parameters(), lr=lr)
    buf = ReplayBuffer(500)
    target = torch.tensor([[0., 1., 0.]], device=device)

    env = gym.make('Pendulum-v1')
    ok_count = 0
    print(f'WM+V Pendulum (gamma={gamma}, lr={lr}, device={device})')

    for ep in range(1, n_episodes + 1):
        obs, _ = env.reset()
        ep_reward = 0
        explore_eps = max(0.02, 0.4 * (0.93 ** ep))

        for step in range(200):
            s_norm = torch.tensor([[obs[0], obs[1], obs[2] / 8.0]],
                                  dtype=torch.float32).to(device)

            a_norm = inverse_optimize(s_norm, wm, V, target,
                                      gamma=gamma, n_iters=30)
            if np.random.random() < explore_eps:
                a_norm = np.random.uniform(-1, 1)

            a_raw = a_norm * 2.0
            obs, _, term, _, _ = env.step([a_raw])

            s_next_norm = torch.tensor([[obs[0], obs[1], obs[2] / 8.0]],
                                       dtype=torch.float32).to(device)
            r_real = ((s_next_norm[:, :2] * target[:, :2]).sum(-1)
                      .clamp(-1, 1).item())
            ep_reward += r_real

            buf.push(s_norm, s_next_norm, r_real)
            if len(buf.buf) >= 16:
                sb, snb, rb = buf.sample(32)
                sb, snb = sb.to(device), snb.to(device)
                rb = rb.to(device).unsqueeze(1)
                with torch.no_grad():
                    tgt = rb + gamma * V(snb)
                V.train()
                opt.zero_grad()
                loss = torch.nn.functional.mse_loss(V(sb), tgt)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(V.parameters(), 1.0)
                opt.step()
                V.eval()

            angle = np.arctan2(obs[1], obs[0])
            if abs(angle - PI_2) < 0.2:
                ok_count += 1
                break

        if ep % 5 == 0 or ep == 1:
            ok_str = 'OK' if abs(np.arctan2(obs[1], obs[0]) - PI_2) < 0.2 else 'FAIL'
            print(f'  Ep {ep:2d}: {ok_str}  reward={ep_reward:.1f}  '
                  f'eps={explore_eps:.3f}  {render_td_state(V, device)}')

    print(f'\n{ok_count}/{n_episodes} OK  |  {time.time()-t0:.0f}s')
    env.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=30)
    parser.add_argument('--gamma', type=float, default=0.97)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--pretrain', type=str, default=None)
    parser.add_argument('--device', type=str, default='mps')
    args = parser.parse_args()
    main(n_episodes=args.episodes, gamma=args.gamma, lr=args.lr,
         pretrain_path=args.pretrain, device=args.device)
