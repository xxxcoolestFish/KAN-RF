"""Render CartPole MPC episodes as GIFs."""
import torch, numpy as np, gymnasium as gym, sys, imageio
from kanrf import KAN


def render_episode(wm, seed, path, max_steps=500):
    env = gym.make('CartPole-v1', render_mode='rgb_array')
    a_oh = torch.zeros(1, 2)
    kn = torch.tensor([[4 / 16.0]])
    obs, _ = env.reset(seed=seed)
    frames = [env.render()]

    for step in range(max_steps):
        sn = torch.tensor(
            [[obs[0]/2.5, obs[1]/3.0, obs[2]/0.3, obs[3]/3.0]],
            dtype=torch.float32)
        bs, ba = float('inf'), 0
        for a in [0, 1]:
            a_oh.zero_()
            a_oh[0, a] = 1.0
            with torch.no_grad():
                pred = wm(torch.cat([sn, a_oh, kn], dim=-1))
            score = abs(pred[0, 2]) * 0.5 + abs(pred[0, 0]) * 0.2
            if score < bs:
                bs, ba = score, a
        obs, _, term, trunc, _ = env.step(ba)
        if step % 4 == 0:  # every 4th frame to keep GIF size reasonable
            frames.append(env.render())
        if term or trunc:
            break

    env.close()
    # Downsample to ~120 frames max
    if len(frames) > 150:
        stride = len(frames) // 120
        frames = frames[::stride]
    imageio.mimsave(path, frames, fps=20, loop=0)
    print(f'  {path}: {len(frames)} frames, result={"OK" if step>=475 else "FAIL"} ({step} steps)')


def main():
    wm = KAN([7, 20, 4], grid_size=5, spline_order=3)
    wm.load_state_dict(torch.load('/Users/zhuangxinyu/KAN/KAN-RF/kan_cartpole.pt',
                                   weights_only=True))
    wm.eval()
    for p in wm.parameters():
        p.requires_grad = False

    # Success: seed=1, cart drifts right slowly but stays within bounds
    render_episode(wm, 1, 'cartpole_success.gif')

    # Failure: seed=0, cart drifts left and hits boundary at step 406
    render_episode(wm, 0, 'cartpole_failure.gif')


if __name__ == '__main__':
    main()
