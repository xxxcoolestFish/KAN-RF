"""Evaluate TinyDecisionNet on Pendulum-v1.

Usage:
  python test_pendulum.py                                    # base (no k)
  python test_pendulum.py --dn tiny_dn_with_k.pt --with-k    # outputs (a,k)
"""
import torch, numpy as np, gymnasium as gym, time, sys, os
from kanrf import KAN
from decision_v2.core import FeatureComputer, TinyDecisionNet

PI_2 = np.pi / 2
MAX_K = 12


def run_episode(fc, dn, env, trial_seed, device, output_k):
    obs, _ = env.reset(seed=trial_seed)
    s_target = torch.tensor([[0., 1., 0.]], device=device)

    for _ in range(60):
        s_norm = torch.tensor([[obs[0], obs[1], obs[2] / 8.0]],
                              dtype=torch.float32).to(device)
        features = fc.compute_features(s_norm, s_target)
        with torch.no_grad():
            out = dn(features, s_norm)
            a_norm = out[0, 0].item()
            k = 1
            if output_k:
                k_cont = out[0, 1].item()
                k = max(1, min(MAX_K, round(k_cont * 16)))

        a_raw = np.clip(a_norm * 2.0, -2.0, 2.0)
        for _ in range(k):
            obs, _, term, trunc, _ = env.step([a_raw])
            if term or trunc:
                break

        if abs(np.arctan2(obs[1], obs[0]) - PI_2) < 0.2:
            return True
        if term or trunc:
            break
    return False


def main(dn_path='decision_v2/tiny_dn_base.pt', with_k=False, device='mps'):
    if device == 'mps' and not torch.backends.mps.is_available():
        device = 'cpu'
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    wm = KAN([4, 12, 3], grid_size=5, spline_order=3)
    wm.load_state_dict(torch.load(os.path.join(root, 'kan_pendulum_model_v4.pt'),
                                   weights_only=True))
    wm.eval().to(device)
    for p in wm.parameters():
        p.requires_grad = False

    fc = FeatureComputer(wm, device=device)
    dn = TinyDecisionNet(hidden=32, output_k=with_k).to(device)
    dn.load_state_dict(torch.load(os.path.join(root, dn_path), weights_only=True))
    dn.eval()

    env = gym.make('Pendulum-v1')
    t0 = time.time()
    ok = 0

    for t in range(10):
        trial_seed = 42 + t * 100
        success = run_episode(fc, dn, env, trial_seed, device, with_k)
        if success:
            ok += 1
        obs, _ = env.reset(seed=trial_seed)
        angle = np.arctan2(obs[1], obs[0])
        err = min(abs(angle - PI_2), 2 * np.pi - abs(angle - PI_2))
        print(f'  T{t+1}: {"OK" if success else "FAIL"}  initial |dθ|={np.rad2deg(err):.0f}deg')

    print(f'\n{ok}/10  |  {time.time()-t0:.0f}s')
    env.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dn', type=str, default='decision_v2/tiny_dn_base.pt')
    parser.add_argument('--with-k', action='store_true', default=False)
    parser.add_argument('--device', type=str, default='mps')
    args = parser.parse_args()
    main(dn_path=args.dn, with_k=args.with_k, device=args.device)
