"""Diagnose: does ProtoKAN WM correctly rank actions?"""
import torch, numpy as np, gymnasium as gym, sys
sys.path.insert(0, '.')
from kanrf import ProtoKAN

MAX_V1, MAX_V2 = 6.0, 8.0
S_TARGET = torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0, 0.0])

wm = ProtoKAN([9, 32, 6], n_prototypes=16)
wm.load_state_dict(torch.load('/tmp/acrobot_protokAN_wm.pt', weights_only=True))
wm.eval()

env = gym.make('Acrobot-v1')
env.reset()

# Test on 20 random states
correct_ranks = 0
total = 0

for test_idx in range(20):
    theta1 = np.random.uniform(-np.pi, np.pi)
    theta2 = np.random.uniform(-np.pi, np.pi)
    dtheta1 = np.random.uniform(-MAX_V1, MAX_V1)
    dtheta2 = np.random.uniform(-MAX_V2, MAX_V2)

    real_next = []
    for a in range(3):
        env.unwrapped.state = (theta1, theta2, dtheta1, dtheta2)
        obs, _, _, _, _ = env.step(a)
        sn = torch.tensor([obs[0], obs[1], obs[2], obs[3],
                          obs[4]/MAX_V1, obs[5]/MAX_V2], dtype=torch.float32)
        real_next.append(sn)

    s_n = torch.tensor([np.cos(theta1), np.sin(theta1), np.cos(theta2), np.sin(theta2),
                        dtheta1/MAX_V1, dtheta2/MAX_V2], dtype=torch.float32)
    wm_preds = []
    for a in range(3):
        a_oh = torch.zeros(1, 3); a_oh[0, a] = 1.0
        wm_in = torch.cat([s_n.unsqueeze(0), a_oh], dim=-1)
        with torch.no_grad():
            sp = wm(wm_in).squeeze(0)
        wm_preds.append(sp)

    w = torch.tensor([5., 5., 5., 5., 1., 1.])
    real_dists = [((r - S_TARGET) ** 2 * w).sum().item() for r in real_next]
    wm_dists = [((p - S_TARGET) ** 2 * w).sum().item() for p in wm_preds]

    real_best = int(np.argmin(real_dists))
    wm_best = int(np.argmin(wm_dists))

    if real_best == wm_best:
        correct_ranks += 1
    total += 1

    if test_idx < 5:
        print(f"State {test_idx}: real_best={real_best} wm_best={wm_best}")
        print(f"  Real dists: {[round(d, 4) for d in real_dists]}")
        print(f"  WM dists:   {[round(d, 4) for d in wm_dists]}")

env.close()
print(f"\nCorrect action ranking: {correct_ranks}/{total} ({correct_ranks/total*100:.0f}%)")
print(f"Random chance: 33%")

# Check: how different are actions really?
real_diffs_all = []
for _ in range(100):
    theta1 = np.random.uniform(-np.pi, np.pi)
    theta2 = np.random.uniform(-np.pi, np.pi)
    dtheta1 = np.random.uniform(-MAX_V1, MAX_V1)
    dtheta2 = np.random.uniform(-MAX_V2, MAX_V2)
    real_next = []
    for a in range(3):
        env.unwrapped.state = (theta1, theta2, dtheta1, dtheta2)
        obs, _, _, _, _ = env.step(a)
        sn = torch.tensor([obs[0], obs[1], obs[2], obs[3],
                          obs[4]/MAX_V1, obs[5]/MAX_V2], dtype=torch.float32)
        real_next.append(sn)
    real_diffs_all.append((real_next[0] - real_next[1]).abs().mean().item())
    real_diffs_all.append((real_next[2] - real_next[1]).abs().mean().item())

env.close()
print(f"\nMean |s(a) - s(a')| between different actions: {np.mean(real_diffs_all):.6f}")
print(f"WM val_mse: 0.000198")
print(f"Signal-to-noise ratio: {np.mean(real_diffs_all)/0.000198:.1f}x")
print(f"(If < 1: action differences smaller than WM error → impossible to rank)")
