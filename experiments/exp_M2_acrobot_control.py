#!/usr/bin/env python3
"""Acrobot control using learned Lagrangian dynamics.

MPC: for each of 3 discrete torques {-1, 0, +1}, predict H-step rollout
via Lagrangian model + semi-implicit Euler. Choose action maximizing
tip height. Replan every step.

Success criterion (gymnasium Acrobot-v1):
  -cos(θ₁) - cos(θ₁ + θ₂) > 1.0  (tip of link 2 above goal line)
"""
import sys, os, math, torch, numpy as np, argparse
import gymnasium as gym

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from lagrangian._acrobot import AcrobotLagNet

DT = 0.2
MAX_STEPS = 500
TORQUES = [-1.0, 0.0, 1.0]  # discrete actions 0, 1, 2


def load_model(path, hidden=32):
    model = AcrobotLagNet(hidden=hidden)
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model


def predict_next(model, cos1, sin1, cos2, sin2, thd1, thd2, torque):
    """Single step via Lagrangian + semi-implicit Euler."""
    c1 = torch.tensor([cos1], dtype=torch.float32, requires_grad=True)
    s1 = torch.tensor([sin1], dtype=torch.float32, requires_grad=True)
    c2 = torch.tensor([cos2], dtype=torch.float32, requires_grad=True)
    s2 = torch.tensor([sin2], dtype=torch.float32, requires_grad=True)
    d1 = torch.tensor([thd1], dtype=torch.float32)
    d2 = torch.tensor([thd2], dtype=torch.float32)
    tq = torch.tensor([torque], dtype=torch.float32)

    thdd1, thdd2 = model(c1, s1, c2, s2, d1, d2, tq)

    # Semi-implicit Euler
    thd1_n = thd1 + DT * thdd1.item()
    thd2_n = thd2 + DT * thdd2.item()
    th1 = math.atan2(sin1, cos1)
    th2 = math.atan2(sin2, cos2)
    th1_n = th1 + DT * thd1_n
    th2_n = th2 + DT * thd2_n

    return (math.cos(th1_n), math.sin(th1_n),
            math.cos(th2_n), math.sin(th2_n), thd1_n, thd2_n)


def tip_height(cos1, sin1, cos2, sin2):
    """Height of link-2 tip above the pivot: -cos(θ₁) - cos(θ₁+θ₂)."""
    cos12 = cos1 * cos2 - sin1 * sin2  # cos(θ₁+θ₂)
    return -cos1 - cos12


def rollout_score(model, cos1, sin1, cos2, sin2, thd1, thd2, torque, H=8):
    """H-step rollout: return max tip height achieved (higher is better)."""
    c1, s1, c2, s2 = cos1, sin1, cos2, sin2
    d1, d2 = thd1, thd2
    best_height = tip_height(c1, s1, c2, s2)

    for _ in range(H):
        c1_n, s1_n, c2_n, s2_n, d1, d2 = predict_next(
            model, c1, s1, c2, s2, d1, d2, torque)
        h = tip_height(c1_n, s1_n, c2_n, s2_n)
        if h > best_height:
            best_height = h
        c1, s1, c2, s2 = c1_n, s1_n, c2_n, s2_n

    return best_height


def mpc_action(model, obs, H=8):
    """Choose torque maximizing rollout tip height."""
    cos1, sin1 = obs[0], obs[1]
    cos2, sin2 = obs[2], obs[3]
    thd1 = obs[4]
    thd2 = obs[5]

    best_torque = 0
    best_height = -float('inf')

    for torque in TORQUES:
        h = rollout_score(model, cos1, sin1, cos2, sin2, thd1, thd2, torque, H)
        if h > best_height:
            best_height = h
            best_torque = torque

    # Convert torque back to discrete action
    action = TORQUES.index(best_torque)
    return action, best_height


def run_episode(model, env, H=8, seed=None):
    """Run one episode. Returns (success, steps, max_height_achieved)."""
    if seed is not None:
        obs, _ = env.reset(seed=seed)
    else:
        obs, _ = env.reset()

    max_h = tip_height(obs[0], obs[1], obs[2], obs[3])

    for step in range(MAX_STEPS):
        action, _ = mpc_action(model, obs, H=H)
        obs, _, term, trunc, _ = env.step(action)

        h = tip_height(obs[0], obs[1], obs[2], obs[3])
        if h > max_h:
            max_h = h

        if term:
            return True, step + 1, max_h  # term=True means success!
        if trunc:
            return False, step + 1, max_h  # ran out of steps without reaching goal

    return False, MAX_STEPS, max_h


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='saved_models/acrobot_lagrangian.pt')
    parser.add_argument('--hidden', type=int, default=32)
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--horizon', type=int, default=8, help='MPC rollout horizon')
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model = load_model(args.model, hidden=args.hidden)
    diag = model.true_params()
    print(f"  a₁={diag['a1']:.3f} b₁={diag['b1']:.3f} "
          f"a₂={diag['a2']:.3f} b₂={diag['b2']:.3f}")

    env = gym.make('Acrobot-v1')

    successes, steps_list = [], []
    for ep in range(args.episodes):
        seed = 42 + ep * 100
        ok, steps, max_h = run_episode(model, env, H=args.horizon, seed=seed)
        successes.append(ok)
        steps_list.append(steps)

        if (ep + 1) % 20 == 0 or ep < 5:
            status = "✓" if ok else f"✗ ({steps} steps, max_h={max_h:.2f})"
            print(f"Ep {ep:3d}: {status}")

    print(f"\n=== Results ===")
    print(f"Success: {sum(successes)}/{args.episodes} "
          f"({100*sum(successes)/args.episodes:.0f}%)")
    print(f"Mean steps: {np.mean(steps_list):.0f} ± {np.std(steps_list):.0f}")
    print(f"Target: ≥ 90% (KAN-based approach: 92%)")

    env.close()


if __name__ == '__main__':
    main()
