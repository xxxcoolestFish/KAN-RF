#!/usr/bin/env python3
"""CartPole control using learned Lagrangian dynamics.

Pure MPC: at each step, try both discrete actions, predict H-step
rollout via Lagrangian + semi-implicit Euler, choose action with
minimum balancing cost.

Success: survive 500 steps without pole falling (|θ| > 12°) or
cart going out of bounds (|x| > 2.4).
"""
import sys, os, math, torch, numpy as np, argparse
import gymnasium as gym

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from lagrangian._cartpole import CartPoleLagNet

DT = 0.02
MAX_STEPS = 500
ACTION_FORCE = [-10.0, 10.0]  # left, right


def load_model(path, hidden=32):
    model = CartPoleLagNet(hidden=hidden)
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model


def predict_next(model, cos_th, sin_th, x_dot, th_dot, force):
    """Single step: Lagrangian model + semi-implicit Euler → next state."""
    c = torch.tensor([cos_th], dtype=torch.float32, requires_grad=True)
    s = torch.tensor([sin_th], dtype=torch.float32, requires_grad=True)
    xd = torch.tensor([x_dot], dtype=torch.float32)
    td = torch.tensor([th_dot], dtype=torch.float32)
    f = torch.tensor([force], dtype=torch.float32)

    x_ddot, th_ddot = model(c, s, xd, td, f)
    x_ddot_i = x_ddot.item()
    th_ddot_i = th_ddot.item()

    # Semi-implicit Euler
    th_dot_next = th_dot + DT * th_ddot_i
    th_next = math.atan2(sin_th, cos_th) + DT * th_dot_next
    x_dot_next = x_dot + DT * x_ddot_i
    x_next = x_dot * DT  # not needed (x is cyclic), but track for bounds

    return math.cos(th_next), math.sin(th_next), x_dot_next, th_dot_next


def rollout_cost(model, cos_th, sin_th, x, x_dot, th, th_dot, force, H=10):
    """H-step rollout cost: Σ(θ² + 0.1θ̇² + 0.01x² + 0.01ẋ²)."""
    c, s, xd, td = cos_th, sin_th, x_dot, th_dot
    total = 0.0

    for _ in range(H):
        c_next, s_next, xd_next, td_next = predict_next(model, c, s, xd, td, force)

        # Compute theta (normalized to [-π, π])
        th = math.atan2(s_next, c_next)
        th_norm = ((th + math.pi) % (2 * math.pi)) - math.pi

        total += th_norm**2 + 0.1 * (td_next)**2

        c, s, xd, td = c_next, s_next, xd_next, td_next

    return total


def mpc_action(model, cos_th, sin_th, x, x_dot, th_dot, H=10):
    """Choose action minimizing rollout cost."""
    best_action = 0
    best_cost = float('inf')

    th = math.atan2(sin_th, cos_th)

    for a_idx, force in enumerate(ACTION_FORCE):
        cost = rollout_cost(model, cos_th, sin_th, x, x_dot, th, th_dot, force, H)
        if cost < best_cost:
            best_cost = cost
            best_action = a_idx

    return best_action, best_cost


def run_episode(model, env, H=10, seed=None):
    """Run one episode, return (survival_steps, success)."""
    if seed is not None:
        obs, _ = env.reset(seed=seed)
    else:
        obs, _ = env.reset()

    for step in range(MAX_STEPS):
        x, x_dot, theta, theta_dot = obs[0], obs[1], obs[2], obs[3]
        cos_th, sin_th = math.cos(theta), math.sin(theta)

        action, cost = mpc_action(model, cos_th, sin_th, x, x_dot, theta_dot, H=H)

        obs, _, term, trunc, _ = env.step(action)

        if term:
            # term=True means pole fell or cart went out of bounds → failure
            return step + 1, False
        if trunc:
            # trunc=True means survived all MAX_STEPS → success
            return step + 1, True

    return MAX_STEPS, True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='saved_models/cartpole_lagrangian.pt')
    parser.add_argument('--hidden', type=int, default=32)
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--horizon', type=int, default=10, help='MPC rollout horizon')
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model = load_model(args.model, hidden=args.hidden)
    print(f"  mp={model.mp.item():.4f}  mc={model.mc.item():.4f}  "
          f"L={model.length.item():.4f}  I={model.I_theta.item():.4f}")

    env = gym.make('CartPole-v1')

    successes, survivals = 0, []
    for ep in range(args.episodes):
        seed = 42 + ep * 100
        steps, ok = run_episode(model, env, H=args.horizon, seed=seed)
        survivals.append(steps)
        if ok:
            successes += 1
        status = "✓" if ok else f"✗ ({steps} steps)"
        if (ep + 1) % 20 == 0 or ep < 5:
            print(f"Ep {ep:3d}: {status}")

    print(f"\n=== Results ===")
    print(f"Success: {successes}/{args.episodes} ({100*successes/args.episodes:.0f}%)")
    print(f"Mean survival: {np.mean(survivals):.0f} ± {np.std(survivals):.0f} steps")
    print(f"Max survival: {np.max(survivals)} steps")

    env.close()


if __name__ == '__main__':
    main()
