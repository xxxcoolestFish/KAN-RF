#!/usr/bin/env python3
"""
Exp L: Lagrangian NN Control - Two-Phase Energy MPC
Uses learned Lagrangian model for energy-based swing-up.
"""
import sys, os, math, torch, numpy as np
import gymnasium as gym

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from lagrangian_v2 import LagNet


MAX_TORQUE = 2.0
MAX_SPEED = 8.0
DT = 0.05


def load_model(path, hidden=32):
    model = LagNet(hidden=hidden)
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model


def compute_energy(model, cos_th, sin_th, thd):
    """E = 0.5*I*θ̇² + U(θ)"""
    c = torch.tensor([cos_th], dtype=torch.float32, requires_grad=True)
    s = torch.tensor([sin_th], dtype=torch.float32, requires_grad=True)
    d = torch.tensor([thd], dtype=torch.float32)
    return model.energy(c, s, d).item()


def predict_next(model, cos_th, sin_th, thd, action):
    """Use Lagrangian model + semi-implicit Euler to predict next state."""
    c = torch.tensor([cos_th], dtype=torch.float32, requires_grad=True)
    s = torch.tensor([sin_th], dtype=torch.float32, requires_grad=True)
    d = torch.tensor([thd], dtype=torch.float32)
    a = torch.tensor([action], dtype=torch.float32)
    
    c.requires_grad_(True); s.requires_grad_(True)
    theta_ddot = model(c, s, d, a).item()
    
    # Semi-implicit Euler
    thd_next = thd + DT * theta_ddot
    thd_next = np.clip(thd_next, -MAX_SPEED, MAX_SPEED)
    
    theta = math.atan2(sin_th, cos_th)
    theta_next = theta + DT * thd_next
    
    cos_next = math.cos(theta_next)
    sin_next = math.sin(theta_next)
    
    return cos_next, sin_next, thd_next


def mpc_energy_pump(model, cos_th, sin_th, thd, target_energy, n_samples=30):
    """Phase 1: Maximize dE/dt by sampling actions."""
    E_now = compute_energy(model, cos_th, sin_th, thd)
    best_u = 0.0
    best_delta_E = -float('inf')
    
    for u in np.linspace(-MAX_TORQUE, MAX_TORQUE, n_samples):
        c_next, s_next, thd_next = predict_next(model, cos_th, sin_th, thd, u)
        E_next = compute_energy(model, c_next, s_next, thd_next)
        delta_E = E_next - E_now
        if delta_E > best_delta_E:
            best_delta_E = delta_E
            best_u = u
    
    return best_u


def mpc_pd(model, cos_th, sin_th, thd):
    """Phase 2: Minimize cost = θ² + 0.1θ̇² over short horizon."""
    best_u = 0.0
    best_cost = float('inf')
    
    for u in np.linspace(-MAX_TORQUE, MAX_TORQUE, 30):
        c, s, td = cos_th, sin_th, thd
        total_cost = 0.0
        
        for step in range(5):  # 5-step horizon
            c_next, s_next, td_next = predict_next(model, c, s, td, u)
            theta = math.atan2(s_next, c_next)
            # Normalize angle to [-pi, pi]
            theta_norm = ((theta + math.pi) % (2 * math.pi)) - math.pi
            total_cost += theta_norm**2 + 0.1 * (td_next / MAX_SPEED)**2
            c, s, td = c_next, s_next, td_next
        
        if total_cost < best_cost:
            best_cost = total_cost
            best_u = u
    
    return best_u


def run_episode(model, env, target_energy=5.0, energy_threshold=4.5, render=False):
    """Run one episode with two-phase energy MPC."""
    obs, _ = env.reset()
    total_reward = 0.0
    swing_up = False
    
    for t in range(200):
        cos_th, sin_th, thd_norm = obs[0], obs[1], obs[2]
        thd = thd_norm * MAX_SPEED
        
        E = compute_energy(model, cos_th, sin_th, thd)
        
        if not swing_up and E >= energy_threshold:
            swing_up = True
        
        if not swing_up:
            # Phase 1: energy pump
            action = mpc_energy_pump(model, cos_th, sin_th, thd, target_energy)
        else:
            # Phase 2: PD to hold upright
            action = mpc_pd(model, cos_th, sin_th, thd)
        
        obs, reward, term, trunc, _ = env.step(np.array([action], dtype=np.float32))
        total_reward += reward
        
        if render:
            env.render()
        
        if term or trunc:
            break
    
    # Check if success: pendulum stays near upright in last steps
    cos_final = obs[0]
    success = cos_final > 0.9  # within ~25° of upright
    
    return total_reward, success, swing_up


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='saved_models/lagrangian_v3_clean.pt')
    parser.add_argument('--hidden', type=int, default=32)
    parser.add_argument('--episodes', type=int, default=20)
    parser.add_argument('--energy_target', type=float, default=5.0)
    parser.add_argument('--energy_threshold', type=float, default=4.5)
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()
    
    print(f"Loading model: {args.model}")
    model = load_model(args.model, hidden=args.hidden)
    print(f"  I = {model.I.item():.4f} (target: 0.3333)")
    
    env = gym.make('Pendulum-v1')
    
    successes = 0
    returns = []
    
    for ep in range(args.episodes):
        ret, success, sw_up = run_episode(
            model, env, args.energy_target, args.energy_threshold,
            render=args.render and ep == 0
        )
        returns.append(ret)
        if success:
            successes += 1
        status = "✓ SWING-UP" if success else ("✗ (pumped)" if sw_up else "✗ (no pump)")
        print(f"Ep {ep:2d} | Return={ret:8.2f} | {status}")
    
    print(f"\n=== Results ===")
    print(f"Success: {successes}/{args.episodes} ({100*successes/args.episodes:.0f}%)")
    print(f"Mean return: {np.mean(returns):.2f} ± {np.std(returns):.2f}")
    
    env.close()


if __name__ == '__main__':
    main()
