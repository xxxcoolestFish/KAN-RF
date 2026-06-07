#!/usr/bin/env python3
"""
Generate swing-up trajectory data for KAN training.

Strategy: energy-pumping controller
  E_des = mgl  (energy needed to reach upright)
  E_cur = 0.5*m*l^2*thd^2 + mgl*(1 - cos(th))
  a = k * (E_des - E_cur) * sign(thd * cos(th))
  
This is a well-known analytic controller that provably swings up
a pendulum. We use it to generate ground-truth swing-up data
for training the KAN world model.
"""
import math, sys, os
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# Pendulum-v1 physical parameters
M = 1.0   # mass
L = 1.0   # length  
G = 10.0  # gravity
MAX_TORQUE = 2.0
DT = 0.05

def energy(theta, theta_dot):
    """Total energy relative to bottom (theta=0=upright in gym convention?)"""
    # Pendulum-v1: theta=0 is pointing DOWN (hanging)
    # For energy to upright: we want kinetic + potential
    # h = L*(1-cos(theta)) where theta measured from downward
    return 0.5 * M * L**2 * theta_dot**2 + M * G * L * (1 - math.cos(theta))


def energy_pump_controller(th, thd):
    """Energy-pumping controller for Pendulum-v1.
    
    Pumps energy to reach upright (theta=0 means hanging down, 
    theta=pi means upright in gym convention? No...)
    
    Actually Pendulum-v1: theta=0 is hanging DOWN. Upright is theta=±pi.
    cos(theta) = 1 at bottom, -1 at upright.
    """
    # desired energy to reach upright: mgl * 2 (bottom to top)
    E_des = M * G * L * 2.0
    E_cur = energy(th, thd)
    E_err = E_des - E_cur
    
    # Energy-shaping control law
    # a = saturate(k * E_err * sign(thd * cos(th)))
    gain = 3.0
    u = gain * E_err * thd * math.cos(th)
    u = max(-MAX_TORQUE, min(MAX_TORQUE, u))
    return u


def generate_trajectories(n_episodes=100, max_steps=500):
    """Generate swing-up trajectories using energy-pumping controller."""
    import gymnasium as gym
    env = gym.make('Pendulum-v1')
    
    all_data = []
    success_count = 0
    
    for ep in range(n_episodes):
        s, _ = env.reset()
        th, thd = float(math.atan2(s[1], s[0])), float(s[2])
        
        traj = []
        for step in range(max_steps):
            a = energy_pump_controller(th, thd)
            s_next, r, term, trunc, _ = env.step(np.array([a]))
            th_next = float(math.atan2(s_next[1], s_next[0]))
            thd_next = float(s_next[2])
            
            # Record: (s_norm, a_norm, s_next_norm)
            traj.append((
                [float(s[0]), float(s[1]), s[2]/8.0],
                a / 2.0,
                [float(s_next[0]), float(s_next[1]), s_next[2]/8.0],
            ))
            
            th, thd = th_next, thd_next
            s = s_next
            
            if term or trunc:
                break
        
        # Check if episode reached upright
        final_err = abs(math.atan2(math.sin(th), math.cos(th)))
        if final_err < 0.1:
            success_count += 1
        
        all_data.extend(traj)
    
    env.close()
    
    print(f"Generated {len(all_data)} transitions from {n_episodes} episodes")
    print(f"Swing-up success: {success_count}/{n_episodes}")
    
    return all_data


def merge_data(original_path, swingup_data, output_path, swingup_weight=3):
    """Merge original data with swing-up data, oversampling swing-up."""
    orig = torch.load(original_path, map_location='cpu', weights_only=False)
    X_orig, Y_orig = orig[0], orig[1]  # both tensors
    
    X_sw = torch.tensor([d[0] + [d[1]] for d in swingup_data], dtype=torch.float32)
    Y_sw = torch.tensor([d[2] for d in swingup_data], dtype=torch.float32)
    
    print(f"Original: {len(X_orig)} transitions")
    print(f"Swing-up: {len(X_sw)} transitions (weighted x{swingup_weight})")
    
    # Repeat swing-up data to give it more weight
    X_sw_repeated = X_sw.repeat(swingup_weight, 1)
    Y_sw_repeated = Y_sw.repeat(swingup_weight, 1)
    
    X_all = torch.cat([X_orig, X_sw_repeated], dim=0)
    Y_all = torch.cat([Y_orig, Y_sw_repeated], dim=0)
    
    print(f"Merged: {len(X_all)} transitions")
    
    torch.save((X_all, Y_all), output_path)
    print(f"Saved: {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=50)
    parser.add_argument('--max-steps', type=int, default=500)
    parser.add_argument('--swingup-weight', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    
    print("=" * 55)
    print("Generating Swing-Up Data via Energy-Pumping Controller")
    print("=" * 55)
    
    sw_data = generate_trajectories(n_episodes=args.episodes, max_steps=args.max_steps)
    
    orig_path = os.path.join(PROJECT_ROOT, 'saved_data', 'pendulum_data_v4.pt')
    out_path = os.path.join(PROJECT_ROOT, 'saved_data', 'pendulum_data_v5_swingup.pt')
    
    merge_data(orig_path, sw_data, out_path, swingup_weight=args.swingup_weight)


if __name__ == '__main__':
    main()
