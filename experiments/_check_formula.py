#!/usr/bin/env python3
"""Check if real data exactly follows θ̈ = 3u + 15sinθ."""
import sys, os, torch, math

data = torch.load('/Users/zhuangxinyu/KAN/KAN-RF_q/saved_data/pendulum_data_v5_swingup.pt')
sa = data[0][:5000]
ns = data[1][:5000]

cos_t, sin_t = sa[:, 0], sa[:, 1]
thd_norm, u_norm = sa[:, 2], sa[:, 3]
thd_next_norm = ns[:, 2]

thd_t = thd_norm * 8.0
thd_n = thd_next_norm * 8.0
u_t = u_norm * 2.0

# True acceleration from data (finite difference)
dt = 0.05
accel_true = (thd_n - thd_t) / dt

# Formula acceleration: θ̈ = 3u + 15sinθ
theta = torch.atan2(sin_t, cos_t)
accel_formula = 3.0 * u_t + 15.0 * torch.sin(theta)

# Error
diff = accel_true - accel_formula
mse = torch.mean(diff ** 2).item()
print(f"MSE(θ̈_data, θ̈_formula) = {mse:.6f}")
print(f"RMSE = {math.sqrt(mse):.4f}")
print(f"Correlation: {torch.corrcoef(torch.stack([accel_true, accel_formula]))[0,1].item():.4f}")

# Check if speed clipping is an issue
clipped = thd_n.abs() >= 7.99
print(f"\nSpeed-clipped transitions: {clipped.sum().item()}/{len(thd_t)} ({100*clipped.sum().item()/len(thd_t):.1f}%)")

if clipped.sum() > 0:
    mse_clip = torch.mean((accel_true[clipped] - accel_formula[clipped])**2).item()
    mse_no_clip = torch.mean((accel_true[~clipped] - accel_formula[~clipped])**2).item()
    print(f"MSE clipped: {mse_clip:.4f}, MSE unclipped: {mse_no_clip:.4f}")

# Sample some predictions
print("\n=== Sample comparisons ===")
for i in [0, 100, 500, 1000, 2000]:
    print(f"  θ={theta[i].item():5.2f} θ̇={thd_t[i].item():6.2f} u={u_t[i].item():5.2f} | "
          f"θ̈_data={accel_true[i].item():7.3f} θ̈_3u+15sinθ={accel_formula[i].item():7.3f} err={abs(accel_true[i]-accel_formula[i]):.3f}")

# Also check formula with semi-implicit Euler
thd_si = thd_t + dt * (3*u_t + 15*torch.sin(theta))
theta_si = theta + dt * thd_si
cos_si = torch.cos(theta_si)
sin_si = torch.sin(theta_si)
thd_norm_si = thd_si / 8.0

state_mse = torch.mean((cos_si - ns[:,0])**2 + (sin_si - ns[:,1])**2 + (thd_norm_si - ns[:,2])**2).item()
print(f"\nState MSE with formula: {state_mse:.6f}")
