#!/usr/bin/env python3
"""Minimal test: Can a network learn U(cos,sin)→-5cosθ through Lagrangian loss?"""
import torch
import torch.nn as nn

# Simple test network
net = nn.Sequential(
    nn.Linear(2, 32), nn.Tanh(),
    nn.Linear(32, 32), nn.Tanh(),
    nn.Linear(32, 1)
)

I_param = nn.Parameter(torch.tensor(0.0))  # log_I, init I=1

optimizer = torch.optim.Adam(list(net.parameters()) + [I_param], lr=0.01)

# Generate test data: θ̈ = 3u + 15sinθ
N = 10000
theta = torch.rand(N) * 2 * torch.pi - torch.pi
u = torch.rand(N) * 4 - 2
cos_t = torch.cos(theta)
sin_t = torch.sin(theta)
thd_t = torch.randn(N) * 4  # random velocities
thd_n = thd_t + 0.05 * (3*u + 15*torch.sin(theta))  # dt=0.05

theta_ddot_true = (thd_n - thd_t) / 0.05

print(f"True θ̈ range: [{theta_ddot_true.min():.2f}, {theta_ddot_true.max():.2f}]")
print(f"I init: {torch.exp(I_param).item():.4f}")

for epoch in range(500):
    # batch
    idx = torch.randint(0, N, (256,))
    c = cos_t[idx].requires_grad_(True)
    s = sin_t[idx].requires_grad_(True)
    thd = thd_t[idx]
    act = u[idx]
    target = theta_ddot_true[idx]
    
    # Forward: compute dU/dθ via autograd
    x = torch.stack([c, s], dim=1)
    U = net(x).squeeze()
    dU_dc, dU_ds = torch.autograd.grad(U.sum(), [c, s], create_graph=True, retain_graph=True)
    dU_dtheta = -dU_dc * s + dU_ds * c
    
    I = torch.exp(I_param).clamp(min=0.01)
    pred = (act - dU_dtheta) / I
    
    loss = torch.mean((pred - target) ** 2)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 50 == 0 or epoch < 5:
        # Check U values
        with torch.no_grad():
            test_c = torch.tensor([1.0, 0.0, -1.0]).unsqueeze(1)
            test_s = torch.tensor([0.0, 1.0, 0.0]).unsqueeze(1)
            test_x = torch.cat([test_c, test_s], dim=1)
            U_vals = net(test_x).squeeze()
            print(f"Epoch {epoch:3d} | Loss={loss.item():.4f} | I={I.item():.4f} | "
                  f"U(θ=0)={U_vals[0]:.3f} U(θ=π/2)={U_vals[1]:.3f} U(θ=π)={U_vals[2]:.3f}")

print(f"\nFinal I: {I.item():.4f} (target: 0.333)")
print(f"Target U: -5cos(θ) → U(0)=-5, U(π/2)=0, U(π)=5")
