#!/usr/bin/env python3
"""Quick diagnostic for Lagrangian model."""
import sys, os, torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from lagrangian_model import LagrangianPendulum

device = torch.device('cpu')

# Load model
ckpt = torch.load('saved_models/lagrangian_pendulum.pt', map_location=device)
model = LagrangianPendulum(hidden_dim=64, device=device)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

# Load data
data = torch.load('saved_data/pendulum_data_v5_swingup.pt', map_location=device)
states = data[0][:1000]  # (cos, sin, thd_norm, u_norm)
next_s = data[1][:1000]

cos_t, sin_t = states[:, 0], states[:, 1]
thd_t_norm = states[:, 2]  # normalized [-1,1]
u_t_norm = states[:, 3]    # normalized [-1,1]

# Denormalize
thd_t = thd_t_norm * 8.0
u_t = u_t_norm * 2.0
thd_n = next_s[:, 2] * 8.0

# True theta_ddot
dt = 0.05
theta_ddot_true = (thd_n - thd_t) / dt

# Enable grad for angle inputs
cos_t.requires_grad_(True)
sin_t.requires_grad_(True)

# Model prediction
theta_ddot_pred = model(cos_t, sin_t, thd_t, u_t)

# Loss
loss_acc = torch.mean((theta_ddot_pred - theta_ddot_true) ** 2)
print(f"Acceleration MSE: {loss_acc.item():.6f}")
print(f"  RMSE: {torch.sqrt(loss_acc).item():.4f}")
print(f"  True θ̈ range: [{theta_ddot_true.min().item():.2f}, {theta_ddot_true.max().item():.2f}]")
print(f"  Pred θ̈ range: [{theta_ddot_pred.min().item():.2f}, {theta_ddot_pred.max().item():.2f}]")

# State prediction
cos_p, sin_p, thd_p = model.predict_next_state(cos_t, sin_t, thd_t, u_t)
thd_n_norm = next_s[:, 2]
loss_state = torch.mean((cos_p - next_s[:, 0])**2 + (sin_p - next_s[:, 1])**2 + (thd_p - thd_n_norm)**2)
print(f"\nState MSE: {loss_state.item():.6f}")

# Check a few examples
print("\n=== Sample predictions ===")
for i in [0, 10, 50, 100]:
    t = theta_ddot_true[i].item()
    p = theta_ddot_pred[i].item()
    cosv = cos_t[i].item()
    sinv = sin_t[i].item()
    thdv = thd_t[i].item()
    uv = u_t[i].item()
    print(f"  θ={torch.atan2(sin_t[i], cos_t[i]).item():5.2f} θ̇={thdv:6.2f} u={uv:5.2f} → θ̈_true={t:7.3f} θ̈_pred={p:7.3f} err={abs(t-p):.3f}")

# Check U values
print("\n=== Potential energy U(θ) ===")
test_angles = torch.linspace(-torch.pi, torch.pi, 20)
cos_a = torch.cos(test_angles).requires_grad_(True)
sin_a = torch.sin(test_angles).requires_grad_(True)
U, dU = model._U_and_grad(cos_a, sin_a)
print(f"  θ=0:  U={U[(test_angles.abs()<0.1)].mean().item():.3f}  dU/dθ={dU[(test_angles.abs()<0.1)].mean().item():.3f}")
print(f"  θ=π/2: U={U[(test_angles-torch.pi/2).abs()<0.1].mean().item():.3f}  dU/dθ={dU[(test_angles-torch.pi/2).abs()<0.1].mean().item():.3f}")
print(f"  θ=π:   U={U[(test_angles.abs()>3.0)].mean().item():.3f}  dU/dθ={dU[(test_angles.abs()>3.0)].mean().item():.3f}")
print(f"  True:  U(0)=0, U(π/2)=10, U(π)=20")

print(f"\n  I = {model.I.item():.4f} (true I = 1.0)")

# Check energy
print("\n=== Energy check ===")
E = model.energy(cos_t, sin_t, thd_t)
print(f"  Mean energy: {E.mean().item():.2f}")
print(f"  Max energy: {E.max().item():.2f}")
print(f"  Min energy: {E.min().item():.2f}")
