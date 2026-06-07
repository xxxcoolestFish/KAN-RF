#!/usr/bin/env python3
"""
Lagrangian Neural Network v2 - Direct & Simple

Key insight: For Gymnasium Pendulum-v1:
  θ̈ = 3u + 15·sin(θ)

The Lagrangian L = ½·I·θ̇² - U(θ) gives:
  I·θ̈ + U'(θ) = u
  → θ̈ = (u - U'(θ)) / I

Matching: I = 1/3, U'(θ) = -5·sin(θ), U(θ) = -5·cos(θ)

This version:
1. Removes state-prediction regularization (focuses on acceleration)
2. Uses direct regression target
3. Larger LR, simpler architecture
"""
import torch
import torch.nn as nn


class LagNet(nn.Module):
    """Simple Lagrangian network for Pendulum."""
    
    def __init__(self, hidden=64):
        super().__init__()
        # I: scalar moment of inertia
        self.log_I = nn.Parameter(torch.tensor(-1.1))  # init: exp(-1.1) ≈ 0.333
        
        # U(cosθ, sinθ): potential energy
        # Simple: 2 hidden layers with tanh
        self.net = nn.Sequential(
            nn.Linear(2, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1)
        )
        
        # Initialize for small output
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.1)
                nn.init.zeros_(layer.bias)
    
    @property
    def I(self):
        return torch.exp(self.log_I).clamp(min=0.01)
    
    def U(self, cos_theta, sin_theta):
        """Potential energy U(cosθ, sinθ)."""
        x = torch.stack([cos_theta, sin_theta], dim=-1)
        return self.net(x).squeeze(-1)
    
    def dU_dtheta(self, cos_theta, sin_theta):
        """∂U/∂θ via autograd."""
        dU_dcos, dU_dsin = torch.autograd.grad(
            outputs=self.U(cos_theta, sin_theta).sum(),
            inputs=[cos_theta, sin_theta],
            create_graph=True,
            retain_graph=True
        )
        # dU/dθ = ∂U/∂cosθ * (-sinθ) + ∂U/∂sinθ * cosθ
        return -dU_dcos * sin_theta + dU_dsin * cos_theta
    
    def forward(self, cos_t, sin_t, theta_dot, action):
        """
        Predict θ̈ = (action - U'(θ)) / I.
        
        Args:
            cos_t, sin_t: angle representation [batch]
            theta_dot: angular velocity [batch]
            action: torque [batch]
        Returns:
            theta_ddot: angular acceleration [batch]
        """
        U_prime = self.dU_dtheta(cos_t, sin_t)
        return (action - U_prime) / self.I
    
    def energy(self, cos_t, sin_t, theta_dot):
        """E = ½Iθ̇² + U(θ)."""
        return 0.5 * self.I * theta_dot**2 + self.U(cos_t, sin_t)


def make_loss_fn():
    """Simple MSE loss on angular acceleration."""
    def loss_fn(model, cos_t, sin_t, thd_t, u_t, cos_n, sin_n, thd_n):
        # Enable grads on angle inputs
        cos_t = cos_t.detach().requires_grad_(True)
        sin_t = sin_t.detach().requires_grad_(True)
        
        # True angular acceleration from θ̇ difference
        dt = 0.05
        theta_ddot_true = (thd_n - thd_t) / dt
        
        # Predicted from Lagrangian
        theta_ddot_pred = model(cos_t, sin_t, thd_t, u_t)
        
        loss = torch.mean((theta_ddot_pred - theta_ddot_true) ** 2)
        
        return loss, {
            'loss': loss.item(),
            'I': model.I.item(),
            'pred_mean': theta_ddot_pred.mean().item(),
            'true_mean': theta_ddot_true.mean().item()
        }
    return loss_fn
