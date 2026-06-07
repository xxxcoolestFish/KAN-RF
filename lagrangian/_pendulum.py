"""Pendulum Lagrangian Network — 1-DoF.

L = ½Iθ̇² - U(cosθ, sinθ)
EL: I·θ̈ + U'(θ) = u  →  θ̈ = (u - U'(θ)) / I

Refactored from lagrangian_v2.py into the lagrangian/ package.
"""
import torch
import torch.nn as nn


class LagNet(nn.Module):
    """Lagrangian network for Pendulum-v1 (1-DoF)."""

    def __init__(self, hidden=64):
        super().__init__()
        self.log_I = nn.Parameter(torch.tensor(-1.1))  # init: exp(-1.1) ≈ 0.333

        self.net = nn.Sequential(
            nn.Linear(2, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1)
        )
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.1)
                nn.init.zeros_(layer.bias)

    @property
    def I(self):
        return torch.exp(self.log_I).clamp(min=0.01)

    def U(self, cos_theta, sin_theta):
        """Potential energy."""
        x = torch.stack([cos_theta, sin_theta], dim=-1)
        return self.net(x).squeeze(-1)

    def dU_dtheta(self, cos_theta, sin_theta):
        """∂U/∂θ via autograd."""
        dU_dcos, dU_dsin = torch.autograd.grad(
            outputs=self.U(cos_theta, sin_theta).sum(),
            inputs=[cos_theta, sin_theta],
            create_graph=True, retain_graph=True
        )
        return -dU_dcos * sin_theta + dU_dsin * cos_theta

    def forward(self, cos_t, sin_t, theta_dot, action):
        """Predict θ̈ = (u - U'(θ)) / I."""
        U_prime = self.dU_dtheta(cos_t, sin_t)
        return (action - U_prime) / self.I

    def energy(self, cos_t, sin_t, theta_dot):
        """E = ½Iθ̇² + U(θ)."""
        return 0.5 * self.I * theta_dot**2 + self.U(cos_t, sin_t)
