"""Acrobot Lagrangian Network — 2-DoF underactuated with structured mass matrix.

Acrobot-v1: two-link pendulum, joint 1 unactuated, joint 2 actuated.
  Generalized coordinates: q = [θ₁, θ₂] (angles from vertically-down)
  State: [cosθ₁, sinθ₁, cosθ₂, sinθ₂, θ̇₁, θ̇₂] (6D)
  Action: {0, 1, 2} → torque = [-1, 0, +1] on joint 2 only

Mass matrix (depends only on θ₂):
  M₁₁ = a₁ + b₁·cosθ₂,   M₁₂ = a₂ + b₂·cosθ₂,   M₂₂ = a₂
  (Note: M₂₂ shares intercept with M₁₂ — true for Acrobot physics)

Derived analytically from:
  a₁ = I₁+m₁lc₁²+m₂(l₁²+lc₂²)+I₂ = 3.5
  b₁ = 2m₂l₁lc₂ = 1.0
  a₂ = m₂lc₂²+I₂ = 1.25
  b₂ = m₂l₁lc₂ = 0.5

Coriolis (from Ṁ = ∂M/∂θ₂ · θ̇₂):
  C₁ = -b₁·sinθ₂·θ̇₁θ̇₂ - b₂·sinθ₂·θ̇₂²
  C₂ =  ½b₁·sinθ₂·θ̇₁²

Euler-Lagrange:  M(q)·q̈ + C(q,q̇)·q̇ + ∇U(q) = [0, τ]ᵀ
  → q̈ = M⁻¹([0, τ]ᵀ − C − ∇U)
"""
import torch
import torch.nn as nn


class AcrobotLagNet(nn.Module):
    """Lagrangian network for Acrobot-v1.

    Learns:
      - Mass matrix parameters: a₁, b₁, a₂, b₂ (4 scalars)
      - Potential energy: U(cosθ₁, sinθ₁, cosθ₂, sinθ₂) via MLP

    Forward: predicts [θ̈₁, θ̈₂] from (cosθ₁, sinθ₁, cosθ₂, sinθ₂, θ̇₁, θ̇₂, torque).
    """

    def __init__(self, hidden=32):
        super().__init__()
        # Mass matrix parameters (init near true values)
        self.a1 = nn.Parameter(torch.tensor(3.5))   # a₁ = 3.5
        self.b1 = nn.Parameter(torch.tensor(1.0))   # b₁ = 1.0
        self.a2 = nn.Parameter(torch.tensor(1.25))  # a₂ = 1.25
        self.b2 = nn.Parameter(torch.tensor(0.5))   # b₂ = 0.5

        # Potential energy U(cosθ₁, sinθ₁, cosθ₂, sinθ₂)
        self.U_net = nn.Sequential(
            nn.Linear(4, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1)
        )
        for layer in self.U_net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.1)
                nn.init.zeros_(layer.bias)

    # -- Mass matrix --
    def mass_matrix(self, cos2):
        """Returns M₁₁, M₁₂, M₂₂ as batch tensors."""
        M11 = self.a1 + self.b1 * cos2
        M12 = self.a2 + self.b2 * cos2
        M22 = self.a2  # shares intercept with M₁₂
        return M11, M12, M22

    # -- Coriolis --
    def coriolis(self, sin2, thd1, thd2):
        """Returns C₁, C₂ (Coriolis + centripetal generalized forces)."""
        C1 = -self.b1 * sin2 * thd1 * thd2 - self.b2 * sin2 * thd2 ** 2
        C2 = 0.5 * self.b1 * sin2 * thd1 ** 2
        return C1, C2

    # -- Potential energy --
    def U(self, cos1, sin1, cos2, sin2):
        """Potential energy U(cosθ₁, sinθ₁, cosθ₂, sinθ₂)."""
        x = torch.stack([cos1, sin1, cos2, sin2], dim=-1)
        return self.U_net(x).squeeze(-1)

    def grad_U(self, cos1, sin1, cos2, sin2):
        """Gradient ∇U = [∂U/∂θ₁, ∂U/∂θ₂] via autograd."""
        grads = torch.autograd.grad(
            outputs=self.U(cos1, sin1, cos2, sin2).sum(),
            inputs=[cos1, sin1, cos2, sin2],
            create_graph=True, retain_graph=True
        )
        dU_dc1, dU_ds1, dU_dc2, dU_ds2 = grads
        # Chain rule: ∂U/∂θ = ∂U/∂cosθ·(-sinθ) + ∂U/∂sinθ·cosθ
        dU_d1 = -dU_dc1 * sin1 + dU_ds1 * cos1
        dU_d2 = -dU_dc2 * sin2 + dU_ds2 * cos2
        return dU_d1, dU_d2

    # -- Forward dynamics --
    def forward(self, cos1, sin1, cos2, sin2, thd1, thd2, torque):
        """Predict angular accelerations [θ̈₁, θ̈₂].

        Args:
            cos1, sin1, cos2, sin2: angle representations [batch]
            thd1, thd2: angular velocities [batch]
            torque: applied torque on joint 2 [batch]
        Returns:
            thdd1, thdd2: angular accelerations [batch]
        """
        M11, M12, M22 = self.mass_matrix(cos2)
        C1, C2 = self.coriolis(sin2, thd1, thd2)
        g1, g2 = self.grad_U(cos1, sin1, cos2, sin2)

        det = M11 * M22 - M12 ** 2

        # RHS = B·τ − C − ∇U,  B = [0, 1]ᵀ
        rhs1 = -C1 - g1                      # no torque on joint 1
        rhs2 = torque - C2 - g2               # torque on joint 2

        thdd1 = (M22 * rhs1 - M12 * rhs2) / det
        thdd2 = (-M12 * rhs1 + M11 * rhs2) / det

        return thdd1, thdd2

    # -- Semi-implicit Euler state prediction --
    def predict_next_state(self, cos1, sin1, cos2, sin2, thd1, thd2, torque, dt=0.2):
        """Predict next state via Lagrangian + semi-implicit Euler.

        Returns (cos1_n, sin1_n, cos2_n, sin2_n, thd1_n, thd2_n).
        """
        thdd1, thdd2 = self.forward(cos1, sin1, cos2, sin2, thd1, thd2, torque)
        theta1 = torch.atan2(sin1, cos1)
        theta2 = torch.atan2(sin2, cos2)

        thd1_n = thd1 + dt * thdd1
        thd2_n = thd2 + dt * thdd2
        theta1_n = theta1 + dt * thd1_n
        theta2_n = theta2 + dt * thd2_n

        return (torch.cos(theta1_n), torch.sin(theta1_n),
                torch.cos(theta2_n), torch.sin(theta2_n),
                thd1_n, thd2_n)

    # -- Energy --
    def energy(self, cos1, sin1, cos2, sin2, thd1, thd2):
        """Total mechanical energy E = T + U."""
        M11, M12, M22 = self.mass_matrix(cos2)
        T = 0.5 * M11 * thd1**2 + M12 * thd1 * thd2 + 0.5 * M22 * thd2**2
        return T + self.U(cos1, sin1, cos2, sin2)

    # -- Diagnostics --
    def true_params(self):
        return {
            'a1': self.a1.item(), 'a1_true': 3.5,
            'b1': self.b1.item(), 'b1_true': 1.0,
            'a2': self.a2.item(), 'a2_true': 1.25,
            'b2': self.b2.item(), 'b2_true': 0.5,
        }
