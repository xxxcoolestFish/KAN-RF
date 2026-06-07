"""CartPole Lagrangian Network — 2-DoF with structured mass matrix.

System: CartPole-v1. Generalized coordinates: q = [x, θ].
  x: cart position (cyclic — ∂L/∂x = 0)
  θ: pole angle from vertical upright (θ=0 → pole up)

Lagrangian:
  L = ½(M+m)ẋ² + ½Iθ̇² + mLẋθ̇cosθ - U(θ)
  where I = 4/3·mL² (pole rotational inertia about pivot)

Euler-Lagrange (with external force F on cart, 0 torque on pole):
  [M+m,  mLcosθ] [ẍ]   [F + mLθ̇²sinθ]
  [mLcosθ, I  ] [θ̈] = [   -U'(θ)    ]

True physics: M=1.0, m=0.1, L=0.5 (half-length), U(θ) = -mgLcosθ
  → mgL = 0.49, I = 0.0333..., -U'(θ) = mgL sinθ
"""
import torch
import torch.nn as nn


class CartPoleLagNet(nn.Module):
    """Lagrangian network for CartPole-v1.

    Learns:
      - Physical parameters: mp (pole mass), mc (cart mass), length
      - Potential energy: U(cosθ, sinθ) via MLP

    Outputs: (x_ddot, th_ddot) given (cosθ, sinθ, x_dot, th_dot, force).
    """

    def __init__(self, hidden=32):
        super().__init__()
        # Physical parameters (log-space for positivity)
        # Init: mp≈0.1, mc≈1.0, length≈0.5
        self.log_mp = nn.Parameter(torch.tensor(-2.3))    # exp(-2.3) ≈ 0.1
        self.log_mc = nn.Parameter(torch.tensor(0.0))     # exp(0) = 1.0
        self.log_len = nn.Parameter(torch.tensor(-0.7))   # exp(-0.7) ≈ 0.5

        # Potential energy U(cosθ, sinθ)
        self.U_net = nn.Sequential(
            nn.Linear(2, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1)
        )
        for layer in self.U_net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.1)
                nn.init.zeros_(layer.bias)

    # -- Derived quantities --
    @property
    def mp(self):
        return torch.exp(self.log_mp).clamp(min=0.001)

    @property
    def mc(self):
        return torch.exp(self.log_mc).clamp(min=0.001)

    @property
    def length(self):
        return torch.exp(self.log_len).clamp(min=0.01)

    @property
    def I_theta(self):
        """Pole rotational inertia about pivot = 4/3·m·L²."""
        return (4.0 / 3.0) * self.mp * self.length ** 2

    # -- Potential energy --
    def U(self, cos_theta, sin_theta):
        """Potential energy U(cosθ, sinθ). True: U = -mgL·cosθ ≈ -0.49·cosθ."""
        x = torch.stack([cos_theta, sin_theta], dim=-1)
        return self.U_net(x).squeeze(-1)

    def dU_dtheta(self, cos_theta, sin_theta):
        """∂U/∂θ via autograd.  True: U'(θ) = mgL·sinθ ≈ 0.49·sinθ."""
        dU_dcos, dU_dsin = torch.autograd.grad(
            outputs=self.U(cos_theta, sin_theta).sum(),
            inputs=[cos_theta, sin_theta],
            create_graph=True, retain_graph=True
        )
        # Chain rule: ∂U/∂θ = ∂U/∂cosθ · (-sinθ) + ∂U/∂sinθ · cosθ
        return -dU_dcos * sin_theta + dU_dsin * cos_theta

    # -- Forward dynamics --
    def forward(self, cos_theta, sin_theta, x_dot, th_dot, force):
        """Predict accelerations [x_ddot, th_ddot].

        Args:
            cos_theta, sin_theta:  pole angle representation  [batch]
            x_dot:                 cart velocity               [batch]
            th_dot:                pole angular velocity       [batch]
            force:                 applied force on cart       [batch]

        Returns:
            x_ddot:   cart acceleration    [batch]
            th_ddot:  pole ang.accel       [batch]
        """
        # Mass matrix
        M11 = self.mc + self.mp
        M12 = self.mp * self.length * cos_theta
        M22 = self.I_theta

        det = M11 * M22 - M12 ** 2  # always >0 for SPD M

        # RHS
        rhs_x = force + self.mp * self.length * th_dot ** 2 * sin_theta
        rhs_th = -self.dU_dtheta(cos_theta, sin_theta)  # generalized force from -U'(θ)

        x_ddot = (M22 * rhs_x - M12 * rhs_th) / det
        th_ddot = (-M12 * rhs_x + M11 * rhs_th) / det

        return x_ddot, th_ddot

    # -- Energy --
    def energy(self, cos_theta, sin_theta, x_dot, th_dot):
        """Total mechanical energy E = T + U.

        E = ½(M+m)ẋ² + ½Iθ̇² + mLẋθ̇cosθ + U(θ)
        """
        T = (0.5 * (self.mc + self.mp) * x_dot ** 2
             + 0.5 * self.I_theta * th_dot ** 2
             + self.mp * self.length * x_dot * th_dot * cos_theta)
        return T + self.U(cos_theta, sin_theta)

    # -- Diagnostics --
    def true_physics(self):
        """Return the true physical parameters (for comparison during training)."""
        return {
            'mp_true': 0.1,
            'mc_true': 1.0,
            'len_true': 0.5,
            'I_true': (4.0/3.0)*0.1*0.25,
            'mp': self.mp.item(),
            'mc': self.mc.item(),
            'length': self.length.item(),
            'I': self.I_theta.item(),
        }
