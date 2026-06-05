import torch


class PointMass:
    """2D point-mass: s_{t+1} = s_t + a_t."""

    def __init__(self, nonlinear: bool = False):
        self.nonlinear = nonlinear
        self.state_dim = 2
        self.action_dim = 2

    def step(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Return next state given current state and action.

        Args:
            s: (batch, 2) current state
            a: (batch, 2) action
        Returns:
            s_next: (batch, 2)
        """
        if self.nonlinear:
            # Add mild nonlinear coupling
            return s + a + 0.1 * torch.sin(s) * torch.cos(a)
        return s + a


def generate_data(env: PointMass, n_samples: int,
                  state_range: float = 1.0, action_range: float = 0.5
                  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate (s, a, s') triples by random sampling."""
    s = (torch.rand(n_samples, 2) * 2 - 1) * state_range
    a = (torch.rand(n_samples, 2) * 2 - 1) * action_range
    with torch.no_grad():
        s_next = env.step(s, a)
    # Input to world model: concat(s, a)
    x = torch.cat([s, a], dim=-1)  # (n, 4)
    return x, s_next
