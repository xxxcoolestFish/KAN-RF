"""Decision KAN: maps (state, target) → (action, horizon_class)."""
import torch
from kanrf import KAN


class DecisionKAN(torch.nn.Module):
    """KAN that outputs action + horizon class given current and target state."""

    def __init__(self, hidden_dim=10, n_horizon_classes=30):
        super().__init__()
        self.n_horizon_classes = n_horizon_classes
        # Input: [cos, sin, thd/8] × (current + target) = 6 dims
        # Output: 1 (action) + n_horizon_classes (H logits)
        out_dim = 1 + n_horizon_classes
        self.kan = KAN([6, hidden_dim, out_dim], grid_size=5, spline_order=3)

    def forward(self, s_norm, s_target_norm):
        x = torch.cat([s_norm, s_target_norm], dim=-1)  # (batch, 6)
        out = self.kan(x)                                 # (batch, 1 + n_classes)
        a_norm = out[:, 0:1]                              # (batch, 1)
        h_logits = out[:, 1:]                             # (batch, n_classes)
        return a_norm, h_logits
