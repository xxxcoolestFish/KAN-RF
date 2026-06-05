"""FeatureComputer + TinyDecisionNet: KAN-informed decision making.

FeatureComputer: computes physics-informed features from frozen KAN world model.
  - drift: f(s, a=0) — natural evolution without control
  - J:     df/da|_s — action Jacobian (via autograd through KAN)
  - rho:   activation density — how well the training data covers this state

TinyDecisionNet: a miniature MLP that synthesizes KAN-computed features into
an action, rather than learning physics from scratch.
"""
import torch
import torch.nn as nn
import numpy as np


class FeatureComputer:
    """Extract physics-informed features from a frozen KAN world model.

    Args:
        kan_model: frozen KAN world model
        k_norm: optional (1,1) tensor — if set, feeds k_norm as extra input dim
                for multi-scale models. e.g. torch.tensor([[0.5]]) for k=8.
    """

    def __init__(self, kan_model, device='cpu', k_norm=None):
        self.model = kan_model
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.device = device
        self.k_norm = k_norm  # (1,1) or None
        self._a_zero = torch.zeros(1, 1, device=device)

    def _build_input(self, s, a):
        """Build model input: [s, a] or [s, a, k_norm] for multi-scale."""
        if self.k_norm is not None:
            return torch.cat([s, a, self.k_norm.expand(s.shape[0], -1)], dim=-1)
        return torch.cat([s, a], dim=-1)

    # ── Drift ──────────────────────────────────────────────
    def compute_drift(self, s):
        """Natural state evolution under zero action: f(s, a=0).

        Args:
            s: (B, 3) normalized state [cos, sin, thd/8]
        Returns:
            drift: (B, 3) predicted next state with a=0
        """
        B = s.shape[0]
        a = self._a_zero.expand(B, -1)
        x = self._build_input(s, a)
        with torch.no_grad():
            return self.model(x)[:, :3]

    # ── Jacobian ───────────────────────────────────────────
    def compute_jacobian(self, s, a=None):
        """Action Jacobian: J = df/da|_s  (evaluated at a=0 by default).

        Uses autograd through the frozen KAN.
        For a [4,12,3] KAN: J is (B, 3, 1) — 3 output dims × 1 action dim.

        Args:
            s: (B, 3)
            a: (B, 1) or None → uses a=0
        Returns:
            J: (B, 3)  — gradient of each output dim w.r.t. a
        """
        B = s.shape[0]
        if a is None:
            a = self._a_zero.expand(B, -1).clone().requires_grad_(True)
        else:
            a = a.clone().requires_grad_(True)

        # We need per-output-dim Jacobian → autograd.grad with multiple outputs
        # Simpler: sum outputs and grads = J summed, but we want per-dim.
        # Use torch.autograd.functional.jacobian
        def fn(a_val):
            return self.model(self._build_input(s, a_val))[:, :3]  # (B, 3)

        J = torch.autograd.functional.jacobian(fn, a).squeeze(-1)  # (B, 3, B, 1) → (B, 3)
        return J

    def compute_jacobian_fast(self, s, a=None):
        """Compute Jacobian df/da via per-output-dim backward passes."""
        B = s.shape[0]
        J_rows = []
        for i in range(3):
            with torch.enable_grad():
                a_val = torch.zeros(B, 1, device=self.device, requires_grad=True)
                out_i = self.model(self._build_input(s, a_val))[:, i].sum()
                out_i.backward()
                J_rows.append(a_val.grad.clone().squeeze(-1))
        return torch.stack(J_rows, dim=1)  # (B, 3)

    # ── Activation Density ─────────────────────────────────
    def compute_density(self, s):
        """B-spline activation density: how well is this state covered?

        Returns rho ∈ [0, 1].  rho ≈ 1 → well-covered training region.
        rho ≈ 0 → model is extrapolating (SiLU baseline only).

        Strategy: compute mean absolute spline contribution vs total output.
        When B-splines are all inactive, spline_contribution ≈ 0.

        Args:
            s: (B, 3) normalized state
        Returns:
            rho: (B,) activation density per sample
        """
        B = s.shape[0]
        a = self._a_zero.expand(B, -1)
        x = self._build_input(s, a)

        # Use return_activations to get B and E
        output, B_list, E_list = self.model(x, return_activations=True)

        # Spline energy fraction per input (averaged across layers)
        densities = []
        for layer_idx, (B_mat, E_mat) in enumerate(zip(B_list, E_list)):
            # B_mat: (B, in_dim, n_basis)
            # E_mat: (B, out_dim, in_dim)
            # Activation density per input dim: fraction of active basis functions
            active = (B_mat > 1e-6).float().mean(dim=-1)  # (B, in_dim)
            densities.append(active)

        # Average across layers and input dimensions
        rho = torch.cat(densities, dim=1).mean(dim=1)  # (B,)
        return rho

    # ── Full feature computation ────────────────────────────
    def compute_features(self, s, s_target):
        """Compute all physics-informed features for decision making.

        Args:
            s:       (B, 3) current normalized state
            s_target:(1, 3) or (B, 3) target state (upright = [0, 1, 0])

        Returns dict with:
            drift:   (B, 3)  natural evolution under a=0
            gap:     (B, 3)  s_target - drift
            J:       (B, 3)  action Jacobian (gradient of each output dim w.r.t. a)
            align:   (B, 1)  cosine similarity between J and gap
            ctrl:    (B, 1)  ||J|| — how much action affects state
            trust:   (B, 1)  activation density
            a_init:  (B, 1)  Newton-step warm start: J^T·gap / (||J||² + ε)
        """
        if s_target.dim() == 2 and s_target.shape[0] == 1:
            s_target = s_target.expand(s.shape[0], -1)

        drift = self.compute_drift(s)
        gap = s_target - drift
        J = self.compute_jacobian_fast(s)   # (B, 3)
        trust = self.compute_density(s).unsqueeze(1)  # (B, 1)

        # Controllability: ||J||
        J_norm_sq = (J ** 2).sum(dim=1, keepdim=True)  # (B, 1)
        ctrl = J_norm_sq.sqrt().clamp(min=1e-8)

        # Directional alignment: cos_sim(J, gap)
        J_dot_gap = (J * gap).sum(dim=1, keepdim=True)  # (B, 1)
        gap_norm = gap.norm(dim=1, keepdim=True).clamp(min=1e-8)
        align = J_dot_gap / (ctrl * gap_norm + 1e-8)  # cos_sim

        # Newton-step warm start
        a_init = J_dot_gap / (J_norm_sq + 1e-4)

        return {
            'drift': drift,
            'gap': gap,
            'J': J,
            'align': align,
            'ctrl': ctrl,
            'trust': trust,
            'a_init': a_init,
        }


class TinyDecisionNet(nn.Module):
    """Micro MLP that synthesizes KAN-computed features into (action, timescale).

    Input (~10D): [a_init, gap(3), align, ctrl, trust, s(3)]
    Output: (a ∈ [-1, 1], k_cont ∈ [0, 1]) — action and continuous timescale.
            k_cont maps to discrete k ∈ {1, 2, 4, 8, 16} via k = round(k_cont * 16).
    """

    def __init__(self, hidden=32, output_k=True):
        super().__init__()
        self.output_k = output_k
        out_dim = 2 if output_k else 1
        self.net = nn.Sequential(
            nn.Linear(10, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, features, s):
        """Returns (a, k_cont) if output_k else a."""
        x = torch.cat([
            features['a_init'],
            features['gap'],
            features['align'],
            features['ctrl'],
            features['trust'],
            s,
        ], dim=-1)
        out = self.net(x)
        if self.output_k:
            a = torch.tanh(out[:, 0:1])         # a ∈ [-1, 1]
            k_cont = torch.sigmoid(out[:, 1:2])  # k_cont ∈ [0, 1]
            return torch.cat([a, k_cont], dim=-1)
        return torch.tanh(out)
