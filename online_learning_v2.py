"""KAN Online Learning v2: Three-factor dynamic learning rate.

Three factors modulate per-control-point learning rate:
  1. Error-driven:   η ∝ min(||e||/σ_train, 10)  (how wrong NOW?)
  2. Training density: η ∝ (1 - ρ_{j,k})           (was this trained before?)
  3. Online count:     η ∝ 1/√(1 + N_{i,j,k})      (how many updates so far?)

Usage:
  # 1. After training, compute statistics:
  stats = compute_training_stats(model, train_x, train_y)

  # 2. During control, after each env step:
  online_update_three_factor(model, stats, s, a, s_true, eta0=1e-3)
"""
import torch
import torch.nn.functional as F
from kanrf import KAN


def compute_training_stats(model: KAN, x_train, y_train):
    """Compute per-basis-function activation density and typical error.

    Returns dict with:
      density: list of (in_dim, n_basis) tensors — activation frequency
      sigma_train: scalar — typical ||prediction - truth||
    """
    model.eval()
    density = []

    with torch.no_grad():
        # Forward pass with activations on training data
        _, B_list, _ = model(x_train, return_activations=True)

        for B in B_list:
            # B: (N, in_dim, n_basis)
            # Frequency: fraction of samples where B_k > 0
            active = (B > 1e-8).float()  # (N, in_dim, n_basis)
            freq = active.mean(dim=0)    # (in_dim, n_basis)
            density.append(freq)

        # Typical prediction error
        pred = model(x_train)
        errors = (pred - y_train).norm(dim=-1)  # per-sample L2
        sigma_train = errors.mean().item()

    model.train()
    return {'density': density, 'sigma_train': sigma_train}


class ThreeFactorUpdater:
    """Online KAN update with three-factor dynamic learning rate.

    Δc_{i,j,k} = -η₀ · min(||e||/σ, 10) · (1-ρ_{j,k}) / √(1+N_{i,j,k}) · ∂L/∂c
    """

    def __init__(self, model: KAN, stats: dict, eta0: float = 1e-3):
        self.model = model
        self.eta0 = eta0
        self.density = stats['density']     # list of (in_dim, n_basis)
        self.sigma_train = stats['sigma_train']

        # Online update counter per control point (same device as model)
        self.online_count = []
        for layer in model.layers:
            shape = layer.spline_weight.shape
            self.online_count.append(torch.zeros(shape, device=layer.spline_weight.device))

    def update(self, s_norm, a_norm, s_true_norm, k_norm=None):
        """One online update with three-factor learning rate.

        Args:
            s_norm: (1, 3) normalized state
            a_norm: (1, 1) normalized action
            s_true_norm: (1, 3) normalized true next state
            k_norm: (1, 1) optional timestep scale (for multi-scale models)
        Returns:
            error_L2: float, prediction error magnitude
            max_eta: float, maximum effective learning rate used
        """
        parts = [s_norm, a_norm]
        if k_norm is not None:
            parts.append(k_norm)
        x = torch.cat(parts, dim=-1)

        # Forward with activations
        y, B_list, _ = self.model(x, return_activations=True)

        # Error
        e = y - s_true_norm
        error_L2 = e.norm().item()

        # Error-driven modulation factor
        err_factor = min(error_L2 / (self.sigma_train + 1e-8), 10.0)

        # Standard backprop to get gradients (with clipping)
        loss = (e ** 2).sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        max_eta_used = 0.0

        with torch.no_grad():
            for layer_idx, layer in enumerate(self.model.layers):
                # --- Base weight: error-driven only (no density, no count) ---
                layer.base_weight.data -= \
                    self.eta0 * err_factor * layer.base_weight.grad
                layer.base_weight.grad.zero_()

                # --- Spline weight: three-factor update ---
                # B_list[layer_idx]: (1, in_dim, n_basis)
                B = B_list[layer_idx].squeeze(0)  # (in_dim, n_basis)
                rho = self.density[layer_idx]     # (in_dim, n_basis)
                count = self.online_count[layer_idx]  # (out_dim, in_dim, n_basis)
                grad = layer.spline_weight.grad       # (out_dim, in_dim, n_basis)

                for j in range(layer.in_dim):
                    for k in range(layer.spline_weight.shape[2]):
                        if B[j, k] < 1e-8:
                            continue  # inactive basis → skip

                        # Three factors
                        rho_factor = 1.0 - rho[j, k].item()     # training density
                        count_factor = 1.0 / (1.0 + count[:, j, k]).sqrt()  # online count
                        eta_eff = self.eta0 * err_factor * rho_factor

                        if eta_eff > max_eta_used:
                            max_eta_used = eta_eff

                        layer.spline_weight.data[:, j, k] -= \
                            (eta_eff * count_factor *
                             grad[:, j, k])

                        self.online_count[layer_idx][:, j, k] += 1.0

                layer.spline_weight.grad.zero_()

        return error_L2, max_eta_used
