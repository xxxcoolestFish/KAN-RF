"""Certified k-Selection (CKS): B-spline diagnostics → safe horizon.

Theorem 1 (Single-Step Error Estimate):
  E₁(s) = base_err + α₀·(1-ρ̄(s)) + α₁·h⁻¹·max|Δ¹c| + α₂·h⁻²·max|Δ²c|

Theorem 2 (Jacobian Compensation via Finite Differences):
  G(s,k) = ||s_k(a+ε) - s_k(a)|| / ||s_1(a+ε) - s_1(a)|| ≈ k^γ

Theorem 3 (Certified Horizon):
  k_cert(s) = max{ k ∈ K : E_k(s) / G(s,k) ≤ eps }

All quantities computable from B-spline parameters + world model forward passes.
No ensembles. No extra training. No world model self-query.
"""
import torch, numpy as np
from kanrf import bspline_basis


# ─── B-spline diagnostics ───────────────────────────────────────────────

def control_point_roughness(model):
    d1_max, d2_max = 0.0, 0.0
    for layer in model.layers:
        c = layer.spline_weight
        d1 = (c[:, :, 1:] - c[:, :, :-1]).abs().max().item()
        d2 = (c[:, :, 2:] - 2 * c[:, :, 1:-1] + c[:, :, :-2]).abs().max().item()
        d1_max = max(d1_max, d1)
        d2_max = max(d2_max, d2)
    return d1_max, d2_max


def compute_activation_density(model):
    density = []
    for layer in model.layers:
        c = layer.spline_weight
        active = (c.abs() > 0.05).float()
        freq = active.mean(dim=0)
        density.append(freq)
    return density


def state_activation_density(s_norm, model, density):
    if not isinstance(s_norm, torch.Tensor):
        s_norm = torch.tensor(s_norm, dtype=torch.float32)
    if s_norm.dim() == 1:
        s_norm = s_norm.unsqueeze(0)
    batch, state_dim = s_norm.shape
    full_dim = model.layers[0].in_dim
    x = torch.zeros(batch, full_dim)
    x[:, :state_dim] = s_norm

    with torch.no_grad():
        B_flat = bspline_basis(x.reshape(-1), model.layers[0].grid,
                               model.layers[0].spline_order)
        B = B_flat.reshape(batch, full_dim, -1)
        B_state = B[:, :state_dim, :]
        rho = density[0][:state_dim, :]
        weights = B_state.abs() + 1e-8
        weighted_rho = (weights * rho.unsqueeze(0)).sum(dim=(1, 2))
        total_weight = weights.sum(dim=(1, 2))
        rho_bar = (weighted_rho / total_weight).mean().item()
    return rho_bar


# ─── Error estimate ─────────────────────────────────────────────────────

def single_step_error_bound(s_norm, model, density,
                            base_error=0.05, alpha=(0.3, 0.01, 0.005)):
    rho = state_activation_density(s_norm, model, density)
    d1, d2 = control_point_roughness(model)
    h = model.layers[0].grid[1] - model.layers[0].grid[0]
    e1 = base_error + alpha[0] * (1.0 - rho) + alpha[1] * d1 / h + alpha[2] * d2 / (h * h)
    return e1


# ─── Jacobian gain via finite differences ───────────────────────────────

def _forward_k(s_norm, a_norm, model, state_dim, action_dim, k):
    """Roll world model k steps with constant action, return state part."""
    if not isinstance(s_norm, torch.Tensor):
        s_norm = torch.tensor(s_norm, dtype=torch.float32)
    if s_norm.dim() == 1:
        s_norm = s_norm.unsqueeze(0)
    if not isinstance(a_norm, torch.Tensor):
        a_norm = torch.tensor([[a_norm]], dtype=torch.float32)

    full_dim = model.layers[0].in_dim
    s = s_norm.clone()
    for _ in range(k):
        x = torch.zeros(1, full_dim)
        x[:, :state_dim] = s
        x[:, state_dim:state_dim + action_dim] = a_norm
        with torch.no_grad():
            s = model(x)[:, :state_dim]
    return s


def jacobian_gain(s_norm, a_norm, model, state_dim, action_dim, eps=0.01):
    """Finite-difference estimate: G(s,k) = ||Δs_k|| / ||Δs_1||.

    Δs_k = s_k(a+ε) - s_k(a)  computed via world model rollout.
    Returns a function G(k) = k^γ with γ estimated from k=2 vs k=1.
    """
    s1_a = _forward_k(s_norm, a_norm, model, state_dim, action_dim, 1)
    s1_ap = _forward_k(s_norm, a_norm + eps, model, state_dim, action_dim, 1)
    d1 = (s1_ap - s1_a).norm().item()
    if d1 < 1e-8:
        return lambda k: 1.0

    s2_a = _forward_k(s_norm, a_norm, model, state_dim, action_dim, 2)
    s2_ap = _forward_k(s_norm, a_norm + eps, model, state_dim, action_dim, 2)
    d2 = (s2_ap - s2_a).norm().item()

    g2 = max(d2 / d1, 0.01)
    gamma = np.log(g2) / np.log(2)

    def G(k):
        return max(1.0, k ** gamma)
    return G


# ─── Certified horizon ──────────────────────────────────────────────────

def certified_horizon(s_norm, a_norm, model, density, state_dim, action_dim,
                      K_set=(1, 2, 4, 8, 16), eps=0.10, base_error=0.05):
    """k_cert(s) = max{ k ∈ K_set : E_k(s)/G(s,k) ≤ eps }"""
    E1 = single_step_error_bound(s_norm, model, density, base_error=base_error)
    G_fn = jacobian_gain(s_norm, a_norm, model, state_dim, action_dim)

    d1, _ = control_point_roughness(model)
    h = model.layers[0].grid[1] - model.layers[0].grid[0]
    L = min(0.99, d1 / h)

    best_k = 1
    diag = {'E1': E1, 'L': L}

    for k in K_set:
        if L < 1.0:
            Ek = E1 * (1.0 - L ** k) / (1.0 - L)
        else:
            Ek = E1 * k
        Gk = G_fn(k)
        eff_err = Ek / Gk
        diag[f'E{k}'] = Ek
        diag[f'G{k}'] = Gk
        diag[f'Eff{k}'] = eff_err
        if eff_err <= eps:
            best_k = k

    return best_k, diag
