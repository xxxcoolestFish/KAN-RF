"""Regularization utilities for KAN training.

Extracted from training scripts so they can be shared without
cross-script imports (eval shouldn't import from train).
"""
import torch


def p_spline_penalty(model):
    """Sum of ||Delta^2 c||^2 across all spline weights, averaged over control points.

    Penalizes first-derivative energy of each B-spline edge function.
    Lower values = smoother derivatives.
    """
    total = 0.0
    for layer in model.layers:
        c = layer.spline_weight          # (out, in, n_basis)
        d2 = c[:, :, :-2] - 2 * c[:, :, 1:-1] + c[:, :, 2:]
        total += (d2 ** 2).mean()
    return total


def true_jacobian(s_next_norm):
    """Analytic Jacobian ds'_norm / da_norm for Pendulum-v1.

    Args:
        s_next_norm: (..., 3) normalized next state [cos, sin, thd/8]
    Returns:
        J: (..., 3) Jacobian [dcos/da, dsin/da, dthd/da] in normalized units
    """
    cos_p, sin_p = s_next_norm[..., 0], s_next_norm[..., 1]
    J_cos = -0.015 * sin_p
    J_sin =  0.015 * cos_p
    J_thd =  0.0375 * torch.ones_like(cos_p)
    return torch.stack([J_cos, J_sin, J_thd], dim=-1)


def jacobian_loss(model, s_batch, a_batch, y_batch, w):
    """Per-sample Jacobian matching loss with controllability weights.

    Penalizes ||w ⊙ (df/da - J_true)||^2.

    Args:
        model: KAN world model
        s_batch: (B, 3) normalized states
        a_batch: (B, 1) normalized actions
        y_batch: (B, 3) normalized next states (for evaluating J_true)
        w: (3,) per-dimension controllability weights
    Returns:
        scalar loss
    """
    a = a_batch.clone().detach().requires_grad_(True)
    s_pred = model(torch.cat([s_batch, a], dim=-1))
    J_model = []
    for dim in range(3):
        g = torch.autograd.grad(
            s_pred[:, dim].sum(), a,
            retain_graph=True, create_graph=True
        )[0]
        J_model.append(g)
    J_model = torch.cat(J_model, dim=-1)  # (B, 3)
    J_true = true_jacobian(y_batch)
    err = (J_model - J_true) ** 2
    weighted = (err * w.unsqueeze(0)).mean()
    return weighted
