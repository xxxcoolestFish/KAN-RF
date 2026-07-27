"""Gradient coupling for ordered KAN spline and ProtoKAN parameters."""

from __future__ import annotations

import torch


def path_graph_laplacian(
    size: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Return the positive-semidefinite Laplacian of a path with ``size`` nodes."""
    if size < 1:
        raise ValueError("size must be at least one")
    dtype = dtype or torch.get_default_dtype()
    laplacian = torch.zeros(size, size, device=device, dtype=dtype)
    if size > 1:
        indices = torch.arange(size - 1, device=device)
        laplacian[indices, indices] += 1
        laplacian[indices + 1, indices + 1] += 1
        laplacian[indices, indices + 1] -= 1
        laplacian[indices + 1, indices] -= 1
    return laplacian


def implicit_diffusion_matrix(
    size: int,
    tau: float,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Return ``(I + tau * L)^-1`` for the path-graph Laplacian ``L``."""
    if tau < 0:
        raise ValueError("tau must be non-negative")
    dtype = dtype or torch.get_default_dtype()
    identity = torch.eye(size, device=device, dtype=dtype)
    if tau == 0:
        return identity
    laplacian = path_graph_laplacian(size, device=device, dtype=dtype)
    return torch.linalg.solve(identity + tau * laplacian, identity)


@torch.no_grad()
def diffuse_gradient_(parameter: torch.nn.Parameter, tau: float) -> None:
    """Diffuse ``parameter.grad`` along its last axis, in place."""
    if parameter.grad is None:
        return
    diffusion = implicit_diffusion_matrix(
        parameter.grad.shape[-1],
        tau,
        device=parameter.grad.device,
        dtype=parameter.grad.dtype,
    )
    parameter.grad.copy_(parameter.grad @ diffusion)


@torch.no_grad()
def diffuse_kan_gradients_(model: torch.nn.Module, tau: float) -> None:
    """Diffuse every available ``spline_weight`` gradient in a KAN."""
    for layer in model.layers:
        diffuse_gradient_(layer.spline_weight, tau)


@torch.no_grad()
def diffuse_protokan_gradients_(
    model: torch.nn.Module,
    tau: float,
) -> None:
    """Diffuse value and derivative prototype gradients in a ProtoKAN."""
    for layer in model.layers:
        diffuse_gradient_(layer.proto_val, tau)
        diffuse_gradient_(layer.proto_der, tau)


@torch.no_grad()
def diffuse_model_gradients_(
    model: torch.nn.Module,
    strength: float,
) -> None:
    """Diffuse ordered spline/prototype gradients of a KAN-like model."""
    for layer in model.layers:
        if hasattr(layer, "spline_weight"):
            diffuse_gradient_(layer.spline_weight, strength)
        elif hasattr(layer, "proto_val") and hasattr(layer, "proto_der"):
            diffuse_gradient_(layer.proto_val, strength)
            diffuse_gradient_(layer.proto_der, strength)
        else:
            raise TypeError(
                "layer must expose spline_weight or both proto_val and proto_der"
            )


@torch.no_grad()
def diffuse_input_dimension_gradients_(
    model: torch.nn.Module,
    input_index: int,
    strength: float,
    layer_index: int = 0,
) -> None:
    """Diffuse one input dimension's gradients in one KAN-like layer."""
    layer = model.layers[layer_index]
    if hasattr(layer, "spline_weight"):
        parameters = (layer.spline_weight,)
    elif hasattr(layer, "proto_val") and hasattr(layer, "proto_der"):
        parameters = (layer.proto_val, layer.proto_der)
    else:
        raise TypeError(
            "layer must expose spline_weight or both proto_val and proto_der"
        )
    if not 0 <= input_index < parameters[0].shape[1]:
        raise IndexError("input_index is out of range")

    for parameter in parameters:
        if parameter.grad is None:
            continue
        selected = parameter.grad[:, input_index, :]
        diffusion = implicit_diffusion_matrix(
            selected.shape[-1],
            strength,
            device=selected.device,
            dtype=selected.dtype,
        )
        selected.copy_(selected @ diffusion)


def protokan_hermite_loss(
    model: torch.nn.Module,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Penalize disagreement between adjacent value secants and derivatives."""
    if eps <= 0:
        raise ValueError("eps must be positive")
    losses = []
    for layer in model.layers:
        positions, order = layer.proto_pos.sort()
        values = layer.proto_val[..., order]
        derivatives = layer.proto_der[..., order]
        spacing = (positions[1:] - positions[:-1]).clamp_min(eps)
        secants = (values[..., 1:] - values[..., :-1]) / spacing
        mean_derivatives = 0.5 * (
            derivatives[..., 1:] + derivatives[..., :-1]
        )
        losses.append((secants - mean_derivatives).square().mean())
    if not losses:
        raise ValueError("model must contain at least one layer")
    return torch.stack(losses).sum()


def protokan_hermite_penalty(
    model: torch.nn.Module,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Alias for :func:`protokan_hermite_loss` for training objectives."""
    return protokan_hermite_loss(model, eps=eps)
