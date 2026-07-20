"""KAN and ProtoKAN primitives retained for the cognitive dynamics model."""

from kanrf._bspline import bspline_basis, bspline_derivative, extend_grid
from kanrf._layer import KANLayer
from kanrf._network import KAN
from kanrf._protokan import ProtoKAN, ProtoKANLayer
from kanrf._regularization import jacobian_loss, p_spline_penalty, true_jacobian
from kanrf._uncertainty import compute_per_step_uncertainty, compute_uncertainty

__all__ = [
    "KAN",
    "KANLayer",
    "ProtoKAN",
    "ProtoKANLayer",
    "bspline_basis",
    "bspline_derivative",
    "compute_per_step_uncertainty",
    "compute_uncertainty",
    "extend_grid",
    "jacobian_loss",
    "p_spline_penalty",
    "true_jacobian",
]
