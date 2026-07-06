"""kanrf — KAN-based differentiable world models for model-based control.

Core library providing:
- KAN, KANLayer: Kolmogorov-Arnold Network with B-spline edge functions
- bspline_basis: vectorized Cox-de Boor recursion
- compute_uncertainty: B-spline activation density as epistemic uncertainty
- p_spline_penalty: control-point curvature regularization (MOPS)
- jacobian_loss, true_jacobian: output-space Jacobian matching (CWS)
"""
from kanrf._bspline import bspline_basis, bspline_derivative, extend_grid
from kanrf._layer import KANLayer
from kanrf._network import KAN
from kanrf._protokan import ProtoKAN, ProtoKANLayer
from kanrf._uncertainty import compute_uncertainty, compute_per_step_uncertainty
from kanrf._regularization import p_spline_penalty, true_jacobian, jacobian_loss

__all__ = [
    'KAN',
    'KANLayer',
    'ProtoKAN',
    'ProtoKANLayer',
    'bspline_basis',
    'bspline_derivative',
    'extend_grid',
    'compute_uncertainty',
    'compute_per_step_uncertainty',
    'p_spline_penalty',
    'true_jacobian',
    'jacobian_loss',
]
