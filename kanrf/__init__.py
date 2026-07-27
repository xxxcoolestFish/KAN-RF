"""Core KAN primitives and control-equivalence utilities."""

from kanrf._bspline import bspline_basis, bspline_derivative, extend_grid
from kanrf._layer import KANLayer
from kanrf._network import KAN
from kanrf._protokan import ProtoKAN, ProtoKANLayer
from kanrf._regularization import jacobian_loss, p_spline_penalty, true_jacobian
from kanrf._uncertainty import compute_per_step_uncertainty, compute_uncertainty
from kanrf.control_equivalence_adapter import (
    ControlEquivalenceSolution,
    normalized_task_metric,
    solve_local_control_equivalence,
    weighted_norm,
)
from kanrf.function_modulated_dynamics import ActionModulatedProtoKAN

__all__ = [
    "KAN",
    "KANLayer",
    "ActionModulatedProtoKAN",
    "ProtoKAN",
    "ProtoKANLayer",
    "bspline_basis",
    "bspline_derivative",
    "compute_per_step_uncertainty",
    "compute_uncertainty",
    "ControlEquivalenceSolution",
    "extend_grid",
    "jacobian_loss",
    "normalized_task_metric",
    "p_spline_penalty",
    "solve_local_control_equivalence",
    "true_jacobian",
    "weighted_norm",
]
