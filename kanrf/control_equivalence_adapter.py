"""Local control-equivalence operators for policy transfer.

The functions in this module contain no environment-specific assumptions.
They solve a regularized local inverse problem:

    B_target * delta_action ~= desired_effect - target_effect

under a positive-semidefinite task metric.  The intended use is to obtain the
smallest action correction that restores the task-relevant effect of a frozen
source policy after the dynamics change.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ControlEquivalenceSolution:
    action: np.ndarray
    correction: np.ndarray
    predicted_residual_norm: float
    condition_number: float
    saturated_fraction: float


def normalized_task_metric(
    gradients: np.ndarray,
    dimension: int,
    isotropic_floor: float = 0.01,
) -> np.ndarray:
    """Construct a scale-stable PSD metric from task-value gradients."""
    gradients = np.asarray(gradients, dtype=np.float64)
    if gradients.ndim == 1:
        gradients = gradients[None, :]
    if gradients.shape[1] != dimension:
        raise ValueError(
            f"expected gradient dimension {dimension}, got {gradients.shape}"
        )
    norms = np.linalg.norm(gradients, axis=1, keepdims=True)
    valid = norms[:, 0] > 1e-10
    if not np.any(valid):
        return np.eye(dimension, dtype=np.float64)
    directions = gradients[valid] / norms[valid]
    metric = directions.T @ directions / len(directions)
    trace = float(np.trace(metric))
    if trace > 1e-12:
        metric *= dimension / trace
    return metric + float(isotropic_floor) * np.eye(dimension)


def weighted_norm(vector: np.ndarray, metric: np.ndarray) -> float:
    vector = np.asarray(vector, dtype=np.float64)
    metric = np.asarray(metric, dtype=np.float64)
    value = float(vector @ metric @ vector)
    return float(np.sqrt(max(value, 0.0)))


def solve_local_control_equivalence(
    reference_action: np.ndarray,
    desired_effect: np.ndarray,
    target_effect: np.ndarray,
    target_action_jacobian: np.ndarray,
    task_metric: np.ndarray,
    action_low: np.ndarray,
    action_high: np.ndarray,
    regularization: float = 0.05,
    trust_radius: float = 0.75,
) -> ControlEquivalenceSolution:
    """Return a bounded minimum-change action that matches a desired effect."""
    reference_action = np.asarray(reference_action, dtype=np.float64)
    desired_effect = np.asarray(desired_effect, dtype=np.float64)
    target_effect = np.asarray(target_effect, dtype=np.float64)
    jacobian = np.asarray(target_action_jacobian, dtype=np.float64)
    metric = np.asarray(task_metric, dtype=np.float64)
    action_low = np.asarray(action_low, dtype=np.float64)
    action_high = np.asarray(action_high, dtype=np.float64)
    if jacobian.shape != (desired_effect.size, reference_action.size):
        raise ValueError("target_action_jacobian has an incompatible shape")
    residual = desired_effect - target_effect
    normal = jacobian.T @ metric @ jacobian
    scale = max(float(np.trace(normal)) / max(reference_action.size, 1), 1e-8)
    system = normal + float(regularization) * scale * np.eye(
        reference_action.size
    )
    right = jacobian.T @ metric @ residual
    try:
        correction = np.linalg.solve(system, right)
    except np.linalg.LinAlgError:
        correction = np.linalg.pinv(system, rcond=1e-7) @ right
    span = np.maximum(action_high - action_low, 1e-8)
    normalized = correction / span
    normalized_norm = float(np.linalg.norm(normalized))
    if trust_radius > 0.0 and normalized_norm > trust_radius:
        correction *= trust_radius / normalized_norm
    unclipped = reference_action + correction
    action = np.clip(unclipped, action_low, action_high)
    correction = action - reference_action
    predicted = residual - jacobian @ correction
    saturated = np.isclose(action, action_low, atol=1e-6) | np.isclose(
        action, action_high, atol=1e-6
    )
    return ControlEquivalenceSolution(
        action=action.astype(np.float32),
        correction=correction.astype(np.float32),
        predicted_residual_norm=weighted_norm(predicted, metric),
        condition_number=float(np.linalg.cond(system)),
        saturated_fraction=float(np.mean(saturated)),
    )
