import numpy as np

from kanrf.control_equivalence_adapter import (
    normalized_task_metric,
    solve_local_control_equivalence,
)


def test_solver_recovers_linear_effect_with_minimum_change():
    jacobian = np.array([[2.0, 0.0], [0.0, 0.5]])
    reference = np.array([0.2, -0.1])
    target_effect = jacobian @ reference
    desired_effect = target_effect + np.array([0.4, -0.1])
    solution = solve_local_control_equivalence(
        reference,
        desired_effect,
        target_effect,
        jacobian,
        np.eye(2),
        -np.ones(2),
        np.ones(2),
        regularization=1e-8,
        trust_radius=10.0,
    )
    np.testing.assert_allclose(
        jacobian @ solution.action, desired_effect, atol=1e-5
    )


def test_task_metric_is_positive_definite_with_floor():
    metric = normalized_task_metric(
        np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]),
        dimension=3,
        isotropic_floor=0.02,
    )
    assert np.all(np.linalg.eigvalsh(metric) > 0.0)
    assert metric[0, 0] > metric[2, 2]
