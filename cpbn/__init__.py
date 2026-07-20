"""Cognitive Pullback Bellman Network (CPBN).

The package keeps cognition and decision training separate while coupling
their forward computations through a differentiable dynamics operator.
"""

from cpbn.acrobot import (
    GOAL,
    SOURCE_FACTOR,
    OracleAcrobotDynamics,
    random_states,
    reset_down_states,
    task_reward,
    tip_height,
)
from cpbn.bellman import (
    ImplicitBellmanAction,
    bellman_return,
    grid_best_action,
)
from cpbn.networks import ValueNetwork

__all__ = [
    "GOAL",
    "SOURCE_FACTOR",
    "ImplicitBellmanAction",
    "OracleAcrobotDynamics",
    "ValueNetwork",
    "bellman_return",
    "grid_best_action",
    "random_states",
    "reset_down_states",
    "task_reward",
    "tip_height",
]
