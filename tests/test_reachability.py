"""Minimal checks for the non-action coarse planner interface."""

import torch

from cpbn import OracleAcrobotDynamics, reset_down_states
from cpbn.reachability import CoarseReachabilityPlanner, state_distance


def test_state_distance_is_zero_on_identical_state():
    state = reset_down_states(4)
    assert torch.allclose(state_distance(state, state), torch.zeros(4))


def test_planner_returns_waypoints_not_actions():
    planner = CoarseReachabilityPlanner(
        OracleAcrobotDynamics(),
        anchor_count=128,
        samples_per_anchor=4,
        macro_steps=24,
        action_segments=1,
        seed=3,
    )
    reference, route_distance = planner.query(reset_down_states(3))
    assert reference.shape == (3, 6)
    assert route_distance.shape == (3,)
    assert reference.isfinite().all()
