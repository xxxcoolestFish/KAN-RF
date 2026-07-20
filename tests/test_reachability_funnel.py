"""Interface checks for empirical reachability regions."""

import torch

from cpbn.reachability_funnel import EmpiricalReachabilityFunnels


def test_normalized_funnel_distance():
    center = torch.zeros(2, 6)
    scale = torch.full((2, 6), 0.5)
    assert torch.allclose(
        EmpiricalReachabilityFunnels.normalized_distance(center, center, scale),
        torch.zeros(2),
    )
    boundary = center.clone()
    boundary[0] = 0.5
    assert torch.isclose(
        EmpiricalReachabilityFunnels.normalized_distance(
            boundary[:1], center[:1], scale[:1],
        )[0],
        torch.tensor(1.0),
    )


def test_inside_returns_batch_mask():
    center = torch.zeros(2, 6)
    scale = torch.ones(2, 6)
    state = torch.stack([torch.zeros(6), torch.full((6,), 2.0)])
    inside = EmpiricalReachabilityFunnels.inside(state, center, scale)
    assert inside.tolist() == [True, False]
