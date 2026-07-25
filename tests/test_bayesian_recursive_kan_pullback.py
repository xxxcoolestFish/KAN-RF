import torch

from cpbn.bayesian_recursive_kan_pullback import (
    BayesianRecursiveChartScaledEstimator,
    posterior_risk_pullback,
)
from cpbn.chart_scaled_kan_atlas import (
    ChartScaledLocalKANDictionary,
    ChartScaledLocalKANContext,
)
from tests.test_chart_scaled_kan_atlas import make_data


def test_posterior_mean_matches_recursive_sufficient_statistics():
    atlas = ChartScaledLocalKANDictionary(
        torch.zeros(1, 4), torch.tensor([[3.0, 2.0]]),
    )
    prior = ChartScaledLocalKANContext.neutral(atlas)
    state, action, next_state, _ = make_data(atlas, 256)
    estimator = BayesianRecursiveChartScaledEstimator(atlas, prior, ridge=0.1)
    estimator.update(state, action, next_state)
    assert torch.allclose(
        estimator.posterior().mean.coefficients,
        estimator.context().coefficients,
    )


def test_gain_uncertainty_decreases_with_more_evidence():
    atlas = ChartScaledLocalKANDictionary(
        torch.zeros(1, 4), torch.tensor([[3.0, 2.0]]),
    )
    prior = ChartScaledLocalKANContext.neutral(atlas)
    state, action, next_state, _ = make_data(atlas, 1024)
    estimator = BayesianRecursiveChartScaledEstimator(atlas, prior, ridge=0.1)
    estimator.update(state[:128], action[:128], next_state[:128])
    early = estimator.posterior().gain_uncertainty(
        atlas, state[:64],
    ).diagonal(dim1=-2, dim2=-1).sum(-1).mean()
    estimator.update(state[128:], action[128:], next_state[128:])
    late = estimator.posterior().gain_uncertainty(
        atlas, state[:64],
    ).diagonal(dim1=-2, dim2=-1).sum(-1).mean()
    assert late < early


def test_directional_risk_anchors_only_uncertain_action_direction():
    gain = torch.eye(2).unsqueeze(0)
    target = torch.tensor([[2.0, 3.0]])
    source = torch.zeros(1, 2)
    risk = torch.diag_embed(torch.tensor([[1e-6, 1e6]]))
    action = posterior_risk_pullback(gain, target, source, risk)
    assert torch.allclose(action[:, 0], torch.tensor([2.0]), atol=1e-4)
    assert action[:, 1].abs().max() < 1e-4


def test_effect_metric_prioritizes_task_relevant_output_direction():
    gain = torch.ones(1, 2, 1)
    target = torch.tensor([[1.0, 2.0]])
    source = torch.zeros(1, 1)
    risk = torch.full((1, 1, 1), 1e-6)
    metric = torch.diag_embed(torch.tensor([[1.0, 100.0]]))
    action = posterior_risk_pullback(
        gain, target, source, risk, effect_metric=metric,
    )
    assert float(action) > 1.9
