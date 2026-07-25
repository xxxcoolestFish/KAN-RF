import torch

from cpbn.chart_scaled_kan_atlas import (
    ChartScaledLocalKANDictionary,
    ChartScaledLocalKANContext,
    RecursiveChartScaledEstimator,
    distill_teacher_to_atlas,
    fit_chart_scaled_context,
)


def make_data(atlas, count=1024):
    state = 0.5 * torch.randn(count, 4)
    action = 3.0 * torch.randn(count, 2)
    drift = torch.stack((state[:, 0] + state[:, 2], -state[:, 1]), dim=-1)
    gain = torch.tensor([[2.0, 0.3], [-0.2, 1.5]])
    acceleration = drift + action @ gain.T
    next_state = state.clone()
    next_state[:, 2:] += 0.02 * acceleration
    return state, action, next_state, acceleration


def test_neutral_physics_is_invariant_to_chart_action_scales():
    centers = torch.tensor([[0.0, 0.0, 0.0, 0.0], [0.5, -0.5, 3.0, -3.0]])
    atlas = ChartScaledLocalKANDictionary(centers, torch.tensor([[1.0, 2.0], [8.0, 4.0]]))
    context = ChartScaledLocalKANContext.neutral(atlas)
    state = torch.randn(128, 4)
    action = torch.randn(128, 2)
    assert torch.allclose(context.acceleration(atlas, state, action), action, atol=1e-5)


def test_recursive_sufficient_statistics_equal_batch_fit():
    atlas = ChartScaledLocalKANDictionary(torch.zeros(1, 4), torch.tensor([[3.0, 2.0]]))
    state, action, next_state, _ = make_data(atlas)
    prior = ChartScaledLocalKANContext.neutral(atlas)
    batch = fit_chart_scaled_context(atlas, state, action, next_state, prior, ridge=0.1)
    recursive = RecursiveChartScaledEstimator(atlas, prior, ridge=0.1)
    for start in range(0, state.shape[0], 128):
        recursive.update(state[start:start + 128], action[start:start + 128], next_state[start:start + 128])
    assert torch.allclose(recursive.context().coefficients, batch.coefficients, atol=2e-4, rtol=2e-4)


def test_teacher_distillation_preserves_local_predictions():
    torch.manual_seed(331)
    atlas = ChartScaledLocalKANDictionary(torch.zeros(1, 4), torch.tensor([[3.0, 3.0]]))
    teacher = fit_chart_scaled_context(atlas, *make_data(atlas)[:3], ridge=1e-3)
    distilled = distill_teacher_to_atlas(atlas, atlas, teacher, samples_per_chart=2048)
    state, action, _, _ = make_data(atlas, 512)
    error = (
        distilled.acceleration(atlas, state, action)
        - teacher.acceleration(atlas, state, action)
    ).square().mean().sqrt()
    assert error < 0.1
