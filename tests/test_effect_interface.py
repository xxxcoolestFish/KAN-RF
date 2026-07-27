import torch

from kanrf import (
    EffectEncoder,
    MLPDynamics,
    ProtoKANDynamics,
    TaskEffectValue,
    control_equivalence_loss,
    controllable_gradient_loss,
    controllability_loss,
    effect_action_jacobian,
    effect_covariance_loss,
)


def test_controllable_gradient_loss_only_penalizes_reachable_direction():
    exact = torch.zeros(2, 3)
    displacement = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    orthogonal_error = torch.tensor(
        [[0.0, 2.0, 0.0], [0.0, 0.0, 3.0]],
        requires_grad=True,
    )
    aligned_error = torch.tensor(
        [[2.0, 0.0, 0.0], [0.0, 3.0, 0.0]],
        requires_grad=True,
    )
    orthogonal_loss = controllable_gradient_loss(
        orthogonal_error,
        exact,
        displacement,
    )
    aligned_loss = controllable_gradient_loss(
        aligned_error,
        exact,
        displacement,
    )
    assert torch.allclose(orthogonal_loss, torch.zeros(()))
    assert aligned_loss > 0
    aligned_loss.backward()
    assert aligned_error.grad is not None


def test_control_equivalence_loss_detects_wrong_best_candidate():
    exact = torch.tensor([[3.0, 2.0, 1.0], [0.0, 0.5, -1.0]])
    correct = exact.clone().requires_grad_(True)
    wrong = torch.tensor(
        [[1.0, 2.0, 3.0], [0.5, 0.0, -1.0]],
        requires_grad=True,
    )
    correct_stats = control_equivalence_loss(correct, exact)
    wrong_stats = control_equivalence_loss(wrong, exact)
    assert correct_stats.advantage_loss < wrong_stats.advantage_loss
    assert correct_stats.margin_loss < wrong_stats.margin_loss
    assert correct_stats.top1_agreement == 1.0
    assert wrong_stats.top1_agreement == 0.0
    (wrong_stats.advantage_loss + wrong_stats.margin_loss).backward()
    assert wrong.grad is not None


def test_task_effect_value_normalization_and_shapes():
    observations = torch.randn(32, 6)
    values = torch.randn(32) * 3.0 + 7.0
    model = TaskEffectValue(obs_dim=6, effect_dim=3, hidden_dim=16)
    model.set_normalization(observations, values)
    effects, predictions = model(observations)
    assert effects.shape == (32, 3)
    assert predictions.shape == (32,)
    assert torch.isfinite(effects).all()
    assert torch.isfinite(predictions).all()
    assert torch.allclose(model.obs_mean, observations.mean(dim=0))
    assert torch.allclose(model.value_mean, values.mean())


def test_effect_jacobian_has_expected_shape_and_gradients():
    encoder = EffectEncoder(obs_dim=23, effect_dim=4, hidden_dim=32)
    dynamics = MLPDynamics(obs_dim=23, action_dim=7, hidden_dim=32)
    states = torch.randn(8, 23)
    actions = torch.randn(8, 7)
    jacobian = effect_action_jacobian(encoder, dynamics, states, actions)
    assert jacobian.shape == (8, 4, 7)
    stats = controllability_loss(jacobian)
    stats.loss.backward()
    assert torch.isfinite(stats.loss)
    assert stats.min_singular_value >= 0
    assert stats.condition_number >= 1


def test_effect_covariance_loss_penalizes_collapse():
    collapsed = torch.zeros(32, 4)
    spread = torch.randn(32, 4)
    assert effect_covariance_loss(collapsed) > effect_covariance_loss(spread)


def test_protokan_dynamics_shape():
    model = ProtoKANDynamics(
        obs_dim=23,
        action_dim=7,
        hidden_dim=16,
        n_prototypes=6,
    )
    next_states = model(torch.randn(3, 23), torch.randn(3, 7))
    assert next_states.shape == (3, 23)
    assert next_states.isfinite().all()
