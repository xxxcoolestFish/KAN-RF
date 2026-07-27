import torch

from kanrf import (
    EffectEncoder,
    MLPDynamics,
    ProtoKANDynamics,
    TaskEffectValue,
    controllability_loss,
    effect_action_jacobian,
    effect_covariance_loss,
)


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
