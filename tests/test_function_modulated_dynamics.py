import torch

from kanrf import ActionModulatedProtoKAN


def test_zero_latent_is_exactly_the_shared_backbone() -> None:
    torch.manual_seed(4)
    model = ActionModulatedProtoKAN(
        input_dim=7,
        hidden_dim=5,
        output_dim=3,
        action_start=4,
        action_dim=3,
        n_prototypes=6,
    )
    inputs = torch.randn(8, 7)
    expected = model.backbone(inputs)
    actual = model(inputs, torch.zeros(8, 1))
    torch.testing.assert_close(actual, expected)


def test_latent_gradient_depends_on_action_function_modes() -> None:
    torch.manual_seed(5)
    model = ActionModulatedProtoKAN(
        input_dim=6,
        hidden_dim=4,
        output_dim=2,
        action_start=4,
        action_dim=2,
        n_prototypes=5,
    )
    inputs = torch.randn(7, 6)
    latent = torch.zeros(1, 1, requires_grad=True)
    model(inputs, latent).square().mean().backward()
    assert latent.grad is not None
    assert torch.isfinite(latent.grad).all()
    assert float(latent.grad.abs().sum()) > 0.0


def test_only_declared_action_slice_drives_mode_outputs() -> None:
    torch.manual_seed(6)
    model = ActionModulatedProtoKAN(
        input_dim=6,
        hidden_dim=4,
        output_dim=2,
        action_start=4,
        action_dim=2,
        n_prototypes=5,
    )
    inputs = torch.randn(7, 6)
    changed_state = inputs.clone()
    changed_state[:, :4] += 10.0
    torch.testing.assert_close(
        model.action_mode_outputs(inputs),
        model.action_mode_outputs(changed_state),
    )

