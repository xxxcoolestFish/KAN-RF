from types import SimpleNamespace

import pytest
import torch

from kanrf import KAN, ProtoKAN
from kanrf.spline_coupling import (
    diffuse_input_dimension_gradients_,
    diffuse_kan_gradients_,
    diffuse_model_gradients_,
    diffuse_protokan_gradients_,
    implicit_diffusion_matrix,
    path_graph_laplacian,
    protokan_hermite_loss,
    protokan_hermite_penalty,
)


def test_path_laplacian_and_implicit_diffusion_are_psd():
    laplacian = path_graph_laplacian(7, dtype=torch.float64)
    diffusion = implicit_diffusion_matrix(7, 0.4, dtype=torch.float64)

    assert torch.allclose(laplacian, laplacian.T)
    assert torch.linalg.eigvalsh(laplacian).min() >= -1e-12
    assert torch.allclose(diffusion, diffusion.T)
    assert torch.linalg.eigvalsh(diffusion).min() > 0


def test_zero_diffusion_is_identity_and_constant_gradients_are_preserved():
    identity = implicit_diffusion_matrix(6, 0.0)
    diffusion = implicit_diffusion_matrix(6, 0.7)

    assert torch.equal(identity, torch.eye(6))
    assert torch.allclose(diffusion @ torch.ones(6), torch.ones(6), atol=1e-6)


def test_local_gradient_pulse_spreads_over_path():
    diffusion = implicit_diffusion_matrix(7, 0.5)
    pulse = torch.zeros(7)
    pulse[3] = 1.0
    spread = pulse @ diffusion

    assert (spread > 0).all()
    assert spread[3] == spread.max()
    assert torch.allclose(spread.sum(), torch.tensor(1.0), atol=1e-6)
    assert torch.allclose(spread, spread.flip(0), atol=1e-6)


@pytest.mark.parametrize(
    ("model", "diffuse", "parameter_names"),
    [
        (KAN([2, 1], grid_size=4), diffuse_kan_gradients_, ("spline_weight",)),
        (
            ProtoKAN([2, 1], n_prototypes=6),
            diffuse_protokan_gradients_,
            ("proto_val", "proto_der"),
        ),
    ],
)
def test_model_gradient_diffusion_is_in_place(model, diffuse, parameter_names):
    grad_objects = {}
    for layer in model.layers:
        for name in parameter_names:
            parameter = getattr(layer, name)
            parameter.grad = torch.zeros_like(parameter)
            parameter.grad[..., parameter.shape[-1] // 2] = 1.0
            grad_objects[id(parameter)] = parameter.grad

    diffuse(model, tau=0.5)

    for layer in model.layers:
        for name in parameter_names:
            parameter = getattr(layer, name)
            assert parameter.grad is grad_objects[id(parameter)]
            assert (parameter.grad > 0).all()
            assert torch.isfinite(parameter.grad).all()


def test_protokan_hermite_loss_is_zero_for_a_line_and_has_finite_gradients():
    layer = ProtoKAN([1, 1], n_prototypes=5).layers[0]
    with torch.no_grad():
        layer.proto_pos.copy_(torch.tensor([0.5, -1.0, 1.0, -0.5, 0.0]))
        layer.proto_val.copy_(2.0 * layer.proto_pos + 3.0)
        layer.proto_der.fill_(2.0)
    model = SimpleNamespace(layers=[layer])

    loss = protokan_hermite_loss(model)
    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)

    with torch.no_grad():
        layer.proto_val[..., 2] += 0.25
    loss = protokan_hermite_loss(model)
    loss.backward()

    assert loss > 0
    for parameter in (layer.proto_pos, layer.proto_val, layer.proto_der):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_public_training_api_dispatches_and_penalty_aliases():
    model = ProtoKAN([1, 1], n_prototypes=4)
    model.layers[0].proto_val.grad = torch.ones_like(model.layers[0].proto_val)
    model.layers[0].proto_der.grad = torch.ones_like(model.layers[0].proto_der)

    diffuse_model_gradients_(model, strength=0.3)

    assert torch.allclose(model.layers[0].proto_val.grad, torch.ones_like(
        model.layers[0].proto_val
    ))
    assert torch.allclose(
        protokan_hermite_penalty(model),
        protokan_hermite_loss(model),
    )


@pytest.mark.parametrize(
    ("model", "parameter_names"),
    [
        (KAN([3, 2], grid_size=4), ("spline_weight",)),
        (ProtoKAN([3, 2], n_prototypes=6), ("proto_val", "proto_der")),
    ],
)
def test_input_dimension_diffusion_changes_only_selected_gradients(
    model, parameter_names
):
    layer = model.layers[0]
    original_gradients = {}
    for name in parameter_names:
        parameter = getattr(layer, name)
        parameter.grad = torch.randn_like(parameter)
        parameter.grad[:, 1, :].zero_()
        parameter.grad[:, 1, parameter.shape[-1] // 2] = 1.0
        original_gradients[name] = parameter.grad.clone()

    diffuse_input_dimension_gradients_(
        model, input_index=1, strength=0.5
    )

    for name in parameter_names:
        gradient = getattr(layer, name).grad
        assert (gradient[:, 1, :] > 0).all()
        assert torch.equal(
            gradient[:, (0, 2), :],
            original_gradients[name][:, (0, 2), :],
        )


def test_sgd_parameter_update_preserves_diffusion_distance_profile():
    model = KAN([2, 1], grid_size=5)
    parameter = model.layers[0].spline_weight
    parameter.grad = torch.zeros_like(parameter)
    center = parameter.shape[-1] // 2
    parameter.grad[:, 1, center] = 1.0
    before = parameter.detach().clone()

    diffuse_input_dimension_gradients_(
        model, input_index=1, strength=0.5
    )
    expected_update = 0.1 * parameter.grad.detach().clone()
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    optimizer.step()

    actual_update = before - parameter.detach()
    assert torch.allclose(actual_update, expected_update, atol=1e-7)
    profile = actual_update[0, 1].abs()
    assert profile[center] == profile.max()
    assert profile[center] > profile[center - 1] > profile[0]
