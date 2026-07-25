"""Minimal regression tests for the retained KAN implementation."""

import torch

from kanrf import KANLayer, ProtoKAN, bspline_basis


def test_bspline_partition_of_unity():
    grid = torch.linspace(-1.0, 1.0, 6)
    x = torch.linspace(-1.0, 1.0, 50)
    basis = bspline_basis(x, grid, k=3)
    assert (basis >= -1e-8).all()
    assert torch.allclose(basis.sum(dim=-1), torch.ones(50), atol=1e-6)
    assert int((basis > 1e-8).sum(dim=-1).max()) <= 4


def test_kan_layer_has_finite_gradients():
    layer = KANLayer(in_dim=4, out_dim=2, grid_size=5, spline_order=3)
    output = layer(torch.randn(8, 4))
    assert output.shape == (8, 2)
    output.sum().backward()
    assert all(
        parameter.grad is not None and parameter.grad.isfinite().all()
        for parameter in layer.parameters()
    )


def test_protokan_forward_shape():
    model = ProtoKAN([7, 16, 6])
    output = model(torch.randn(5, 7))
    assert output.shape == (5, 6)
    assert output.isfinite().all()


def test_kan_spline_edges_do_not_mix_input_dimensions():
    layer = KANLayer(in_dim=2, out_dim=1, grid_size=5, spline_order=3)
    with torch.no_grad():
        layer.base_weight.zero_()
        layer.spline_weight.zero_()
        layer.spline_weight[0, 0].fill_(1.0)

    first = layer(torch.tensor([[0.0, -0.5]]))
    second = layer(torch.tensor([[0.0, 0.5]]))
    assert torch.allclose(first, second, atol=1e-7)
