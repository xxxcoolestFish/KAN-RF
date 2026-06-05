import torch.nn as nn
from kanrf._layer import KANLayer


class KAN(nn.Module):
    """Multi-layer KAN network."""

    def __init__(self, layer_dims: list, grid_size: int = 5,
                 spline_order: int = 3, grid_range: float = 1.0):
        super().__init__()
        self.layers = nn.ModuleList([
            KANLayer(layer_dims[i], layer_dims[i + 1],
                     grid_size=grid_size, spline_order=spline_order,
                     grid_range=grid_range)
            for i in range(len(layer_dims) - 1)
        ])

    def forward(self, x, return_activations: bool = False):
        if return_activations:
            B_list, E_list = [], []
            for layer in self.layers:
                x, B, E = layer(x, return_activations=True)
                B_list.append(B)
                E_list.append(E)
            return x, B_list, E_list
        for layer in self.layers:
            x = layer(x)
        return x
