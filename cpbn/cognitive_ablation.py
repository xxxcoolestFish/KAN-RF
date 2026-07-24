"""Leak-free interventions on inferred cognitive coordinates."""

from __future__ import annotations

import torch


def intervene_on_coordinates(coordinates, condition: str):
    """Apply a deterministic cognition intervention for causal ablations."""
    if condition in {"inferred", "frozen"}:
        return coordinates
    if condition == "zero":
        return torch.zeros_like(coordinates)
    if condition == "shuffled":
        if coordinates.numel() < 2:
            raise ValueError(
                "Shuffling requires at least two cognitive coordinates.",
            )
        return torch.roll(coordinates, shifts=1, dims=0)
    raise ValueError(f"Unknown cognition condition: {condition}")
