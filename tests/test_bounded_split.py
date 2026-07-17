import torch

from physics_transfer.bounded_split import BoundedSplitCognitivePredictor
from physics_transfer.multifactor_data import sample_multifactor_batch


def test_bounded_residual():
    batch = sample_multifactor_batch(
        4, 8,
        ((7.35, 0.0, 0.8, 0.8), (14.7, 0.08, 1.2, 1.2)),
        torch.Generator().manual_seed(0),
    )
    model = BoundedSplitCognitivePredictor(
        residual_scale=0.1, state_dim=6, action_dim=1, history_steps=8
    )
    output = model(batch["history"], batch["state"], batch["action"])
    assert torch.linalg.vector_norm(output["physics_residual"], dim=-1).max() < 0.25
