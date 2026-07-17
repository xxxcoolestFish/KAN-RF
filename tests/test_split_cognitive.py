import torch

from physics_transfer.multifactor_data import sample_multifactor_batch
from physics_transfer.split_cognitive import SplitCognitivePredictor


def test_split_cognitive_shapes():
    batch = sample_multifactor_batch(
        4, 8,
        ((7.35, 0.0, 0.8, 0.8), (14.7, 0.08, 1.2, 1.2)),
        torch.Generator().manual_seed(0),
    )
    model = SplitCognitivePredictor(6, 1, 8, token_count=8, token_dim=8)
    output = model(batch["history"], batch["state"], batch["action"])
    assert output["physics_pooled"].shape == (4, 8)
    assert output["state_memory"].shape == (4, 16)
    assert output["next_state"].shape == (4, 6)
