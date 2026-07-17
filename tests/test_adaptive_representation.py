import torch

from physics_transfer.adaptive import AdaptiveCognitivePredictor
from physics_transfer.data import sample_acrobot_batch


def test_adaptive_representation_shapes():
    batch = sample_acrobot_batch(4, 3, (7.35, 14.7), torch.Generator().manual_seed(0))
    model = AdaptiveCognitivePredictor(
        6, 1, 21, token_count=5, token_dim=4, hidden_dim=8, n_prototypes=4
    )
    output = model(batch["history"], batch["state"], batch["action"])
    assert output["tokens"].shape == (4, 5, 4)
    assert output["gates"].shape == (4, 5)
    assert output["next_state"].shape == (4, 6)
