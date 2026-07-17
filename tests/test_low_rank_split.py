import torch

from physics_transfer.low_rank_split import LowRankSplitCognitivePredictor
from physics_transfer.multifactor_data import sample_multifactor_batch


def test_low_rank_residual_shapes_and_finiteness():
    batch = sample_multifactor_batch(
        4, 8,
        ((7.35, 0.0, 0.8, 0.8), (14.7, 0.08, 1.2, 1.2)),
        torch.Generator().manual_seed(0),
    )
    model = LowRankSplitCognitivePredictor(
        physics_rank=4, residual_scale=0.1, state_dim=6,
        action_dim=1, history_steps=8,
    )
    output = model(batch["history"], batch["state"], batch["action"])
    assert output["physics_coefficients"].shape == (4, 4)
    assert output["physics_basis"].shape == (4, 6, 4)
    assert torch.isfinite(output["next_state"]).all()
    assert torch.isfinite(output["basis_gram_error"]).all()
