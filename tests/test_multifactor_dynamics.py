import torch

from physics_transfer.multifactor_data import sample_multifactor_batch


def test_multifactor_batch_is_finite():
    batch = sample_multifactor_batch(
        8, 4,
        ((7.35, 0.0, 0.8, 0.8), (14.7, 0.08, 1.2, 1.2)),
        torch.Generator().manual_seed(0),
    )
    assert batch["history"].shape == (8, 28)
    assert batch["next_state"].shape == (8, 6)
    assert torch.isfinite(batch["next_state"]).all()
