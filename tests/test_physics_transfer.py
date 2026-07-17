import torch

from physics_transfer import CognitiveDecisionSystem


def test_physics_transfer_shapes():
    model = CognitiveDecisionSystem(
        state_dim=6, action_dim=1, history_dim=14,
        physics_dim=2, hidden_dim=8, n_prototypes=4,
    )
    output = model(torch.randn(3, 14), torch.randn(3, 6), torch.randn(3, 1))
    assert output["physics"].shape == (3, 2)
    assert output["receiver"].shape == (3, 2)
    assert output["next_state"].shape == (3, 6)
    assert output["action"].shape == (3, 1)
