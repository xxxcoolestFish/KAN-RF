import torch

from kanrf.cognition_modulated_actor import (
    CognitionModulatedActor,
    LowRankCognitiveLinear,
)


def test_zero_cognition_is_exactly_the_shared_actor() -> None:
    torch.manual_seed(10)
    actor = CognitionModulatedActor(
        state_dim=6,
        action_dim=2,
        cognition_dim=3,
        hidden_dims=(8, 7),
        rank=2,
    )
    states = torch.randn(5, 6)
    cognition = torch.zeros(5, 3)
    torch.testing.assert_close(
        actor(states, cognition),
        actor.base_forward(states),
        rtol=0.0,
        atol=0.0,
    )


def test_actor_stops_cognition_gradient_but_trains_transport() -> None:
    torch.manual_seed(11)
    actor = CognitionModulatedActor(
        state_dim=4,
        action_dim=2,
        cognition_dim=2,
        hidden_dims=(6,),
        rank=2,
    )
    states = torch.randn(5, 4)
    cognition = torch.randn(5, 2, requires_grad=True)
    actor(states, cognition).square().mean().backward()

    assert cognition.grad is None
    for layer in actor.layers:
        gradient = layer.cognition_to_gate.weight.grad
        assert gradient is not None
        assert torch.isfinite(gradient).all()
        assert float(gradient.abs().sum()) > 0.0


def test_cognition_changes_actions_only_through_low_rank_update() -> None:
    torch.manual_seed(12)
    actor = CognitionModulatedActor(
        state_dim=5,
        action_dim=2,
        cognition_dim=2,
        hidden_dims=(7,),
        rank=2,
    )
    with torch.no_grad():
        for layer in actor.layers:
            layer.cognition_to_gate.weight.fill_(0.5)

    states = torch.randn(6, 5)
    cognition_a = torch.zeros(6, 2)
    cognition_b = torch.ones(6, 2)
    assert not torch.allclose(
        actor(states, cognition_a),
        actor(states, cognition_b),
    )

    for layer in actor.layers:
        matrix = layer.adaptation_matrix(torch.ones(2))
        assert int(torch.linalg.matrix_rank(matrix)) <= layer.rank


def test_single_cognition_vector_broadcasts_over_state_batch() -> None:
    torch.manual_seed(13)
    layer = LowRankCognitiveLinear(4, 3, cognition_dim=2, rank=2)
    with torch.no_grad():
        layer.cognition_to_gate.weight.normal_()
    inputs = torch.randn(5, 4)
    cognition = torch.randn(2)
    expected = layer(inputs, cognition.expand(5, -1))
    torch.testing.assert_close(layer(inputs, cognition), expected)
