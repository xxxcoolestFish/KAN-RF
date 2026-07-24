import torch

from cpbn.generic_affine_kan import LearnedMLPDictionary


def test_learned_mlp_dictionary_preserves_affine_context_interface():
    dictionary = LearnedMLPDictionary(
        torch.ones(5),
        torch.ones(2),
        feature_dim=17,
        hidden_dim=8,
    )
    state = torch.randn(7, 5)
    action = torch.randn(7, 2)

    features = dictionary(state)
    design = dictionary.context_features(state, action)

    assert features.shape == (7, 17)
    assert design.shape == (7, 51)
    torch.testing.assert_close(features[:, 0], torch.ones(7))
    torch.testing.assert_close(design[:, :17], features)
