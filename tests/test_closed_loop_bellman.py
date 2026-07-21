import torch
from torch import nn

from cpbn import OracleAcrobotDynamics
from cpbn.closed_loop_bellman import ClosedLoopBellmanActor, shift_corridor
from scripts.validate_target_online_adaptation import HEAVY_INERTIA_FACTOR


def nominal_state(batch):
    state = torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    return state.repeat(batch, 1)


def nominal_corridor(batch, horizon):
    corridor = nominal_state(batch).unsqueeze(1).repeat(1, horizon, 1)
    corridor[:, :, 1] = torch.linspace(0.01, 0.08, horizon)
    corridor[:, :, 0] = torch.sqrt(1.0 - corridor[:, :, 1].square())
    return corridor


def test_closed_loop_actor_uses_full_grid_and_is_trainable():
    torch.manual_seed(3)
    actor = ClosedLoopBellmanActor(
        hidden_dim=8, corridor_horizon=4,
        action_bins=5, backup_depth=2,
    )
    cognition = OracleAcrobotDynamics()
    state = nominal_state(3)
    corridor = nominal_corridor(3, 4)
    logits = actor.action_logits(state, corridor, cognition)
    assert logits.shape == (3, 5)
    assert torch.isfinite(logits).all()
    action, log_prob = actor.sample(
        state, corridor, cognition, deterministic=True,
    )
    assert action.shape == (3, 1)
    assert torch.isin(action.squeeze(-1), actor.action_grid).all()
    evaluated, entropy = actor.evaluate(
        state, corridor, cognition, action,
    )
    assert torch.allclose(log_prob, evaluated)
    assert torch.isfinite(entropy).all()
    loss = -logits[:, 0].mean()
    loss.backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in actor.parameters()
    )


def test_backup_rebranches_at_each_imagined_state():
    class CountingCognition(nn.Module):
        def __init__(self):
            super().__init__()
            self.batch_sizes = []

        def forward(self, state, action):
            self.batch_sizes.append(state.shape[0])
            return state

    actor = ClosedLoopBellmanActor(
        hidden_dim=8, corridor_horizon=3,
        action_bins=3, backup_depth=2,
    )
    cognition = CountingCognition()
    actor.action_logits(
        nominal_state(2), nominal_corridor(2, 3), cognition,
    )
    assert cognition.batch_sizes == [6, 18]


def test_cognition_swap_changes_bellman_action_scores():
    torch.manual_seed(7)
    actor = ClosedLoopBellmanActor(
        hidden_dim=8, corridor_horizon=4,
        action_bins=5, backup_depth=2,
    )
    state = nominal_state(4)
    state[:, 4:] = torch.tensor([0.7, -0.4])
    corridor = nominal_corridor(4, 4)
    correct = actor.action_logits(state, corridor, OracleAcrobotDynamics())
    wrong = actor.action_logits(
        state, corridor, OracleAcrobotDynamics(HEAVY_INERTIA_FACTOR),
    )
    assert float((correct - wrong).abs().max()) > 1e-5


def test_shift_corridor_repeats_terminal_state():
    corridor = torch.arange(2 * 4 * 6).view(2, 4, 6)
    shifted = shift_corridor(corridor)
    assert torch.equal(shifted[:, :-1], corridor[:, 1:])
    assert torch.equal(shifted[:, -1], corridor[:, -1])


def test_macro_backup_accumulates_before_rebranching():
    class CountingCognition(nn.Module):
        def __init__(self):
            super().__init__()
            self.batch_sizes = []

        def forward(self, state, action):
            self.batch_sizes.append(state.shape[0])
            return state

    actor = ClosedLoopBellmanActor(
        hidden_dim=8,
        corridor_horizon=3,
        action_bins=3,
        backup_depth=2,
        macro_steps=2,
    )
    cognition = CountingCognition()
    actor.action_logits(
        nominal_state(2), nominal_corridor(2, 3), cognition,
    )
    assert cognition.batch_sizes == [6, 6, 18, 18]


def test_cached_depth_two_tree_is_exact_and_differentiable():
    torch.manual_seed(11)
    actor = ClosedLoopBellmanActor(
        hidden_dim=8,
        corridor_horizon=4,
        action_bins=3,
        backup_depth=2,
        macro_steps=2,
    )
    cognition = OracleAcrobotDynamics()
    state = nominal_state(3)
    state[:, 4:] = torch.tensor([0.4, -0.2])
    corridor = nominal_corridor(3, 4)
    direct = actor.action_logits(state, corridor, cognition)
    tree = actor.precompute_depth_two_tree(state, corridor, cognition)
    cached = actor.logits_from_depth_two_tree(tree)
    assert torch.allclose(direct, cached, atol=1e-6, rtol=1e-6)
    cached.square().mean().backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in actor.parameters()
    )
