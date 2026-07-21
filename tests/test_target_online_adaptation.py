"""Checks for real-feedback target policy adaptation rollouts."""

from argparse import Namespace

import torch

from cpbn import OracleAcrobotDynamics
from cpbn.corridor_policy import CorridorCritic, DirectCorridorActor
from scripts.validate_target_online_adaptation import collect_feedback_rollout


def test_feedback_adaptation_rollout_has_finite_training_targets():
    torch.manual_seed(13)
    center = torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    reference = center.view(1, 6).expand(20, -1).clone()
    actor = DirectCorridorActor(hidden_dim=8)
    critic = CorridorCritic(hidden_dim=8)
    args = Namespace(
        num_envs=4,
        initial_noise=0.01,
        rollout_horizon=3,
        corridor_horizon=4,
        phase_backtrack=1,
        phase_advance=2,
        progress_clip=0.25,
        corridor_radius=0.12,
        progress_reward=1.0,
        phase_progress_reward=0.04,
        inside_reward=0.08,
        stagnation_penalty=0.01,
        success_reward=3.0,
        action_penalty=0.002,
        gamma=0.99,
        gae_lambda=0.95,
    )
    rollout, diagnostics = collect_feedback_rollout(
        actor, critic, OracleAcrobotDynamics(), reference, args, seed=17,
    )
    assert rollout.state.shape == (12, 6)
    assert rollout.action.shape == (12, 1)
    assert torch.isfinite(rollout.advantage).all()
    assert torch.isfinite(rollout.returns).all()
    assert diagnostics["return_target_std"] >= 0.0
