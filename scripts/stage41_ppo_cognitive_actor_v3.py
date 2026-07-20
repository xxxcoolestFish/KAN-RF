"""Fixed-start PPO with absolute height shaping for Acrobot swing-up."""

from __future__ import annotations

from scripts import stage41_ppo_cognitive_actor_v2 as base


def absolute_height_reward(state, next_state, action, goal):
    # The terminal event is unchanged. Absolute height supplies a smoother
    # signal than pure potential-difference shaping for the underactuated
    # swing-up phase.
    height = base.tip_height(next_state)
    success = (height >= 1.0).float()
    return 0.25 * height + 5.0 * success - 0.005 * action.square().sum(dim=-1)


base.dense_reward = absolute_height_reward
base.main()
