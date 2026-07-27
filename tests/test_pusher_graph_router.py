import gymnasium as gym
import numpy as np

from kanrf.pusher_graph_router import PusherOracleGraphRouter


def test_graph_router_restores_state_and_predicts_executed_edge():
    env = gym.make("Pusher-v5")
    observation, _ = env.reset(seed=1811)
    qpos = env.unwrapped.data.qpos.copy()
    qvel = env.unwrapped.data.qvel.copy()
    router = PusherOracleGraphRouter(
        depth=2,
        branching=8,
        beam_width=4,
        seed=1811,
    )
    plan = router.plan(env)
    assert plan.action.shape == env.action_space.shape
    assert plan.sequence.shape == (2, env.action_space.shape[0])
    assert len(plan.layers) == 2
    assert np.isfinite(plan.predicted_return)
    assert np.allclose(env.unwrapped.data.qpos, qpos, atol=1e-12)
    assert np.allclose(env.unwrapped.data.qvel, qvel, atol=1e-12)
    next_observation, reward, _, _, _ = env.step(plan.action)
    assert np.allclose(
        next_observation,
        plan.predicted_first_observation,
        atol=1e-7,
    )
    assert np.isclose(reward, plan.predicted_first_reward, atol=1e-9)
    assert not np.allclose(observation, next_observation)
    router.close()
    env.close()


def test_graph_router_actions_are_bounded_and_reports_merging():
    env = gym.make("Pusher-v5")
    env.reset(seed=1911)
    router = PusherOracleGraphRouter(
        depth=2,
        branching=8,
        beam_width=4,
        merge_radius=1e6,
        seed=1911,
    )
    plan = router.plan(env)
    assert (plan.sequence >= env.action_space.low).all()
    assert (plan.sequence <= env.action_space.high).all()
    assert all(layer.unique <= layer.expanded for layer in plan.layers)
    assert any(layer.merged > 0 for layer in plan.layers)
    router.close()
    env.close()
