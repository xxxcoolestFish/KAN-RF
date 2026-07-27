import gymnasium as gym
import numpy as np

from kanrf.pusher_oracle import PusherOracleCEM


def test_oracle_rollout_restores_state_and_is_deterministic():
    env = gym.make("Pusher-v5")
    env.reset(seed=1811)
    qpos = env.unwrapped.data.qpos.copy()
    qvel = env.unwrapped.data.qvel.copy()
    planner = PusherOracleCEM(
        horizon=2,
        action_repeat=1,
        population=4,
        iterations=1,
        initial_std_scale=0.25,
        seed=1811,
    )
    sequences = np.zeros((4, 2, env.action_space.shape[0]), dtype=np.float32)
    first = planner.evaluate_sequences(qpos, qvel, sequences)
    second = planner.evaluate_sequences(qpos, qvel, sequences)
    assert np.allclose(first, second, atol=1e-9)
    assert np.allclose(planner.env.unwrapped.data.qpos, qpos, atol=1e-12)
    assert np.allclose(planner.env.unwrapped.data.qvel, qvel, atol=1e-12)
    planner.close()
    env.close()


def test_oracle_action_is_finite_and_bounded():
    env = gym.make("Pusher-v5")
    env.reset(seed=1811)
    planner = PusherOracleCEM(
        horizon=2,
        action_repeat=1,
        population=8,
        iterations=1,
        initial_std_scale=0.25,
        seed=1811,
    )
    result = planner.plan(env)
    assert result.action.shape == env.action_space.shape
    assert result.sequence.shape == (2, env.action_space.shape[0])
    assert np.isfinite(result.action).all()
    assert np.isfinite(result.sequence).all()
    assert (result.action >= env.action_space.low).all()
    assert (result.action <= env.action_space.high).all()
    planner.close()
    env.close()


def test_terminal_value_is_added_to_oracle_return():
    env = gym.make("Pusher-v5")
    env.reset(seed=1811)
    qpos = env.unwrapped.data.qpos.copy()
    qvel = env.unwrapped.data.qvel.copy()
    planner = PusherOracleCEM(
        horizon=2,
        action_repeat=1,
        population=4,
        iterations=1,
        discount=0.9,
        seed=1811,
    )
    sequences = np.zeros((4, 2, env.action_space.shape[0]), dtype=np.float32)
    plain = planner.evaluate_sequences(qpos, qvel, sequences)
    valued = planner.evaluate_sequences(
        qpos,
        qvel,
        sequences,
        terminal_value_fn=lambda states: np.ones(len(states)),
    )
    assert np.allclose(valued - plain, 0.9**2)
    planner.close()
    env.close()


def test_rollout_sequences_returns_terminal_batch_and_restores_state():
    env = gym.make("Pusher-v5")
    env.reset(seed=1811)
    qpos = env.unwrapped.data.qpos.copy()
    qvel = env.unwrapped.data.qvel.copy()
    planner = PusherOracleCEM(
        horizon=2,
        action_repeat=2,
        population=4,
        iterations=1,
        discount=0.9,
        seed=1811,
    )
    sequences = np.zeros((4, 2, env.action_space.shape[0]), dtype=np.float32)
    rewards, terminal_observations, discounts = planner.rollout_sequences(
        qpos,
        qvel,
        sequences,
    )
    assert rewards.shape == (4,)
    assert terminal_observations.shape == (4, 23)
    assert np.allclose(discounts, 0.9**4)
    assert np.allclose(planner.env.unwrapped.data.qpos, qpos, atol=1e-12)
    assert np.allclose(planner.env.unwrapped.data.qvel, qvel, atol=1e-12)
    planner.close()
    env.close()


def test_action_knots_are_linearly_expanded():
    planner = PusherOracleCEM(
        horizon=3,
        action_repeat=1,
        interpolation_steps=3,
        population=4,
        iterations=1,
        seed=1811,
    )
    knots = np.zeros((3, planner.action_dim), dtype=np.float64)
    knots[1] = 0.9
    knots[2] = -0.6
    expanded = planner.expand_sequence(knots)
    assert expanded.shape == (9, planner.action_dim)
    assert np.allclose(expanded[0], 0.0)
    assert np.allclose(expanded[1], 0.3)
    assert np.allclose(expanded[2], 0.6)
    assert np.allclose(expanded[3], 0.9)
    assert np.allclose(expanded[4], 0.4)
    assert np.allclose(expanded[5], -0.1)
    assert np.allclose(expanded[6:], -0.6)
    planner.close()
