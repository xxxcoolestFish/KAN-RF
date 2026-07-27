"""Real-simulator Oracle-CEM for the fixed-physics Pusher gate.

This planner is diagnostic only. It restores MuJoCo qpos/qvel before every
candidate rollout, so its predictions contain no learned-model error.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable

import gymnasium as gym
import numpy as np


@dataclass(frozen=True)
class CEMIteration:
    iteration: int
    best_return: float
    elite_mean: float
    population_mean: float
    population_std: float
    action_std_mean: float


@dataclass(frozen=True)
class PlanResult:
    action: np.ndarray
    predicted_return: float
    planning_ms: float
    iterations: tuple[CEMIteration, ...]


def object_goal_distance(env: gym.Env) -> float:
    base = env.unwrapped
    obj = np.asarray(base.get_body_com("object"), dtype=np.float64)
    goal = np.asarray(base.get_body_com("goal"), dtype=np.float64)
    return float(np.linalg.norm(obj - goal))


class PusherOracleCEM:
    """CEM planner that queries a cloned Pusher simulator state."""

    def __init__(
        self,
        horizon: int = 6,
        action_repeat: int = 2,
        population: int = 128,
        elite_fraction: float = 0.1,
        iterations: int = 3,
        discount: float = 0.99,
        initial_std_scale: float = 0.35,
        temporal_correlation: float = 0.7,
        seed: int = 1811,
    ):
        if horizon < 1 or action_repeat < 1 or population < 2 or iterations < 1:
            raise ValueError("invalid positive planner configuration")
        self.env = gym.make("Pusher-v5")
        self.env.reset(seed=seed)
        self.horizon = horizon
        self.action_repeat = action_repeat
        self.population = population
        self.elite_count = max(2, int(population * elite_fraction))
        self.iterations = iterations
        self.discount = discount
        self.initial_std_scale = initial_std_scale
        self.temporal_correlation = temporal_correlation
        self.rng = np.random.default_rng(seed)
        self.low = np.asarray(self.env.action_space.low, dtype=np.float64)
        self.high = np.asarray(self.env.action_space.high, dtype=np.float64)
        self.action_dim = int(self.low.size)
        self._warm_mean = np.zeros((horizon, self.action_dim), dtype=np.float64)

    def close(self) -> None:
        self.env.close()

    def _restore(self, qpos: np.ndarray, qvel: np.ndarray) -> None:
        base = self.env.unwrapped
        base.set_state(qpos.copy(), qvel.copy())
        base.data.ctrl[:] = 0.0

    def evaluate_sequences(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
        sequences: np.ndarray,
        terminal_value_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> np.ndarray:
        returns, final_observations, terminal_discounts = (
            self.rollout_sequences(qpos, qvel, sequences)
        )
        if terminal_value_fn is not None:
            terminal_values = np.asarray(
                terminal_value_fn(final_observations),
                dtype=np.float64,
            )
            if terminal_values.shape != (len(sequences),):
                raise ValueError(
                    "terminal_value_fn must return one scalar per observation"
                )
            returns += terminal_discounts * terminal_values
        return returns

    def rollout_sequences(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
        sequences: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return rewards, terminal observations and terminal discounts."""
        if sequences.shape != (len(sequences), self.horizon, self.action_dim):
            raise ValueError(
                f"expected [N,{self.horizon},{self.action_dim}], "
                f"got {sequences.shape}"
            )
        returns = np.zeros(len(sequences), dtype=np.float64)
        final_observations: list[np.ndarray] = []
        terminal_discounts = np.zeros(len(sequences), dtype=np.float64)
        for candidate_index, sequence in enumerate(sequences):
            self._restore(qpos, qvel)
            discount = 1.0
            total = 0.0
            for action in sequence:
                for _ in range(self.action_repeat):
                    observation, reward, terminated, truncated, _ = (
                        self.env.unwrapped.step(
                        action.astype(np.float32)
                    )
                    )
                    total += discount * float(reward)
                    discount *= self.discount
                    if terminated or truncated:
                        break
                if terminated or truncated:
                    break
            returns[candidate_index] = total
            final_observations.append(np.asarray(observation, dtype=np.float32))
            terminal_discounts[candidate_index] = discount
        self._restore(qpos, qvel)
        return returns, np.stack(final_observations), terminal_discounts

    def policy_proposal(
        self,
        source_env: gym.Env,
        policy_fn: Callable[[np.ndarray], np.ndarray],
    ) -> np.ndarray:
        """Roll a closed-loop policy through the exact simulator as a CEM seed."""
        source = source_env.unwrapped
        qpos = np.asarray(source.data.qpos, dtype=np.float64).copy()
        qvel = np.asarray(source.data.qvel, dtype=np.float64).copy()
        self._restore(qpos, qvel)
        proposal = []
        observation = self.env.unwrapped._get_obs()
        for _ in range(self.horizon):
            action = np.asarray(policy_fn(observation), dtype=np.float32)
            proposal.append(action)
            for _ in range(self.action_repeat):
                observation, _, terminated, truncated, _ = (
                    self.env.unwrapped.step(action)
                )
                if terminated or truncated:
                    break
        self._restore(qpos, qvel)
        return np.stack(proposal)

    def plan(
        self,
        source_env: gym.Env,
        debug_callback: Callable[[CEMIteration], None] | None = None,
        terminal_value_fn: Callable[[np.ndarray], np.ndarray] | None = None,
        proposal_sequence: np.ndarray | None = None,
    ) -> PlanResult:
        start = perf_counter()
        source = source_env.unwrapped
        qpos = np.asarray(source.data.qpos, dtype=np.float64).copy()
        qvel = np.asarray(source.data.qvel, dtype=np.float64).copy()

        mean = (
            np.asarray(proposal_sequence, dtype=np.float64).copy()
            if proposal_sequence is not None
            else self._warm_mean.copy()
        )
        if mean.shape != self._warm_mean.shape:
            raise ValueError(
                f"proposal must have shape {self._warm_mean.shape}, got {mean.shape}"
            )
        std = np.broadcast_to(
            (self.high - self.low) * self.initial_std_scale,
            mean.shape,
        ).copy()
        best_sequence = mean.copy()
        best_return = -np.inf
        traces: list[CEMIteration] = []

        for iteration in range(self.iterations):
            noise = self.rng.standard_normal(
                (self.population, self.horizon, self.action_dim)
            )
            for time_index in range(1, self.horizon):
                noise[:, time_index] = (
                    self.temporal_correlation * noise[:, time_index - 1]
                    + (1.0 - self.temporal_correlation) * noise[:, time_index]
                )
            candidates = np.clip(
                mean[None, ...] + std[None, ...] * noise,
                self.low,
                self.high,
            )
            candidates[0] = np.clip(mean, self.low, self.high)
            probe_magnitude = 0.5 * (self.high - self.low)
            for action_index in range(self.action_dim):
                positive_index = 1 + 2 * action_index
                negative_index = positive_index + 1
                if negative_index >= self.population:
                    break
                candidates[positive_index] = mean
                candidates[negative_index] = mean
                candidates[positive_index, :, action_index] = np.clip(
                    mean[:, action_index] + probe_magnitude[action_index],
                    self.low[action_index],
                    self.high[action_index],
                )
                candidates[negative_index, :, action_index] = np.clip(
                    mean[:, action_index] - probe_magnitude[action_index],
                    self.low[action_index],
                    self.high[action_index],
                )
            returns = self.evaluate_sequences(
                qpos,
                qvel,
                candidates,
                terminal_value_fn=terminal_value_fn,
            )
            elite_indices = np.argpartition(returns, -self.elite_count)[
                -self.elite_count:
            ]
            elites = candidates[elite_indices]
            elite_returns = returns[elite_indices]
            mean = elites.mean(axis=0)
            std = elites.std(axis=0).clip(0.05, None)

            local_best = int(np.argmax(returns))
            if float(returns[local_best]) > best_return:
                best_return = float(returns[local_best])
                best_sequence = candidates[local_best].copy()

            trace = CEMIteration(
                iteration=iteration,
                best_return=float(returns.max()),
                elite_mean=float(elite_returns.mean()),
                population_mean=float(returns.mean()),
                population_std=float(returns.std()),
                action_std_mean=float(std.mean()),
            )
            traces.append(trace)
            if debug_callback is not None:
                debug_callback(trace)

        self._warm_mean[:-1] = best_sequence[1:]
        self._warm_mean[-1] = 0.0
        return PlanResult(
            action=best_sequence[0].astype(np.float32),
            predicted_return=best_return,
            planning_ms=(perf_counter() - start) * 1000.0,
            iterations=tuple(traces),
        )
