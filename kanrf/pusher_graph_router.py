"""Exact-dynamics reachability graph router for the Pusher diagnostic gate."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable

import gymnasium as gym
import numpy as np

from kanrf.pusher_oracle import object_goal_distance


def fingertip_object_distance(env: gym.Env) -> float:
    base = env.unwrapped
    fingertip = np.asarray(
        base.get_body_com("tips_arm"),
        dtype=np.float64,
    )
    obj = np.asarray(base.get_body_com("object"), dtype=np.float64)
    return float(np.linalg.norm(fingertip - obj))


@dataclass(frozen=True)
class GraphLayerTrace:
    depth: int
    parents: int
    expanded: int
    unique: int
    merged: int
    kept: int
    best_score: float
    route_distance: float
    minimum_distance: float
    route_contact_distance: float
    minimum_contact_distance: float
    route_state_reward: float


@dataclass(frozen=True)
class GraphPlanResult:
    action: np.ndarray
    sequence: np.ndarray
    predicted_return: float
    predicted_first_observation: np.ndarray
    predicted_first_reward: float
    predicted_first_distance: float
    planning_ms: float
    layers: tuple[GraphLayerTrace, ...]


@dataclass
class _GraphNode:
    qpos: np.ndarray
    qvel: np.ndarray
    observation: np.ndarray
    actions: tuple[np.ndarray, ...]
    discounted_return: float
    discount: float
    distance: float
    contact_distance: float
    state_reward: float
    first_observation: np.ndarray
    first_reward: float
    first_distance: float


class PusherOracleGraphRouter:
    """Build a local action-labelled reachability graph and route through it."""

    def __init__(
        self,
        depth: int = 6,
        branching: int = 24,
        beam_width: int = 48,
        action_scale: float = 0.5,
        noise_scale: float = 0.25,
        merge_radius: float = 0.35,
        discount: float = 0.99,
        heuristic_steps: int = 100,
        seed: int = 1811,
    ):
        if depth < 1 or branching < 2 or beam_width < 1:
            raise ValueError("depth, branching and beam_width must be positive")
        if action_scale <= 0.0 or noise_scale < 0.0:
            raise ValueError("invalid action sampling scale")
        if merge_radius <= 0.0:
            raise ValueError("merge_radius must be positive")
        if heuristic_steps < 0:
            raise ValueError("heuristic_steps must be non-negative")
        self.env = gym.make("Pusher-v5")
        self.env.reset(seed=seed)
        self.depth = depth
        self.branching = branching
        self.beam_width = beam_width
        self.action_scale = action_scale
        self.noise_scale = noise_scale
        self.merge_radius = merge_radius
        self.discount = discount
        self.heuristic_steps = heuristic_steps
        self.rng = np.random.default_rng(seed)
        self.low = np.asarray(self.env.action_space.low, dtype=np.float64)
        self.high = np.asarray(self.env.action_space.high, dtype=np.float64)
        self.action_dim = int(self.low.size)
        self._warm_sequence = np.zeros(
            (depth, self.action_dim),
            dtype=np.float64,
        )

    def close(self) -> None:
        self.env.close()

    def _restore(self, qpos: np.ndarray, qvel: np.ndarray) -> None:
        base = self.env.unwrapped
        base.set_state(qpos.copy(), qvel.copy())
        base.data.ctrl[:] = 0.0

    def _action_library(self, depth: int) -> np.ndarray:
        warm = self._warm_sequence[depth]
        magnitude = 0.5 * (self.high - self.low) * self.action_scale
        actions = [
            np.zeros(self.action_dim, dtype=np.float64),
            np.clip(warm, self.low, self.high),
        ]
        for action_index in range(self.action_dim):
            positive = np.zeros(self.action_dim, dtype=np.float64)
            positive[action_index] = magnitude[action_index]
            actions.extend((positive, -positive))
        while len(actions) < self.branching:
            perturbation = self.rng.normal(
                size=self.action_dim
            ) * (self.high - self.low) * self.noise_scale
            actions.append(np.clip(warm + perturbation, self.low, self.high))
        return np.stack(actions[: self.branching])

    def _merge(self, nodes: list[_GraphNode]) -> list[_GraphNode]:
        observations = np.stack([node.observation for node in nodes])
        scale = observations.std(axis=0).clip(1e-3, None)
        normalized = observations / scale
        keys = np.rint(normalized / self.merge_radius).astype(np.int64)
        best_by_key: dict[tuple[int, ...], _GraphNode] = {}
        for key_array, node in zip(keys, nodes, strict=True):
            key = tuple(int(value) for value in key_array)
            previous = best_by_key.get(key)
            if (
                previous is None
                or self._score(node) > self._score(previous)
            ):
                best_by_key[key] = node
        return list(best_by_key.values())

    def _score(self, node: _GraphNode) -> float:
        if self.heuristic_steps == 0:
            return node.discounted_return
        if np.isclose(self.discount, 1.0):
            tail_multiplier = float(self.heuristic_steps)
        else:
            tail_multiplier = (
                1.0 - self.discount**self.heuristic_steps
            ) / (1.0 - self.discount)
        return (
            node.discounted_return
            + node.discount * tail_multiplier * node.state_reward
        )

    def _contact_distance(self) -> float:
        return fingertip_object_distance(self.env)

    def plan(
        self,
        source_env: gym.Env,
        debug_callback: Callable[[GraphLayerTrace], None] | None = None,
    ) -> GraphPlanResult:
        start = perf_counter()
        source = source_env.unwrapped
        source_qpos = np.asarray(source.data.qpos, dtype=np.float64).copy()
        source_qvel = np.asarray(source.data.qvel, dtype=np.float64).copy()
        self._restore(source_qpos, source_qvel)
        source_observation = np.asarray(
            self.env.unwrapped._get_obs(),
            dtype=np.float32,
        )
        root = _GraphNode(
            qpos=source_qpos,
            qvel=source_qvel,
            observation=source_observation,
            actions=(),
            discounted_return=0.0,
            discount=1.0,
            distance=object_goal_distance(self.env),
            contact_distance=self._contact_distance(),
            state_reward=(
                -object_goal_distance(self.env) - self._contact_distance()
            ),
            first_observation=source_observation,
            first_reward=0.0,
            first_distance=object_goal_distance(self.env),
        )
        frontier = [root]
        traces: list[GraphLayerTrace] = []

        for depth in range(self.depth):
            action_library = self._action_library(depth)
            successors: list[_GraphNode] = []
            for parent in frontier:
                for action in action_library:
                    self._restore(parent.qpos, parent.qvel)
                    observation, reward, terminated, truncated, info = (
                        self.env.unwrapped.step(action.astype(np.float32))
                    )
                    observation = np.asarray(observation, dtype=np.float32)
                    distance = object_goal_distance(self.env)
                    contact_distance = self._contact_distance()
                    state_reward = float(
                        info.get("reward_dist", -distance)
                        + info.get("reward_near", -contact_distance)
                    )
                    first_observation = (
                        observation
                        if not parent.actions
                        else parent.first_observation
                    )
                    first_reward = (
                        float(reward)
                        if not parent.actions
                        else parent.first_reward
                    )
                    first_distance = (
                        distance
                        if not parent.actions
                        else parent.first_distance
                    )
                    successors.append(
                        _GraphNode(
                            qpos=np.asarray(
                                self.env.unwrapped.data.qpos,
                                dtype=np.float64,
                            ).copy(),
                            qvel=np.asarray(
                                self.env.unwrapped.data.qvel,
                                dtype=np.float64,
                            ).copy(),
                            observation=observation,
                            actions=parent.actions + (action.copy(),),
                            discounted_return=(
                                parent.discounted_return
                                + parent.discount * float(reward)
                            ),
                            discount=parent.discount * self.discount,
                            distance=distance,
                            contact_distance=contact_distance,
                            state_reward=state_reward,
                            first_observation=first_observation,
                            first_reward=first_reward,
                            first_distance=first_distance,
                        )
                    )
                    if terminated or truncated:
                        continue
            merged = self._merge(successors)
            merged.sort(
                key=self._score,
                reverse=True,
            )
            frontier = merged[: self.beam_width]
            route = frontier[0]
            trace = GraphLayerTrace(
                depth=depth + 1,
                parents=max(1, len(successors) // len(action_library)),
                expanded=len(successors),
                unique=len(merged),
                merged=len(successors) - len(merged),
                kept=len(frontier),
                best_score=float(self._score(route)),
                route_distance=route.distance,
                minimum_distance=float(
                    min(node.distance for node in frontier)
                ),
                route_contact_distance=route.contact_distance,
                minimum_contact_distance=float(
                    min(node.contact_distance for node in frontier)
                ),
                route_state_reward=route.state_reward,
            )
            traces.append(trace)
            if debug_callback is not None:
                debug_callback(trace)

        best = max(frontier, key=self._score)
        sequence = np.stack(best.actions)
        self._warm_sequence[:-1] = sequence[1:]
        self._warm_sequence[-1] = 0.0
        self._restore(source_qpos, source_qvel)
        return GraphPlanResult(
            action=sequence[0].astype(np.float32),
            sequence=sequence.astype(np.float32),
            predicted_return=float(self._score(best)),
            predicted_first_observation=best.first_observation,
            predicted_first_reward=best.first_reward,
            predicted_first_distance=best.first_distance,
            planning_ms=(perf_counter() - start) * 1000.0,
            layers=tuple(traces),
        )
