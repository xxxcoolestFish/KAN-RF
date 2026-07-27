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
class ControlSensitivity:
    rank: int
    singular_values: tuple[float, ...]
    task_gradient_norm: float
    directed_actions: np.ndarray


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
    sensitivity: ControlSensitivity | None


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
        action_strategy: str = "primitive",
        sensitivity_probe: float = 0.5,
        sensitivity_steps: int = 3,
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
        if action_strategy not in ("primitive", "sensitivity"):
            raise ValueError("unknown action strategy")
        if sensitivity_probe <= 0.0 or sensitivity_steps < 1:
            raise ValueError("invalid sensitivity configuration")
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
        self.action_strategy = action_strategy
        self.sensitivity_probe = sensitivity_probe
        self.sensitivity_steps = sensitivity_steps
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

    def _action_library(
        self,
        depth: int,
        sensitivity: ControlSensitivity | None,
    ) -> np.ndarray:
        warm = self._warm_sequence[depth]
        magnitude = 0.5 * (self.high - self.low) * self.action_scale
        actions = [
            np.zeros(self.action_dim, dtype=np.float64),
            np.clip(warm, self.low, self.high),
        ]
        if sensitivity is None:
            for action_index in range(self.action_dim):
                positive = np.zeros(self.action_dim, dtype=np.float64)
                positive[action_index] = magnitude[action_index]
                actions.extend((positive, -positive))
        else:
            actions.extend(sensitivity.directed_actions)
        while len(actions) < self.branching:
            center = (
                sensitivity.directed_actions[
                    len(actions) % len(sensitivity.directed_actions)
                ]
                if (
                    sensitivity is not None
                    and len(sensitivity.directed_actions) > 0
                )
                else warm
            )
            perturbation = self.rng.normal(
                size=self.action_dim
            ) * (self.high - self.low) * self.noise_scale
            actions.append(np.clip(center + perturbation, self.low, self.high))
        return np.stack(actions[: self.branching])

    def _probe(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        self._restore(qpos, qvel)
        observation = None
        info = {}
        for _ in range(self.sensitivity_steps):
            observation, _, terminated, truncated, info = (
                self.env.unwrapped.step(action.astype(np.float32))
            )
            if terminated or truncated:
                break
        state_reward = float(
            info.get("reward_dist", -object_goal_distance(self.env))
            + info.get("reward_near", -self._contact_distance())
        )
        return np.asarray(observation, dtype=np.float64), state_reward

    def _control_sensitivity(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
    ) -> ControlSensitivity:
        jacobian_columns = []
        task_gradient = np.zeros(self.action_dim, dtype=np.float64)
        for action_index in range(self.action_dim):
            positive = np.zeros(self.action_dim, dtype=np.float64)
            positive[action_index] = self.sensitivity_probe
            negative = -positive
            positive_observation, positive_reward = self._probe(
                qpos,
                qvel,
                positive,
            )
            negative_observation, negative_reward = self._probe(
                qpos,
                qvel,
                negative,
            )
            denominator = 2.0 * self.sensitivity_probe
            jacobian_columns.append(
                (positive_observation - negative_observation) / denominator
            )
            task_gradient[action_index] = (
                positive_reward - negative_reward
            ) / denominator
        jacobian = np.stack(jacobian_columns, axis=1)
        _, singular_values, right_vectors = np.linalg.svd(
            jacobian,
            full_matrices=False,
        )
        target_norm = float(
            np.mean(0.5 * (self.high - self.low)) * self.action_scale
        )
        directed_actions = []
        task_gradient_norm = float(np.linalg.norm(task_gradient))
        if task_gradient_norm > 1e-9:
            task_direction = (
                task_gradient / task_gradient_norm * target_norm
            )
            directed_actions.extend(
                (
                    0.5 * task_direction,
                    task_direction,
                    -task_direction,
                )
            )
        for direction in right_vectors:
            scaled = direction * target_norm
            directed_actions.extend((scaled, -scaled))
        self._restore(qpos, qvel)
        return ControlSensitivity(
            rank=int(np.linalg.matrix_rank(jacobian)),
            singular_values=tuple(
                float(value) for value in singular_values
            ),
            task_gradient_norm=task_gradient_norm,
            directed_actions=np.asarray(
                directed_actions,
                dtype=np.float64,
            ),
        )

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
        sensitivity = (
            self._control_sensitivity(source_qpos, source_qvel)
            if self.action_strategy == "sensitivity"
            else None
        )

        for depth in range(self.depth):
            action_library = self._action_library(depth, sensitivity)
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
            sensitivity=sensitivity,
        )
