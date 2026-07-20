"""Coarse state-space reachability routing derived from a dynamics model."""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import torch
from torch import nn

from cpbn.acrobot import GOAL, random_states, tip_height


STATE_METRIC = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.35, 0.35])


def state_distance(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    weights = STATE_METRIC.to(dtype=left.dtype, device=left.device)
    return ((left - right) * weights).square().sum(dim=-1).sqrt()


@dataclass
class ReachabilityDiagnostics:
    anchor_count: int
    edge_count: int
    goal_anchor_count: int
    routed_fraction: float
    mean_snap_error: float
    start_has_route: bool
    start_route_cost: float
    start_route_hops: int


class CoarseReachabilityPlanner:
    """Build a directed macro-state graph without exposing rollout actions.

    Candidate action sequences are used only to estimate which state regions
    are reachable.  They are discarded after graph construction.  Runtime
    queries return a state waypoint and a scalar graph distance, never an
    action or action sequence.
    """

    def __init__(
        self,
        dynamics: nn.Module,
        anchor_count: int = 1024,
        samples_per_anchor: int = 16,
        macro_steps: int = 16,
        action_segments: int = 4,
        maximum_snap_error: float = 0.65,
        seed: int = 0,
    ):
        if macro_steps % action_segments:
            raise ValueError("macro_steps must be divisible by action_segments")
        self.dynamics = dynamics
        self.anchor_count = anchor_count
        self.samples_per_anchor = samples_per_anchor
        self.macro_steps = macro_steps
        self.action_segments = action_segments
        self.maximum_snap_error = maximum_snap_error
        self.seed = seed
        self.metric = STATE_METRIC
        self.anchors: torch.Tensor
        self.distance: torch.Tensor
        self.successor: torch.Tensor
        self.waypoint: torch.Tensor
        self.diagnostics = self._build()

    @torch.no_grad()
    def _build(self) -> ReachabilityDiagnostics:
        generator = torch.Generator().manual_seed(self.seed)
        anchors = random_states(self.anchor_count, generator)
        anchors[0] = torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        self.anchors = anchors

        count = self.anchor_count * self.samples_per_anchor
        state = anchors[:, None, :].expand(
            -1, self.samples_per_anchor, -1,
        ).reshape(count, 6).clone()
        actions = torch.rand(
            count, self.action_segments, generator=generator,
        ) * 2.0 - 1.0
        segment_steps = self.macro_steps // self.action_segments
        for segment in range(self.action_segments):
            action = actions[:, segment:segment + 1]
            for _ in range(segment_steps):
                state = self.dynamics(state, action)
        endpoints = state

        weighted_anchors = anchors * self.metric
        target_parts, error_parts = [], []
        batch = 1024
        for start in range(0, count, batch):
            weighted_endpoint = endpoints[start:start + batch] * self.metric
            distances = torch.cdist(weighted_endpoint, weighted_anchors)
            error, target = distances.min(dim=1)
            target_parts.append(target)
            error_parts.append(error)
        target = torch.cat(target_parts)
        snap_error = torch.cat(error_parts)
        source = torch.arange(self.anchor_count).repeat_interleave(
            self.samples_per_anchor,
        )
        valid = (target != source) & (snap_error <= self.maximum_snap_error)

        source = source[valid]
        target = target[valid]
        endpoints = endpoints[valid]
        snap_error = snap_error[valid]
        reverse: list[list[tuple[int, float, int]]] = [
            [] for _ in range(self.anchor_count)
        ]
        for edge in range(source.numel()):
            edge_source = int(source[edge])
            edge_target = int(target[edge])
            cost = 1.0 + float(snap_error[edge])
            reverse[edge_target].append((edge_source, cost, edge))

        goal_mask = tip_height(anchors) >= 1.0
        distance = torch.full((self.anchor_count,), float("inf"))
        successor = torch.full((self.anchor_count,), -1, dtype=torch.long)
        waypoint = GOAL.view(1, -1).expand(self.anchor_count, -1).clone()
        queue: list[tuple[float, int]] = []
        for index in torch.where(goal_mask)[0].tolist():
            distance[index] = 0.0
            heapq.heappush(queue, (0.0, index))

        while queue:
            current_distance, node = heapq.heappop(queue)
            if current_distance > float(distance[node]) + 1e-8:
                continue
            for predecessor, edge_cost, edge in reverse[node]:
                candidate = current_distance + edge_cost
                if candidate < float(distance[predecessor]):
                    distance[predecessor] = candidate
                    successor[predecessor] = node
                    waypoint[predecessor] = endpoints[edge]
                    heapq.heappush(queue, (candidate, predecessor))

        self.distance = distance
        self.successor = successor
        self.waypoint = waypoint
        hops = self._route_hops(0)
        return ReachabilityDiagnostics(
            anchor_count=self.anchor_count,
            edge_count=int(source.numel()),
            goal_anchor_count=int(goal_mask.sum()),
            routed_fraction=float(torch.isfinite(distance).float().mean()),
            mean_snap_error=float(snap_error.mean()) if snap_error.numel() else float("nan"),
            start_has_route=bool(torch.isfinite(distance[0])),
            start_route_cost=float(distance[0]),
            start_route_hops=hops,
        )

    def _route_hops(self, start: int) -> int:
        visited = set()
        node = start
        hops = 0
        while node not in visited and int(self.successor[node]) >= 0:
            visited.add(node)
            node = int(self.successor[node])
            hops += 1
            if float(self.distance[node]) == 0.0:
                return hops
        return -1

    @torch.no_grad()
    def nearest_anchor(self, state: torch.Tensor) -> torch.Tensor:
        weighted_state = state * self.metric.to(state)
        weighted_anchor = self.anchors.to(state) * self.metric.to(state)
        return torch.cdist(weighted_state, weighted_anchor).argmin(dim=1)

    @torch.no_grad()
    def query(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        anchor = self.nearest_anchor(state.cpu())
        reference = self.waypoint[anchor].to(state)
        route_distance = self.distance[anchor].to(state)
        unrouted = ~torch.isfinite(route_distance)
        if unrouted.any():
            reference[unrouted] = GOAL.to(state)
        return reference, route_distance

