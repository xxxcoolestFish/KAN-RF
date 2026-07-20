"""Empirical feedback-reachability funnels for coarse graph edges."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from cpbn.reachability import CoarseReachabilityPlanner, state_distance


@dataclass
class FunnelDiagnostics:
    edge_count: int
    start_route_edge_count: int
    mean_nominal_error: float
    maximum_nominal_error: float
    mean_scale: float
    source_inside_fraction: float


class EmpiricalReachabilityFunnels:
    """Turn selected point edges into regions estimated from dynamics rollouts.

    Nominal and perturbed actions exist only while the regions are estimated.
    The retained interface consists solely of source states, target centers,
    diagonal reachability scales, and graph identities.
    """

    def __init__(
        self,
        planner: CoarseReachabilityPlanner,
        dynamics: nn.Module,
        edge_count: int = 96,
        action_grid: int = 65,
        perturbations: int = 64,
        perturbation_segments: int = 4,
        action_noise: float = 0.15,
        scale_floor: float = 0.06,
        scale_ceiling: float = 0.50,
        seed: int = 0,
    ):
        if planner.macro_steps % perturbation_segments:
            raise ValueError("macro steps must be divisible by perturbation segments")
        self.planner = planner
        self.dynamics = dynamics
        self.edge_count = edge_count
        self.action_grid = action_grid
        self.perturbations = perturbations
        self.perturbation_segments = perturbation_segments
        self.action_noise = action_noise
        self.scale_floor = scale_floor
        self.scale_ceiling = scale_ceiling
        self.seed = seed
        self.graph_index, self.start_route_mask = self._select_edges()
        self.source = planner.anchors[self.graph_index].clone()
        self.center = planner.waypoint[self.graph_index].clone()
        self.scale, self.diagnostics = self._estimate_regions()

    def _start_route(self) -> list[int]:
        route, visited = [], set()
        node = 0
        while node not in visited and int(self.planner.successor[node]) >= 0:
            visited.add(node)
            route.append(node)
            node = int(self.planner.successor[node])
            if float(self.planner.distance[node]) == 0.0:
                break
        return route

    def _select_edges(self) -> tuple[torch.Tensor, torch.Tensor]:
        generator = torch.Generator().manual_seed(self.seed)
        start_route = self._start_route()
        routed = torch.where(
            (self.planner.successor >= 0)
            & torch.isfinite(self.planner.distance)
            & (self.planner.distance <= 8.0)
        )[0]
        start_set = set(start_route)
        pool = torch.tensor([int(i) for i in routed if int(i) not in start_set])
        needed = max(0, self.edge_count - len(start_route))
        if needed > pool.numel():
            raise ValueError("not enough routed edges for requested funnel count")
        selected = pool[torch.randperm(pool.numel(), generator=generator)[:needed]]
        graph_index = torch.tensor(start_route + selected.tolist(), dtype=torch.long)
        start_mask = torch.zeros(graph_index.shape[0], dtype=torch.bool)
        start_mask[:len(start_route)] = True
        return graph_index, start_mask

    @torch.no_grad()
    def _simulate(self, state: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        segment_steps = self.planner.macro_steps // actions.shape[1]
        for segment in range(actions.shape[1]):
            action = actions[:, segment:segment + 1]
            for _ in range(segment_steps):
                state = self.dynamics(state, action)
        return state

    def _estimate_regions(self) -> tuple[torch.Tensor, FunnelDiagnostics]:
        generator = torch.Generator().manual_seed(self.seed + 1)
        edges = self.source.shape[0]
        action_grid = torch.linspace(-1.0, 1.0, self.action_grid)
        state = self.source[:, None, :].expand(-1, self.action_grid, -1).reshape(-1, 6)
        actions = action_grid.view(1, -1, 1).expand(edges, -1, -1).reshape(-1, 1)
        endpoint = self._simulate(state.clone(), actions).view(edges, self.action_grid, 6)
        target = self.center[:, None, :].expand_as(endpoint)
        error = state_distance(endpoint, target)
        best = error.argmin(dim=1)
        nominal_action = action_grid[best]
        nominal_endpoint = endpoint[torch.arange(edges), best]
        nominal_error = state_distance(nominal_endpoint, self.center)

        count = edges * self.perturbations
        state = self.source[:, None, :].expand(
            -1, self.perturbations, -1,
        ).reshape(count, 6).clone()
        base = nominal_action[:, None, None].expand(
            -1, self.perturbations, self.perturbation_segments,
        )
        noise = torch.randn(
            edges, self.perturbations, self.perturbation_segments,
            generator=generator,
        ) * self.action_noise
        actions = (base + noise).clamp(-1.0, 1.0).reshape(
            count, self.perturbation_segments,
        )
        endpoint = self._simulate(state, actions).view(edges, self.perturbations, 6)
        deviation = endpoint - self.center[:, None, :]
        scale = deviation.square().mean(dim=1).sqrt().clamp(
            self.scale_floor, self.scale_ceiling,
        )
        source_inside = self.inside(self.source, self.center, scale)
        return scale, FunnelDiagnostics(
            edge_count=edges,
            start_route_edge_count=int(self.start_route_mask.sum()),
            mean_nominal_error=float(nominal_error.mean()),
            maximum_nominal_error=float(nominal_error.max()),
            mean_scale=float(scale.mean()),
            source_inside_fraction=float(source_inside.float().mean()),
        )

    @staticmethod
    def normalized_distance(
        state: torch.Tensor, center: torch.Tensor, scale: torch.Tensor,
    ) -> torch.Tensor:
        return (((state - center) / scale).square().mean(dim=-1)).sqrt()

    @classmethod
    def inside(
        cls, state: torch.Tensor, center: torch.Tensor, scale: torch.Tensor,
    ) -> torch.Tensor:
        return cls.normalized_distance(state, center, scale) <= 1.0

    def sample_initial(
        self,
        edge_index: torch.Tensor,
        generator: torch.Generator,
        angle_noise: float = 0.025,
        velocity_noise: float = 0.025,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        source = self.source[edge_index]
        theta1 = torch.atan2(source[:, 1], source[:, 0])
        theta2 = torch.atan2(source[:, 3], source[:, 2])
        theta1 = theta1 + torch.randn(theta1.shape, generator=generator) * angle_noise
        theta2 = theta2 + torch.randn(theta2.shape, generator=generator) * angle_noise
        velocity = source[:, 4:] + torch.randn(
            source.shape[0], 2, generator=generator,
        ) * velocity_noise
        state = torch.stack(
            [
                torch.cos(theta1), torch.sin(theta1),
                torch.cos(theta2), torch.sin(theta2),
                velocity[:, 0].clamp(-1.0, 1.0),
                velocity[:, 1].clamp(-1.0, 1.0),
            ],
            dim=-1,
        )
        return state, self.center[edge_index], self.scale[edge_index]
