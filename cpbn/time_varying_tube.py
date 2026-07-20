"""Continuous nominal routes and time-varying feedback tube geometry."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from cpbn.acrobot import tip_height


def tangent_coordinates(state: torch.Tensor) -> torch.Tensor:
    """Map the redundant six-state encoding to local angle/velocity coordinates."""
    return torch.stack(
        [
            torch.atan2(state[..., 1], state[..., 0]),
            torch.atan2(state[..., 3], state[..., 2]),
            state[..., 4],
            state[..., 5],
        ],
        dim=-1,
    )


def tangent_error(state: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
    state_x = tangent_coordinates(state)
    center_x = tangent_coordinates(center)
    angle = state_x[..., :2] - center_x[..., :2]
    angle = torch.atan2(torch.sin(angle), torch.cos(angle))
    return torch.cat([angle, state_x[..., 2:] - center_x[..., 2:]], dim=-1)


def apply_tangent_error(center: torch.Tensor, error: torch.Tensor) -> torch.Tensor:
    center_x = tangent_coordinates(center)
    angle = center_x[..., :2] + error[..., :2]
    velocity = center_x[..., 2:] + error[..., 2:]
    return torch.stack(
        [
            torch.cos(angle[..., 0]), torch.sin(angle[..., 0]),
            torch.cos(angle[..., 1]), torch.sin(angle[..., 1]),
            velocity[..., 0].clamp(-1.0, 1.0),
            velocity[..., 1].clamp(-1.0, 1.0),
        ],
        dim=-1,
    )


@dataclass
class RouteDiagnostics:
    segment_count: int
    segment_steps: int
    maximum_height: float
    success_step: int
    selected_segments: list[int]
    selected_labels: list[str]


@dataclass
class NominalRoute:
    states: torch.Tensor
    actions: torch.Tensor
    selected_segments: torch.Tensor
    diagnostics: RouteDiagnostics


@torch.no_grad()
def plan_continuous_cem_route(
    dynamics: nn.Module,
    segment_count: int = 20,
    segment_steps: int = 24,
    population: int = 2048,
    elite_count: int = 128,
    iterations: int = 12,
    seed: int = 0,
) -> NominalRoute:
    """Search construction-only macro actions and retain their state route."""
    torch.manual_seed(seed)
    mean = torch.zeros(segment_count)
    std = torch.ones(segment_count)
    start = torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    best_height = -float("inf")
    best_actions = None
    best_step = -1
    for _ in range(iterations):
        actions = (mean + std * torch.randn(population, segment_count)).clamp(-1.0, 1.0)
        state = start.view(1, -1).expand(population, -1).clone()
        maximum = torch.full((population,), -2.0)
        maximum_step = torch.zeros(population, dtype=torch.long)
        for segment in range(segment_count):
            action = actions[:, segment:segment + 1]
            for local_step in range(segment_steps):
                state = dynamics(state, action)
                height = tip_height(state)
                improved = height > maximum
                maximum = torch.maximum(maximum, height)
                step = segment * segment_steps + local_step + 1
                maximum_step = torch.where(
                    improved, torch.full_like(maximum_step, step), maximum_step,
                )
        score = maximum - 0.002 * actions.square().mean(dim=1)
        elite = score.topk(elite_count).indices
        mean = 0.25 * mean + 0.75 * actions[elite].mean(dim=0)
        std = (0.25 * std + 0.75 * actions[elite].std(dim=0)).clamp(0.05, 1.0)
        index = int(score.argmax())
        if float(maximum[index]) > best_height:
            best_height = float(maximum[index])
            best_actions = actions[index].clone()
            best_step = int(maximum_step[index])
    if best_actions is None or best_height < 1.0:
        raise RuntimeError(f"CEM failed to find a successful continuous route: {best_height:.3f}")

    state = start.view(1, -1)
    states = [state.squeeze(0).clone()]
    for segment in range(segment_count):
        action = best_actions[segment].view(1, 1)
        for _ in range(segment_steps):
            state = dynamics(state, action)
            states.append(state.squeeze(0).clone())
    states = torch.stack(states)
    success_segment = min(segment_count - 1, (best_step - 1) // segment_steps)
    segment_starts = states[::segment_steps][:segment_count]
    speed = segment_starts[:, 4:].square().sum(dim=-1)
    if success_segment > 1:
        middle = int(speed[1:success_segment].argmax()) + 1
    else:
        middle = max(0, success_segment - 1)
    selected = []
    labels = []
    for index, label in ((0, "initial_energy"), (middle, "high_speed"),
                         (success_segment, "terminal_swing")):
        if index not in selected:
            selected.append(index)
            labels.append(label)
    if len(selected) < 3:
        for index in range(segment_count):
            if index not in selected:
                selected.insert(-1, index)
                labels.insert(-1, "intermediate")
            if len(selected) == 3:
                break
    return NominalRoute(
        states=states,
        actions=best_actions,
        selected_segments=torch.tensor(selected, dtype=torch.long),
        diagnostics=RouteDiagnostics(
            segment_count=segment_count,
            segment_steps=segment_steps,
            maximum_height=best_height,
            success_step=best_step,
            selected_segments=selected,
            selected_labels=labels,
        ),
    )


def _local_jacobians(
    dynamics: nn.Module,
    center: torch.Tensor,
    next_center: torch.Tensor,
    nominal_action: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    zero_state = torch.zeros(4, requires_grad=True)
    zero_action = torch.zeros(1, requires_grad=True)

    def local_map(state_error, action_error):
        state = apply_tangent_error(center, state_error).view(1, 6)
        action = (nominal_action + action_error).view(1, 1).clamp(-1.0, 1.0)
        predicted = dynamics(state, action)
        return tangent_error(predicted, next_center.view(1, 6)).squeeze(0)

    state_jacobian, action_jacobian = torch.autograd.functional.jacobian(
        local_map, (zero_state, zero_action), vectorize=True,
    )
    return state_jacobian.detach(), action_jacobian.detach()


def _riccati_gains(
    dynamics: nn.Module,
    centers: torch.Tensor,
    nominal_action: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    horizon = centers.shape[0] - 1
    state_jacobians, action_jacobians = [], []
    for step in range(horizon):
        state_jacobian, action_jacobian = _local_jacobians(
            dynamics, centers[step], centers[step + 1], nominal_action,
        )
        state_jacobians.append(state_jacobian)
        action_jacobians.append(action_jacobian)
    state_jacobians = torch.stack(state_jacobians)
    action_jacobians = torch.stack(action_jacobians)

    state_cost = torch.diag(torch.tensor([2.0, 2.0, 0.5, 0.5]))
    value = torch.diag(torch.tensor([8.0, 8.0, 2.0, 2.0]))
    action_cost = torch.tensor(0.20)
    gains = torch.zeros(horizon, 1, 4)
    for step in reversed(range(horizon)):
        state_matrix = state_jacobians[step]
        action_matrix = action_jacobians[step]
        denominator = action_cost + (action_matrix.T @ value @ action_matrix).squeeze()
        gain = -(action_matrix.T @ value @ state_matrix) / denominator.clamp_min(1e-6)
        gains[step] = gain
        value = (
            state_cost
            + state_matrix.T @ value @ state_matrix
            + state_matrix.T @ value @ action_matrix @ gain
        )
        value = 0.5 * (value + value.T) + 1e-6 * torch.eye(4)
    return gains, state_jacobians, action_jacobians


@dataclass
class TubeDiagnostics:
    edge_count: int
    horizon: int
    construction_lqr_completion: list[float]
    construction_lqr_tube_adherence: list[float]
    mean_condition_number: float


class TimeVaryingTubeSet:
    """Three time-indexed full-matrix tubes derived from a continuous route."""

    def __init__(
        self,
        dynamics: nn.Module,
        route: NominalRoute,
        construction_samples: int = 1024,
        quantile: float = 0.99,
        angle_noise: float = 0.025,
        velocity_noise: float = 0.025,
        seed: int = 0,
    ):
        self.dynamics = dynamics
        self.route = route
        self.horizon = route.diagnostics.segment_steps
        self.edge_count = int(route.selected_segments.numel())
        self.angle_noise = angle_noise
        self.velocity_noise = velocity_noise
        self.seed = seed
        centers, gains, precision, radius = [], [], [], []
        completion, adherence, condition = [], [], []
        for edge, segment_tensor in enumerate(route.selected_segments):
            segment = int(segment_tensor)
            start = segment * self.horizon
            edge_centers = route.states[start:start + self.horizon + 1].clone()
            nominal_action = route.actions[segment].clone()
            edge_gains, _, _ = _riccati_gains(
                dynamics, edge_centers, nominal_action,
            )
            errors = self._closed_loop_errors(
                edge_centers, nominal_action, edge_gains,
                construction_samples, seed + edge,
            )
            edge_precision, edge_radius = [], []
            for step in range(self.horizon + 1):
                error = errors[:, step]
                covariance = error.T @ error / max(1, construction_samples - 1)
                covariance = covariance + torch.diag(
                    torch.tensor([0.03**2, 0.03**2, 0.02**2, 0.02**2]),
                )
                inverse = torch.linalg.inv(covariance)
                distance = torch.einsum("bi,ij,bj->b", error, inverse, error)
                edge_precision.append(inverse)
                edge_radius.append(torch.quantile(distance, quantile).clamp_min(1.0))
                condition.append(float(torch.linalg.cond(covariance)))
            edge_precision = torch.stack(edge_precision)
            edge_radius = torch.stack(edge_radius)
            normalized = torch.einsum(
                "bti,tij,btj->bt", errors, edge_precision, errors,
            ) / edge_radius.unsqueeze(0)
            completion.append(float((normalized[:, -1] <= 1.0).float().mean()))
            adherence.append(float((normalized <= 1.0).all(dim=1).float().mean()))
            centers.append(edge_centers); gains.append(edge_gains)
            precision.append(edge_precision); radius.append(edge_radius)
        self.centers = torch.stack(centers)
        self._construction_actions = route.actions[route.selected_segments].clone()
        self._construction_gains = torch.stack(gains)
        self.precision = torch.stack(precision)
        self.radius = torch.stack(radius)
        normalized_precision = self.precision / self.radius[..., None, None]
        self.root = torch.linalg.cholesky(normalized_precision)
        self.diagnostics = TubeDiagnostics(
            edge_count=self.edge_count,
            horizon=self.horizon,
            construction_lqr_completion=completion,
            construction_lqr_tube_adherence=adherence,
            mean_condition_number=sum(condition) / len(condition),
        )

    def sample_initial(
        self, edge: torch.Tensor, generator: torch.Generator,
    ) -> torch.Tensor:
        center = self.centers[edge, 0]
        error = torch.cat(
            [
                torch.randn(edge.shape[0], 2, generator=generator) * self.angle_noise,
                torch.randn(edge.shape[0], 2, generator=generator) * self.velocity_noise,
            ],
            dim=-1,
        )
        return apply_tangent_error(center, error)

    def _closed_loop_errors(
        self,
        centers: torch.Tensor,
        nominal_action: torch.Tensor,
        gains: torch.Tensor,
        sample_count: int,
        seed: int,
    ) -> torch.Tensor:
        generator = torch.Generator().manual_seed(seed)
        edge = torch.zeros(sample_count, dtype=torch.long)
        center = centers[0].view(1, 6).expand(sample_count, -1)
        error = torch.cat(
            [
                torch.randn(sample_count, 2, generator=generator) * self.angle_noise,
                torch.randn(sample_count, 2, generator=generator) * self.velocity_noise,
            ],
            dim=-1,
        )
        del edge
        state = apply_tangent_error(center, error)
        errors = [tangent_error(state, centers[0].view(1, 6))]
        with torch.no_grad():
            for step in range(self.horizon):
                error = tangent_error(state, centers[step].view(1, 6))
                correction = error @ gains[step].T
                action = (nominal_action + correction).clamp(-1.0, 1.0)
                state = self.dynamics(state, action)
                errors.append(tangent_error(state, centers[step + 1].view(1, 6)))
        return torch.stack(errors, dim=1)

    def normalized_distance(
        self, state: torch.Tensor, edge: torch.Tensor, phase: torch.Tensor,
    ) -> torch.Tensor:
        center = self.centers[edge, phase]
        error = tangent_error(state, center)
        root = self.root[edge, phase]
        whitened = torch.bmm(error.unsqueeze(1), root).squeeze(1)
        return whitened.square().sum(dim=-1)

    def policy_features(
        self,
        state: torch.Tensor,
        descriptor_edge: torch.Tensor,
        phase: torch.Tensor,
    ) -> torch.Tensor:
        center = self.centers[descriptor_edge, phase]
        next_phase = (phase + 1).clamp_max(self.horizon)
        next_center = self.centers[descriptor_edge, next_phase]
        error = tangent_error(state, center)
        root = self.root[descriptor_edge, phase]
        whitened = torch.bmm(error.unsqueeze(1), root).squeeze(1)
        desired = tangent_error(next_center, state)
        phase_feature = phase.to(state.dtype).unsqueeze(-1) / self.horizon
        return torch.cat([state, whitened, desired, phase_feature], dim=-1)

    @torch.no_grad()
    def evaluate_hidden_lqr(
        self, trials_per_edge: int, seed: int,
    ) -> dict:
        generator = torch.Generator().manual_seed(seed)
        edge = torch.arange(self.edge_count).repeat_interleave(trials_per_edge)
        state = self.sample_initial(edge, generator)
        inside_all = torch.ones(edge.shape[0], dtype=torch.bool)
        for step in range(self.horizon):
            center = self.centers[edge, step]
            error = tangent_error(state, center)
            gain = self._construction_gains[edge, step]
            action = self._construction_actions[edge].unsqueeze(-1) + torch.bmm(
                gain, error.unsqueeze(-1),
            ).squeeze(-1)
            state = self.dynamics(state, action.clamp(-1.0, 1.0))
            phase = torch.full_like(edge, step + 1)
            inside_all &= self.normalized_distance(state, edge, phase) <= 1.0
        final_phase = torch.full_like(edge, self.horizon)
        completed = self.normalized_distance(state, edge, final_phase) <= 1.0
        per_edge = completed.view(self.edge_count, trials_per_edge).float().mean(1)
        return {
            "completion_rate": float(completed.float().mean()),
            "per_edge_completion": per_edge.tolist(),
            "full_tube_adherence": float(inside_all.float().mean()),
        }
