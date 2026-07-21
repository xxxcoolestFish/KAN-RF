"""Counterfactual Bellman/Sobolev refinement of the cognitive adjoint Actor."""

from __future__ import annotations

import torch

from cpbn.cognitive_adjoint import BellmanAdjointActor
from cpbn.receding_tube import local_state_distance
from cpbn.time_varying_tube import tangent_coordinates


class CounterfactualAdjointActor(BellmanAdjointActor):
    """Use a one-step task utility plus future potential in the pullback."""

    def __init__(
        self,
        hidden_dim: int = 64,
        corridor_horizon: int = 12,
        log_std: float = 0.0,
        ridge: float = 1e-5,
        log_gain: float = -3.0,
        gamma: float = 0.99,
        progress_weight: float = 1.0,
        action_penalty: float = 0.002,
    ):
        super().__init__(
            hidden_dim, corridor_horizon, log_std, ridge, log_gain,
        )
        self.gamma = gamma
        self.progress_weight = progress_weight
        self.action_penalty = action_penalty

    def immediate_utility(self, action, predicted, corridor):
        distance = local_state_distance(predicted, corridor[:, 0])
        return (
            -self.progress_weight * distance
            - self.action_penalty * action.square().sum(dim=-1)
        )

    def model_q(self, state, corridor, cognition, action):
        predicted = cognition(state, action)
        immediate = self.immediate_utility(action, predicted, corridor)
        future = self.potential_value(predicted, corridor)
        return immediate + self.gamma * future

    def target_q(self, state, corridor, cognition, action, target_value):
        predicted = cognition(state, action)
        immediate = self.immediate_utility(action, predicted, corridor)
        return immediate + self.gamma * target_value(predicted, corridor)

    def mean_with_diagnostics(self, state, corridor, cognition):
        build_graph = torch.is_grad_enabled()
        with torch.enable_grad():
            anchor = torch.zeros(
                state.shape[0], 1, dtype=state.dtype, device=state.device,
                requires_grad=True,
            )
            predicted = cognition(state.detach(), anchor)
            tangent = tangent_coordinates(predicted)
            control_rows = []
            for dimension in range(tangent.shape[-1]):
                row = torch.autograd.grad(
                    tangent[:, dimension].sum(), anchor,
                    retain_graph=True, create_graph=False,
                )[0]
                control_rows.append(row.detach())
            control_map = torch.stack(control_rows, dim=1)
            gramian = control_map.square().sum(dim=1)
            objective = (
                self.immediate_utility(anchor, predicted, corridor.detach())
                + self.gamma * self.potential_value(
                    predicted, corridor.detach(),
                )
            )
            pullback = torch.autograd.grad(
                objective.sum(), anchor,
                create_graph=build_graph, retain_graph=build_graph,
            )[0]
            inverse_control = pullback / (gramian + self.ridge)
            unconstrained = self.log_gain.clamp(-7.0, 3.0).exp()
            unconstrained = unconstrained * inverse_control
            mean = 5.0 * torch.tanh(unconstrained / 5.0)
        if not build_graph:
            mean = mean.detach()
            pullback = pullback.detach()
            gramian = gramian.detach()
            objective = objective.detach()
        return mean, {
            "pullback": pullback,
            "gramian": gramian,
            "one_step_objective": objective,
        }
