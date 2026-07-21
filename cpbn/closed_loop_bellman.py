"""Closed-loop soft Bellman policy with a mandatory cognitive operator."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.distributions import Categorical

from cpbn.acrobot import tip_height
from cpbn.cognitive_adjoint import CorridorPotential
from cpbn.receding_tube import local_state_distance


def shift_corridor(corridor: torch.Tensor) -> torch.Tensor:
    """Advance an ordered corridor by one step while preserving its length."""
    return torch.cat([corridor[:, 1:], corridor[:, -1:]], dim=1)


class ClosedLoopBellmanActor(nn.Module):
    """A discrete-action Bellman policy used as an Oracle structure gate.

    The fixed action grid is diagnostic rather than the intended final action
    representation.  Unlike the adjoint Actor, this policy evaluates the full
    action range and performs feedback Bellman backups at every imagined state.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        corridor_horizon: int = 12,
        action_bins: int = 7,
        backup_depth: int = 2,
        macro_steps: int = 1,
        temperature: float = 0.25,
        gamma: float = 0.99,
        progress_weight: float = 1.0,
        progress_clip: float = 0.25,
        inside_reward: float = 0.08,
        success_reward: float = 3.0,
        action_penalty: float = 0.002,
        corridor_radius: float = 0.12,
    ):
        super().__init__()
        if action_bins < 2:
            raise ValueError("action_bins must be at least two")
        if backup_depth < 1:
            raise ValueError("backup_depth must be positive")
        if macro_steps < 1:
            raise ValueError("macro_steps must be positive")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        self.potential = CorridorPotential(hidden_dim, corridor_horizon)
        self.action_bins = action_bins
        self.backup_depth = backup_depth
        self.macro_steps = macro_steps
        self.gamma = gamma
        self.macro_discount = gamma ** macro_steps
        self.progress_weight = progress_weight
        self.progress_clip = progress_clip
        self.inside_reward = inside_reward
        self.success_reward = success_reward
        self.action_penalty = action_penalty
        self.corridor_radius = corridor_radius
        self.register_buffer(
            "action_grid", torch.linspace(-1.0, 1.0, action_bins),
        )
        self.register_buffer("temperature", torch.tensor(float(temperature)))

    def potential_value(self, state, corridor):
        return self.potential(state, corridor)

    def immediate_utility(self, state, action, predicted, corridor):
        target = corridor[:, 0]
        before = local_state_distance(state, target)
        after = local_state_distance(predicted, target)
        progress = (before - after).clamp(
            -self.progress_clip, self.progress_clip,
        )
        inside = (after <= self.corridor_radius).to(state.dtype)
        success = (tip_height(predicted) >= 1.0).to(state.dtype)
        return (
            self.progress_weight * progress
            + self.inside_reward * inside
            + self.success_reward * success
            - self.action_penalty * action.square().squeeze(-1)
        )

    def _candidate_transitions(self, state, corridor, cognition):
        batch = state.shape[0]
        actions = self.action_grid.to(state).view(1, -1, 1).expand(
            batch, -1, -1,
        )
        states = state.unsqueeze(1).expand(-1, self.action_bins, -1)
        corridors = corridor.unsqueeze(1).expand(
            -1, self.action_bins, -1, -1,
        )
        flat_state = states.reshape(-1, state.shape[-1])
        flat_action = actions.reshape(-1, 1)
        flat_corridor = corridors.reshape(
            -1, corridor.shape[1], corridor.shape[2],
        )
        utility = torch.zeros(
            flat_state.shape[0], dtype=state.dtype, device=state.device,
        )
        discount = 1.0
        predicted = flat_state
        for _ in range(self.macro_steps):
            predicted = cognition(predicted, flat_action)
            utility = utility + discount * self.immediate_utility(
                flat_state, flat_action, predicted, flat_corridor,
            )
            flat_state = predicted
            flat_corridor = shift_corridor(flat_corridor)
            discount *= self.gamma
        return (
            predicted,
            utility.view(batch, self.action_bins),
            flat_corridor,
        )

    def _soft_value(self, state, corridor, cognition, depth):
        if depth == 0:
            return self.potential_value(state, corridor)
        q_values = self._q_values(state, corridor, cognition, depth)
        scaled = q_values / self.temperature.to(q_values)
        return self.temperature.to(q_values) * (
            torch.logsumexp(scaled, dim=-1) - math.log(self.action_bins)
        )

    def _q_values(self, state, corridor, cognition, depth):
        predicted, utility, next_corridor = self._candidate_transitions(
            state, corridor, cognition,
        )
        future = self._soft_value(
            predicted, next_corridor, cognition, depth - 1,
        ).view(state.shape[0], self.action_bins)
        return utility + self.macro_discount * future

    def action_logits(self, state, corridor, cognition):
        q_values = self._q_values(
            state, corridor, cognition, self.backup_depth,
        )
        return q_values / self.temperature.to(q_values)

    @torch.no_grad()
    def precompute_depth_two_tree(self, state, corridor, cognition):
        """Cache the fixed cognitive tree used by a depth-two PPO update."""
        if self.backup_depth != 2:
            raise ValueError("tree caching currently requires backup_depth=2")
        first_state, first_utility, first_corridor = (
            self._candidate_transitions(state, corridor, cognition)
        )
        second_state, second_utility, second_corridor = (
            self._candidate_transitions(
                first_state, first_corridor, cognition,
            )
        )
        batch = state.shape[0]
        return {
            "root_utility": first_utility,
            "child_utility": second_utility.view(
                batch, self.action_bins, self.action_bins,
            ),
            "leaf_state": second_state.view(
                batch, self.action_bins, self.action_bins, -1,
            ),
            "leaf_corridor": second_corridor.view(
                batch, self.action_bins, self.action_bins,
                corridor.shape[1], corridor.shape[2],
            ),
        }

    def logits_from_depth_two_tree(self, tree):
        """Re-evaluate only the learned potential on a cached tree."""
        root_utility = tree["root_utility"]
        batch = root_utility.shape[0]
        leaf_state = tree["leaf_state"]
        leaf_corridor = tree["leaf_corridor"]
        leaf_value = self.potential_value(
            leaf_state.reshape(-1, leaf_state.shape[-1]),
            leaf_corridor.reshape(
                -1, leaf_corridor.shape[-2], leaf_corridor.shape[-1],
            ),
        ).view(batch, self.action_bins, self.action_bins)
        child_q = (
            tree["child_utility"] + self.macro_discount * leaf_value
        )
        temperature = self.temperature.to(child_q)
        child_value = temperature * (
            torch.logsumexp(child_q / temperature, dim=-1)
            - math.log(self.action_bins)
        )
        root_q = root_utility + self.macro_discount * child_value
        return root_q / temperature

    def evaluate_precomputed(self, tree, action):
        distribution = Categorical(
            logits=self.logits_from_depth_two_tree(tree),
        )
        distance = (
            action.squeeze(-1).unsqueeze(-1)
            - self.action_grid.to(action).view(1, -1)
        ).abs()
        index = distance.argmin(dim=-1)
        return distribution.log_prob(index), distribution.entropy()

    def distribution(self, state, corridor, cognition):
        return Categorical(logits=self.action_logits(state, corridor, cognition))

    def sample(self, state, corridor, cognition, deterministic=False):
        distribution = self.distribution(state, corridor, cognition)
        index = distribution.logits.argmax(dim=-1) if deterministic else distribution.sample()
        action = self.action_grid.to(state)[index].unsqueeze(-1)
        return action, distribution.log_prob(index)

    def evaluate(self, state, corridor, cognition, action):
        distribution = self.distribution(state, corridor, cognition)
        distance = (
            action.squeeze(-1).unsqueeze(-1)
            - self.action_grid.to(action).view(1, -1)
        ).abs()
        index = distance.argmin(dim=-1)
        return distribution.log_prob(index), distribution.entropy()

    @torch.no_grad()
    def diagnostics(self, state, corridor, cognition):
        logits = self.action_logits(state, corridor, cognition)
        index = logits.argmax(dim=-1)
        return {
            "mean_logit_range": float(
                (logits.max(dim=-1).values - logits.min(dim=-1).values).mean()
            ),
            "boundary_action_rate": float(
                ((index == 0) | (index == self.action_bins - 1)).float().mean()
            ),
        }
