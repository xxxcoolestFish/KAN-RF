"""TD3 with a residual-action trust region and critic-disagreement risk."""

from __future__ import annotations

import numpy as np
import torch
from stable_baselines3 import TD3
from stable_baselines3.common.utils import polyak_update
from torch.nn import functional as F


class TrustRegionTD3(TD3):
    """Keep a residual policy near zero until its critics agree."""

    def __init__(
        self,
        *args,
        source_trust_coefficient: float = 0.1,
        uncertainty_coefficient: float = 0.1,
        behavior_coefficient: float = 0.0,
        adaptive_q_coefficient: float = 2.5,
        **kwargs,
    ):
        self.source_trust_coefficient = float(
            source_trust_coefficient,
        )
        self.uncertainty_coefficient = float(
            uncertainty_coefficient,
        )
        self.behavior_coefficient = float(behavior_coefficient)
        self.adaptive_q_coefficient = float(adaptive_q_coefficient)
        super().__init__(*args, **kwargs)

    def train(self, gradient_steps: int, batch_size: int = 100):
        self.policy.set_training_mode(True)
        self._update_learning_rate(
            [self.actor.optimizer, self.critic.optimizer],
        )
        actor_losses = []
        critic_losses = []
        trust_losses = []
        uncertainty_losses = []
        behavior_losses = []
        q_weights = []
        for _ in range(gradient_steps):
            self._n_updates += 1
            replay = self.replay_buffer.sample(
                batch_size, env=self._vec_normalize_env,
            )
            with torch.no_grad():
                noise = replay.actions.clone().data.normal_(
                    0, self.target_policy_noise,
                )
                noise = noise.clamp(
                    -self.target_noise_clip,
                    self.target_noise_clip,
                )
                next_action = (
                    self.actor_target(replay.next_observations) + noise
                ).clamp(-1.0, 1.0)
                target_pair = torch.cat(
                    self.critic_target(
                        replay.next_observations, next_action,
                    ),
                    dim=1,
                )
                target_q = target_pair.min(
                    dim=1, keepdim=True,
                ).values
                target_q = (
                    replay.rewards
                    + (1.0 - replay.dones) * self.gamma * target_q
                )
            current_pair = self.critic(
                replay.observations, replay.actions,
            )
            critic_loss = sum(
                F.mse_loss(current, target_q)
                for current in current_pair
            )
            critic_losses.append(float(critic_loss.detach()))
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            if self._n_updates % self.policy_delay == 0:
                action = self.actor(replay.observations)
                q_pair = self.critic(replay.observations, action)
                pessimistic = torch.minimum(q_pair[0], q_pair[1])
                trust = action.square().mean()
                uncertainty = (q_pair[0] - q_pair[1]).abs().mean()
                behavior = F.mse_loss(action, replay.actions)
                q_weight = (
                    self.adaptive_q_coefficient
                    / pessimistic.abs().mean().detach().clamp_min(1e-6)
                    if self.behavior_coefficient > 0.0
                    else torch.ones((), device=action.device)
                )
                actor_loss = (
                    -q_weight * pessimistic.mean()
                    + self.source_trust_coefficient * trust
                    + self.uncertainty_coefficient * uncertainty
                    + self.behavior_coefficient * behavior
                )
                actor_losses.append(float(actor_loss.detach()))
                trust_losses.append(float(trust.detach()))
                uncertainty_losses.append(float(uncertainty.detach()))
                behavior_losses.append(float(behavior.detach()))
                q_weights.append(float(q_weight.detach()))
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()
                polyak_update(
                    self.critic.parameters(),
                    self.critic_target.parameters(),
                    self.tau,
                )
                polyak_update(
                    self.actor.parameters(),
                    self.actor_target.parameters(),
                    self.tau,
                )
                polyak_update(
                    self.critic_batch_norm_stats,
                    self.critic_batch_norm_stats_target,
                    1.0,
                )
                polyak_update(
                    self.actor_batch_norm_stats,
                    self.actor_batch_norm_stats_target,
                    1.0,
                )
        self.logger.record(
            "train/n_updates", self._n_updates, exclude="tensorboard",
        )
        if actor_losses:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
            self.logger.record(
                "train/residual_trust", np.mean(trust_losses),
            )
            self.logger.record(
                "train/critic_disagreement",
                np.mean(uncertainty_losses),
            )
            self.logger.record(
                "train/behavior_trust", np.mean(behavior_losses),
            )
            self.logger.record(
                "train/adaptive_q_weight", np.mean(q_weights),
            )
        if critic_losses:
            self.logger.record(
                "train/critic_loss", np.mean(critic_losses),
            )
