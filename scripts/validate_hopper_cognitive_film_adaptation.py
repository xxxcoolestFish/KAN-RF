"""Online Hopper adaptation by ProtoKAN-conditioned source-Actor FiLM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.utils import explained_variance
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from torch.nn import functional as F

from cpbn.generic_affine_kan import AffineKANContext
from cpbn.global_mechanism_kan import (
    GlobalMechanismKANDynamics,
    RecursiveGlobalMechanismEstimator,
)
from cpbn.hopper_cognitive_modulation import (
    HopperCognitiveFiLMExtractor,
    HopperCognitiveLoRAExtractor,
    HopperMechanismAffineExtractor,
)
from scripts.prescreen_hopper_physics_shifts import SHIFTS, make_shifted_env
from scripts.validate_hopper_joint_online_adaptation import (
    FrozenSourcePolicy,
    cognition_warmup,
    load_cognition,
)
from scripts.validate_hopper_low_rank_global_context_online import (
    FrozenContextEstimator,
    LowRankGlobalContextEstimator,
    ValidatedResidualModeEstimator,
    discover_modes,
)


def configure_online_cognition(
    mode,
    basis,
    source_context,
    full_estimator,
    args,
):
    if mode == "full":
        return full_estimator
    discovered_context = full_estimator.context()
    if mode == "frozen":
        return FrozenContextEstimator(discovered_context)
    modes, coordinates, retained_energy = discover_modes(
        source_context,
        discovered_context,
        args.cognition_rank,
    )
    print(
        {
            "stage": "configure_online_cognition",
            "mode": mode,
            "rank": args.cognition_rank,
            "discovered_mode_energy": retained_energy,
        },
        flush=True,
    )
    low_rank = LowRankGlobalContextEstimator(
        basis,
        source_context,
        modes,
        coordinates,
        ridge=args.cognition_coordinate_ridge,
        forgetting_factor=args.cognition_forgetting_factor,
    )
    if mode == "fixed_low_rank":
        return low_rank
    return ValidatedResidualModeEstimator(
        low_rank,
        rank=args.cognition_rank,
        novelty_threshold=args.cognition_refresh_threshold,
        residual_ridge=args.cognition_residual_ridge,
        mode_step=args.cognition_mode_step,
        validation_fraction=args.cognition_validation_fraction,
        functional_proximal_weight=(
            args.cognition_functional_proximal_weight
        ),
        posterior_transport=args.cognition_posterior_transport,
    )


class CognitiveFiLMPolicy(ActorCriticPolicy):
    def __init__(
        self,
        *args,
        source_policy_state,
        source_observation_mean,
        source_observation_variance,
        modulation_scale=0.25,
        operator_dimension=44,
        **kwargs,
    ):
        self._source_policy_state = {
            key: value.detach().cpu()
            for key, value in source_policy_state.items()
        }
        self._source_observation_mean = (
            source_observation_mean.detach().cpu()
        )
        self._source_observation_variance = (
            source_observation_variance.detach().cpu()
        )
        self._modulation_scale = modulation_scale
        self._operator_dimension = operator_dimension
        super().__init__(*args, **kwargs)
        self.mlp_extractor.load_source_weights(
            self._source_policy_state,
        )
        # ActorCriticPolicy orthogonally initializes custom submodules after
        # construction, so restore zero-output modulation explicitly.
        self.mlp_extractor.reset_adapter()
        with torch.no_grad():
            self.action_net.weight.copy_(
                self._source_policy_state["action_net.weight"],
            )
            self.action_net.bias.copy_(
                self._source_policy_state["action_net.bias"],
            )
            self.value_net.weight.copy_(
                self._source_policy_state["value_net.weight"],
            )
            self.value_net.bias.copy_(
                self._source_policy_state["value_net.bias"],
            )
            self.log_std.copy_(
                self._source_policy_state["log_std"],
            )
        self.action_net.requires_grad_(False)

    def _build_mlp_extractor(self):
        self.mlp_extractor = HopperCognitiveFiLMExtractor(
            self._source_policy_state,
            self._source_observation_mean,
            self._source_observation_variance,
            operator_dim=self._operator_dimension,
            modulation_scale=self._modulation_scale,
        )


class CognitiveLoRAPolicy(ActorCriticPolicy):
    def __init__(
        self,
        *args,
        source_policy_state,
        source_observation_mean,
        source_observation_variance,
        modulation_scale=0.25,
        cognitive_lora_rank=4,
        operator_dimension=44,
        **kwargs,
    ):
        self._source_policy_state = {
            key: value.detach().cpu()
            for key, value in source_policy_state.items()
        }
        self._source_observation_mean = (
            source_observation_mean.detach().cpu()
        )
        self._source_observation_variance = (
            source_observation_variance.detach().cpu()
        )
        self._modulation_scale = modulation_scale
        self._cognitive_lora_rank = cognitive_lora_rank
        self._operator_dimension = operator_dimension
        super().__init__(*args, **kwargs)
        self.mlp_extractor.load_source_weights(
            self._source_policy_state,
        )
        self.mlp_extractor.reset_adapter()
        with torch.no_grad():
            self.action_net.weight.zero_()
            self.action_net.weight[:, :64].copy_(
                self._source_policy_state["action_net.weight"],
            )
            self.action_net.weight[:, 64:].copy_(
                torch.eye(
                    3,
                    dtype=self.action_net.weight.dtype,
                    device=self.action_net.weight.device,
                )
            )
            self.action_net.bias.copy_(
                self._source_policy_state["action_net.bias"],
            )
            self.value_net.weight.copy_(
                self._source_policy_state["value_net.weight"],
            )
            self.value_net.bias.copy_(
                self._source_policy_state["value_net.bias"],
            )
            self.log_std.copy_(
                self._source_policy_state["log_std"],
            )
        self.action_net.requires_grad_(False)

    def _build_mlp_extractor(self):
        self.mlp_extractor = HopperCognitiveLoRAExtractor(
            self._source_policy_state,
            self._source_observation_mean,
            self._source_observation_variance,
            operator_dim=self._operator_dimension,
            rank=self._cognitive_lora_rank,
            modulation_scale=self._modulation_scale,
        )


class MechanismAffinePolicy(CognitiveLoRAPolicy):
    def __init__(
        self,
        *args,
        mechanism_policy_shifts=None,
        **kwargs,
    ):
        self._mechanism_policy_shifts = mechanism_policy_shifts
        super().__init__(*args, **kwargs)
        if mechanism_policy_shifts is not None:
            self.mlp_extractor.load_policy_shifts(
                {
                    key: value.to(self.device)
                    for key, value in mechanism_policy_shifts.items()
                }
            )

    def _build_mlp_extractor(self):
        self.mlp_extractor = HopperMechanismAffineExtractor(
            self._source_policy_state,
            self._source_observation_mean,
            self._source_observation_variance,
            mechanism_dim=self._operator_dimension,
            modulation_scale=self._modulation_scale,
        )


def build_mechanism_policy_shifts(
    source_state,
    target_model_paths,
    training_latents,
    latent_scale,
    modulation_scale,
):
    """Regress past successful Actor changes onto cognitive coordinates."""
    target_states = [
        PPO.load(path, device="cpu").policy.state_dict()
        for path in target_model_paths
    ]
    coordinates = training_latents / latent_scale.unsqueeze(0)
    layer_keys = {
        "first_weight_shift": "mlp_extractor.policy_net.0.weight",
        "first_bias_shift": "mlp_extractor.policy_net.0.bias",
        "second_weight_shift": "mlp_extractor.policy_net.2.weight",
        "second_bias_shift": "mlp_extractor.policy_net.2.bias",
        "action_weight_shift": "action_net.weight",
        "action_bias_shift": "action_net.bias",
    }
    shifts = {}
    for output_name, state_key in layer_keys.items():
        source = source_state[state_key].detach().cpu()
        differences = torch.stack(
            [
                state[state_key].detach().cpu() - source
                for state in target_states
            ],
            dim=0,
        )
        solution = torch.linalg.lstsq(
            coordinates,
            differences.flatten(start_dim=1),
        ).solution
        shifts[output_name] = (
            solution.reshape(
                coordinates.shape[1], *differences.shape[1:],
            )
            / modulation_scale
        )
    return shifts


class SourceAnchoredCognitivePPO(PPO):
    """PPO with a fixed-source behavior trust region.

    The ordinary PPO clip only limits one update relative to the immediately
    preceding policy.  This additional KL keeps the cumulative cognitive
    adaptation close to the reliable frozen source behavior.
    """

    def __init__(
        self,
        *args,
        source_kl_coefficient=0.0,
        **kwargs,
    ):
        self.source_kl_coefficient = float(source_kl_coefficient)
        super().__init__(*args, **kwargs)
        self._fixed_source_log_std = (
            self.policy._source_policy_state["log_std"]
            .to(self.device)
            .detach()
        )

    def source_behavior_kl(self, observation):
        current_latent = self.policy.mlp_extractor.forward_actor(
            observation,
        )
        current_mean = self.policy.action_net(current_latent)
        source_observation = observation.clone()
        source_observation[..., 11:] = 0.0
        source_latent = self.policy.mlp_extractor.forward_actor(
            source_observation,
        )
        source_mean = self.policy.action_net(source_latent).detach()
        current_log_std = self.policy.log_std.expand_as(current_mean)
        source_log_std = self._fixed_source_log_std.expand_as(
            current_mean,
        )
        source_variance = torch.exp(2.0 * source_log_std)
        kl = (
            source_log_std
            - current_log_std
            + (
                torch.exp(2.0 * current_log_std)
                + (current_mean - source_mean).square()
            )
            / (2.0 * source_variance)
            - 0.5
        )
        return kl.sum(dim=-1).mean()

    def train(self):
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(
            self._current_progress_remaining,
        )
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(
                self._current_progress_remaining,
            )

        entropy_losses = []
        policy_losses = []
        value_losses = []
        source_kls = []
        clip_fractions = []
        approximate_kls = []
        continue_training = True
        loss = torch.zeros((), device=self.device)
        for epoch in range(self.n_epochs):
            epoch_kls = []
            for rollout_data in self.rollout_buffer.get(
                self.batch_size,
            ):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = actions.long().flatten()
                values, log_probability, entropy = (
                    self.policy.evaluate_actions(
                        rollout_data.observations, actions,
                    )
                )
                values = values.flatten()
                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (
                        advantages - advantages.mean()
                    ) / (advantages.std() + 1e-8)
                ratio = torch.exp(
                    log_probability - rollout_data.old_log_prob,
                )
                unclipped = advantages * ratio
                clipped = advantages * torch.clamp(
                    ratio,
                    1.0 - clip_range,
                    1.0 + clip_range,
                )
                policy_loss = -torch.min(
                    unclipped, clipped,
                ).mean()
                policy_losses.append(float(policy_loss.detach()))
                clip_fractions.append(float(
                    (
                        torch.abs(ratio - 1.0) > clip_range
                    ).float().mean(),
                ))

                if self.clip_range_vf is None:
                    predicted_values = values
                else:
                    predicted_values = (
                        rollout_data.old_values
                        + torch.clamp(
                            values - rollout_data.old_values,
                            -clip_range_vf,
                            clip_range_vf,
                        )
                    )
                value_loss = F.mse_loss(
                    rollout_data.returns, predicted_values,
                )
                value_losses.append(float(value_loss.detach()))
                entropy_loss = (
                    -torch.mean(-log_probability)
                    if entropy is None
                    else -torch.mean(entropy)
                )
                entropy_losses.append(float(entropy_loss.detach()))
                source_kl = self.source_behavior_kl(
                    rollout_data.observations,
                )
                source_kls.append(float(source_kl.detach()))
                loss = (
                    policy_loss
                    + self.ent_coef * entropy_loss
                    + self.vf_coef * value_loss
                    + self.source_kl_coefficient * source_kl
                )

                with torch.no_grad():
                    log_ratio = (
                        log_probability
                        - rollout_data.old_log_prob
                    )
                    approximate_kl = torch.mean(
                        (torch.exp(log_ratio) - 1.0) - log_ratio,
                    ).cpu().numpy()
                    epoch_kls.append(approximate_kl)
                    approximate_kls.append(float(approximate_kl))
                if (
                    self.target_kl is not None
                    and approximate_kl > 1.5 * self.target_kl
                ):
                    continue_training = False
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm,
                )
                self.policy.optimizer.step()
            self._n_updates += 1
            if not continue_training:
                break

        self.logger.record(
            "train/entropy_loss", np.mean(entropy_losses),
        )
        self.logger.record(
            "train/policy_gradient_loss", np.mean(policy_losses),
        )
        self.logger.record(
            "train/value_loss", np.mean(value_losses),
        )
        self.logger.record(
            "train/approx_kl", np.mean(approximate_kls),
        )
        self.logger.record(
            "train/source_behavior_kl", np.mean(source_kls),
        )
        self.logger.record(
            "train/clip_fraction", np.mean(clip_fractions),
        )
        self.logger.record("train/loss", float(loss.detach()))
        self.logger.record(
            "train/explained_variance",
            explained_variance(
                self.rollout_buffer.values.flatten(),
                self.rollout_buffer.returns.flatten(),
            ),
        )
        self.logger.record(
            "train/std",
            torch.exp(self.policy.log_std).mean().item(),
        )
        self.logger.record(
            "train/n_updates", self._n_updates,
            exclude="tensorboard",
        )
        self.logger.record("train/clip_range", clip_range)


@torch.no_grad()
def operator_observation(
    observation,
    basis,
    source_context,
    target_context,
    delta_scale,
    representation="raw",
    canonical_damping=1e-3,
):
    state = torch.as_tensor(
        observation,
        dtype=torch.float32,
        device=basis.grid.device,
    ).unsqueeze(0)
    source_drift, source_gain = source_context.drift_and_gain(
        basis, state,
    )
    target_drift, target_gain = target_context.drift_and_gain(
        basis, state,
    )
    if representation == "canonical":
        source_gain_normalized = (
            source_gain / delta_scale.unsqueeze(-1)
        )
        target_gain_normalized = (
            target_gain / delta_scale.unsqueeze(-1)
        )
        drift_gap = (
            (source_drift - target_drift) / delta_scale
        )
        target_transpose = target_gain_normalized.transpose(
            -2, -1,
        )
        identity = torch.eye(
            target_gain.shape[-1],
            dtype=target_gain.dtype,
            device=target_gain.device,
        ).unsqueeze(0)
        gram = (
            target_transpose @ target_gain_normalized
            + canonical_damping * identity
        )
        action_transform = torch.linalg.solve(
            gram,
            target_transpose @ source_gain_normalized,
        )
        drift_compensation = torch.linalg.solve(
            gram,
            target_transpose @ drift_gap.unsqueeze(-1),
        ).squeeze(-1)
        gain_residual = (
            target_gain_normalized @ action_transform
            - source_gain_normalized
        )
        drift_residual = (
            target_gain_normalized
            @ drift_compensation.unsqueeze(-1)
        ).squeeze(-1) - drift_gap
        gain_residual_norm = torch.sqrt(
            gain_residual.square().mean(dim=-2)
        )
        drift_residual_norm = torch.sqrt(
            drift_residual.square().mean(dim=-1, keepdim=True)
        )
        signature = torch.cat(
            (
                (action_transform - identity).flatten(
                    start_dim=-2
                ),
                drift_compensation,
                gain_residual_norm,
                drift_residual_norm,
            ),
            dim=-1,
        ).clamp(-10.0, 10.0)
        return torch.cat((state, signature), dim=-1)[
            0
        ].cpu().numpy().astype(np.float32)
    drift_gap = (
        (target_drift - source_drift) / delta_scale
    ).clamp(-10.0, 10.0)
    gain_gap = (
        (target_gain - source_gain)
        / delta_scale.unsqueeze(-1)
    ).clamp(-10.0, 10.0)
    return torch.cat(
        (state, drift_gap, gain_gap.flatten(start_dim=-2)),
        dim=-1,
    )[0].cpu().numpy().astype(np.float32)


@torch.no_grad()
def trajectory_canonical_signature(
    states,
    basis,
    source_context,
    target_context,
    delta_scale,
    damping,
):
    state = torch.as_tensor(
        states,
        dtype=torch.float32,
        device=basis.grid.device,
    )
    source_drift, source_gain = source_context.drift_and_gain(
        basis, state,
    )
    target_drift, target_gain = target_context.drift_and_gain(
        basis, state,
    )
    source_gain = source_gain / delta_scale.unsqueeze(-1)
    target_gain = target_gain / delta_scale.unsqueeze(-1)
    drift_gap = (source_drift - target_drift) / delta_scale
    target_transpose = target_gain.transpose(-2, -1)
    identity = torch.eye(
        target_gain.shape[-1],
        dtype=target_gain.dtype,
        device=target_gain.device,
    )
    gram = torch.einsum(
        "nai,naj->ij", target_gain, target_gain,
    ) + damping * state.shape[0] * identity
    action_rhs = torch.einsum(
        "nai,naj->ij", target_gain, source_gain,
    )
    drift_rhs = torch.einsum(
        "nai,na->i", target_gain, drift_gap,
    )
    action_transform = torch.linalg.solve(gram, action_rhs)
    drift_compensation = torch.linalg.solve(gram, drift_rhs)
    gain_residual = (
        target_gain @ action_transform - source_gain
    )
    drift_residual = (
        target_gain
        @ drift_compensation.unsqueeze(-1)
    ).squeeze(-1) - drift_gap
    gain_residual_norm = torch.sqrt(
        gain_residual.square().mean(dim=(0, 1))
    )
    drift_residual_norm = torch.sqrt(
        drift_residual.square().mean()
    ).reshape(1)
    return torch.cat(
        (
            (action_transform - identity).flatten(),
            drift_compensation,
            gain_residual_norm,
            drift_residual_norm,
        )
    ).clamp(-10.0, 10.0).cpu().numpy().astype(np.float32)


def operator_tensor(
    state,
    basis,
    source_context,
    target_context,
    delta_scale,
):
    source_drift, source_gain = source_context.drift_and_gain(
        basis, state,
    )
    target_drift, target_gain = target_context.drift_and_gain(
        basis, state,
    )
    drift_gap = (
        (target_drift - source_drift) / delta_scale
    ).clamp(-10.0, 10.0)
    gain_gap = (
        (target_gain - source_gain)
        / delta_scale.unsqueeze(-1)
    ).clamp(-10.0, 10.0)
    return torch.cat(
        (state, drift_gap, gain_gap.flatten(start_dim=-2)),
        dim=-1,
    )


def straight_through_clip(action):
    clipped = action.clamp(-1.0, 1.0)
    return action + (clipped - action).detach()


def finite_horizon_cognitive_pretrain(
    model,
    replay_states,
    basis,
    source_context,
    target_context,
    delta_scale,
    args,
):
    if args.model_pretrain_steps <= 0:
        return []
    device = basis.grid.device
    states = torch.as_tensor(
        replay_states, dtype=torch.float32, device=device,
    )
    adapter = model.policy.mlp_extractor.adapter
    optimizer = torch.optim.Adam(
        adapter.parameters(), lr=args.model_learning_rate,
    )
    generator = torch.Generator(device=device).manual_seed(
        args.seed + 17,
    )
    records = []
    for step in range(1, args.model_pretrain_steps + 1):
        indices = torch.randint(
            states.shape[0],
            (min(args.model_batch_size, states.shape[0]),),
            generator=generator,
            device=device,
        )
        target_state = states[indices]
        source_state = target_state.detach().clone()
        trajectory_loss = torch.zeros((), device=device)
        anchor_loss = torch.zeros((), device=device)
        discount = 1.0
        for _ in range(args.model_horizon):
            target_input = operator_tensor(
                target_state,
                basis,
                source_context,
                target_context,
                delta_scale,
            )
            target_action = straight_through_clip(
                model.policy._predict(
                    target_input, deterministic=True,
                )
            )
            zero_operator = torch.zeros(
                target_state.shape[0], 44, device=device,
            )
            nominal_target_action = straight_through_clip(
                model.policy._predict(
                    torch.cat((target_state, zero_operator), dim=-1),
                    deterministic=True,
                )
            )
            target_delta = target_context.acceleration(
                basis,
                target_state,
                target_action - nominal_target_action,
            )

            source_input = torch.cat(
                (source_state, zero_operator), dim=-1,
            )
            source_action = straight_through_clip(
                model.policy._predict(
                    source_input, deterministic=True,
                )
            )
            source_delta = source_context.acceleration(
                basis,
                source_state,
                torch.zeros_like(source_action),
            )
            target_state = target_state + target_delta
            source_state = source_state + source_delta
            normalized_gap = (
                (target_state - source_state) / delta_scale
            ).clamp(-20.0, 20.0)
            trajectory_loss = (
                trajectory_loss
                + discount * normalized_gap.square().mean()
            )
            anchor_loss = (
                anchor_loss
                + discount
                * (target_action - nominal_target_action).square().mean()
            )
            discount *= args.gamma
        loss = (
            trajectory_loss
            + args.model_anchor_weight * anchor_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            adapter.parameters(), args.model_gradient_clip,
        )
        optimizer.step()
        if step == 1 or step % args.model_report_every == 0:
            record = {
                "gradient_step": step,
                "trajectory_loss": float(trajectory_loss.detach()),
                "anchor_loss": float(anchor_loss.detach()),
            }
            records.append(record)
            print({"model_pretrain": record}, flush=True)
    return records


class CognitiveFiLMHopper(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        source_policy,
        basis,
        source_context,
        estimator,
        delta_scale,
        args,
        *,
        update_cognition,
        seed_offset,
        signature_states=None,
    ):
        super().__init__()
        if not getattr(basis, "policy_centered", False):
            raise ValueError("FiLM adaptation requires policy-centered cognition")
        self.environment = make_shifted_env(
            SHIFTS[args.target], args.seed + seed_offset,
        )()
        self.source_policy = source_policy
        self.basis = basis
        self.source_context = source_context
        self.estimator = estimator
        self.cognitive_context = estimator.context()
        self.target_context = self.cognitive_context
        self.delta_scale = delta_scale
        self.args = args
        self.update_cognition = update_cognition
        self.action_space = gym.spaces.Box(
            -1.0, 1.0, shape=(3,), dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(
            -np.inf,
            np.inf,
            shape=(11 + args.operator_dimension,),
            dtype=np.float32,
        )
        self.raw_observation = None
        self.buffer = []
        self.cognition_transfer_rates = []
        self.signature_states = (
            [
                state.copy()
                for state in np.asarray(signature_states)[
                    -args.trajectory_signature_window:
                ]
            ]
            if signature_states is not None
            else []
        )
        self.global_operator_signature = None
        self.mechanism_latent_scale = getattr(
            args, "mechanism_latent_scale", None,
        )
        if (
            args.operator_representation
            == "trajectory_canonical"
            and self.signature_states
        ):
            self.refresh_global_operator_signature()
        elif args.operator_representation == "mechanism":
            self.refresh_mechanism_signature()

    def refresh_global_operator_signature(self):
        self.global_operator_signature = (
            trajectory_canonical_signature(
                np.asarray(self.signature_states),
                self.basis,
                self.source_context,
                self.target_context,
                self.delta_scale,
                self.args.canonical_operator_damping,
            )
        )

    def refresh_mechanism_signature(self):
        latent = self.estimator.latent()
        scale = torch.as_tensor(
            self.mechanism_latent_scale,
            dtype=latent.dtype,
            device=latent.device,
        )
        self.global_operator_signature = (
            (latent / scale).clamp(-10.0, 10.0)
            .cpu()
            .numpy()
            .astype(np.float32)
        )

    def transformed_observation(self, observation):
        if self.args.operator_representation in (
            "trajectory_canonical",
            "mechanism",
        ):
            if self.global_operator_signature is None:
                signature = np.zeros(
                    self.args.operator_dimension,
                    dtype=np.float32,
                )
            else:
                signature = self.global_operator_signature
            return np.concatenate(
                (
                    np.asarray(observation, dtype=np.float32),
                    signature,
                )
            ).astype(np.float32)
        return operator_observation(
            observation,
            self.basis,
            self.source_context,
            self.target_context,
            self.delta_scale,
            self.args.operator_representation,
            self.args.canonical_operator_damping,
        )

    def reset(self, *, seed=None, options=None):
        observation, info = self.environment.reset(
            seed=seed, options=options,
        )
        self.raw_observation = observation
        return self.transformed_observation(observation), info

    def step(self, action):
        action = np.clip(
            np.asarray(action, dtype=np.float32), -1.0, 1.0,
        )
        state_tensor = torch.as_tensor(
            self.raw_observation,
            dtype=torch.float32,
            device=self.basis.grid.device,
        ).unsqueeze(0)
        zero = torch.zeros(
            1, 3, dtype=torch.float32, device=self.basis.grid.device,
        )
        with torch.no_grad():
            source_reference = self.source_context.acceleration(
                self.basis, state_tensor, zero,
            )[0].cpu().numpy()
            nominal_action = self.source_policy.action(
                self.raw_observation,
            ).cpu().numpy()
        following, task_reward, terminated, truncated, info = (
            self.environment.step(action)
        )
        actual_delta = following - self.raw_observation
        normalized_error = (
            (actual_delta - source_reference)
            / self.delta_scale.cpu().numpy()
        )
        tracking_cost = float(np.mean(np.square(normalized_error)))
        if self.args.decision_objective == "tracking":
            reward = -tracking_cost
        elif self.args.decision_objective == "task":
            reward = float(task_reward)
        else:
            reward = (
                float(task_reward)
                - self.args.tracking_weight * tracking_cost
            )
        if self.update_cognition:
            self.buffer.append((
                self.raw_observation.copy(),
                (action - nominal_action).copy(),
                actual_delta.copy(),
            ))
            if len(self.buffer) >= self.args.cognition_batch:
                state, innovation, delta = zip(*self.buffer)
                self.signature_states.extend(
                    item.copy() for item in state
                )
                self.signature_states = self.signature_states[
                    -self.args.trajectory_signature_window:
                ]
                self.estimator.update(
                    torch.as_tensor(
                        np.asarray(state),
                        dtype=torch.float32,
                        device=self.basis.grid.device,
                    ),
                    torch.as_tensor(
                        np.asarray(innovation),
                        dtype=torch.float32,
                        device=self.basis.grid.device,
                    ),
                    torch.as_tensor(
                        np.asarray(delta),
                        dtype=torch.float32,
                        device=self.basis.grid.device,
                    ),
                )
                self.cognitive_context = self.estimator.context()
                if self.args.cognition_transfer_mode == "confidence":
                    confidence = float(
                        getattr(
                            self.estimator,
                            "last_confidence",
                            0.0,
                        )
                    )
                    tau = confidence
                else:
                    confidence = None
                    tau = self.args.cognition_actor_context_tau
                tau = float(np.clip(tau, 0.0, 1.0))
                self.cognition_transfer_rates.append(tau)
                self.target_context = AffineKANContext(
                    self.target_context.coefficients
                    + tau
                    * (
                        self.cognitive_context.coefficients
                        - self.target_context.coefficients
                    )
                )
                if (
                    self.args.operator_representation
                    == "trajectory_canonical"
                ):
                    self.refresh_global_operator_signature()
                elif self.args.operator_representation == "mechanism":
                    self.refresh_mechanism_signature()
                print(
                    {
                        "stage": "cognition_transfer",
                        "tau": tau,
                        "confidence": confidence,
                        "novelty": getattr(
                            self.estimator,
                            "last_novelty",
                            None,
                        ),
                        "refresh_count": getattr(
                            self.estimator,
                            "refresh_count",
                            None,
                        ),
                    },
                    flush=True,
                )
                self.buffer.clear()
        self.raw_observation = following
        info["task_reward"] = float(task_reward)
        info["tracking_cost"] = tracking_cost
        return (
            self.transformed_observation(following),
            reward,
            terminated,
            truncated,
            info,
        )

    def close(self):
        self.environment.close()


@torch.no_grad()
def evaluate(
    model,
    basis,
    source_context,
    target_context,
    delta_scale,
    args,
    global_operator_signature=None,
):
    environment = make_shifted_env(
        SHIFTS[args.target], args.seed + 10000,
    )()
    returns, lengths = [], []
    healthy = 0
    actions = []
    raw_actions = []
    operator_magnitudes = []
    for episode in range(args.evaluation_episodes):
        observation, _ = environment.reset(
            seed=args.seed + 10000 + episode,
        )
        total = 0.0
        length = 0
        while True:
            if args.operator_representation in (
                "trajectory_canonical",
                "mechanism",
            ):
                if global_operator_signature is None:
                    global_operator_signature = np.zeros(
                        args.operator_dimension,
                        dtype=np.float32,
                    )
                transformed = np.concatenate(
                    (
                        np.asarray(observation, dtype=np.float32),
                        global_operator_signature,
                    )
                ).astype(np.float32)
            else:
                transformed = operator_observation(
                    observation,
                    basis,
                    source_context,
                    target_context,
                    delta_scale,
                    args.operator_representation,
                    args.canonical_operator_damping,
                )
            operator_magnitudes.extend(np.abs(transformed[11:]))
            policy_observation, _ = model.policy.obs_to_tensor(transformed)
            raw_action = (
                model.policy._predict(
                    policy_observation, deterministic=True,
                )[0]
                .detach()
                .cpu()
                .numpy()
            )
            action = np.clip(raw_action, -1.0, 1.0)
            observation, reward, terminated, truncated, _ = (
                environment.step(action)
            )
            total += float(reward)
            length += 1
            actions.append(np.abs(action))
            raw_actions.append(np.abs(raw_action))
            if terminated or truncated:
                healthy += int(truncated and not terminated)
                break
        returns.append(total)
        lengths.append(length)
    environment.close()
    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_episode_length": float(np.mean(lengths)),
        "healthy_completion_rate": healthy / args.evaluation_episodes,
        "action_abs_p95": float(np.quantile(actions, 0.95)),
        "raw_action_abs_p95": float(np.quantile(raw_actions, 0.95)),
        "action_clip_fraction": float(
            np.mean(np.asarray(raw_actions) >= 1.0)
        ),
        "operator_abs_p95": float(
            np.quantile(operator_magnitudes, 0.95)
        ),
    }


class RecoveryCallback(BaseCallback):
    def __init__(self, train_environment, components, args):
        super().__init__(verbose=0)
        self.train_environment = train_environment
        self.components = components
        self.args = args
        self.history = []
        self.next_evaluation = args.evaluate_every

    def _on_step(self):
        if self.num_timesteps < self.next_evaluation:
            return True
        basis, source_context, delta_scale = self.components
        metrics = evaluate(
            self.model,
            basis,
            source_context,
            self.train_environment.target_context,
            delta_scale,
            self.args,
            self.train_environment.global_operator_signature,
        )
        record = {
            "target_transitions": (
                self.args.cognition_warmup + self.num_timesteps
            ),
            **metrics,
        }
        self.history.append(record)
        print(record, flush=True)
        self.next_evaluation += self.args.evaluate_every
        return True


def main(args):
    args.operator_dimension = (
        3
        if args.operator_representation == "mechanism"
        else
        16
        if args.operator_representation
        in ("canonical", "trajectory_canonical")
        else 44
    )
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    source_policy = FrozenSourcePolicy(
        args.source_model, args.source_norm, device, args.seed,
    )
    basis, source_context, estimator, delta_scale = load_cognition(
        args, device,
    )
    if args.operator_representation == "mechanism":
        payload = torch.load(
            args.mechanism_checkpoint,
            map_location=device,
            weights_only=True,
        )
        checkpoint_source = payload[
            "source_coefficients"
        ].to(device)
        source_gap = float(
            (checkpoint_source - source_context.coefficients).abs().max()
        )
        if source_gap > 1e-5:
            raise ValueError(
                "mechanism checkpoint and cognition source disagree: "
                f"max gap={source_gap}"
            )
        mechanism_model = GlobalMechanismKANDynamics(
            source_context,
            payload["mechanisms"].to(device),
        )
        estimator = RecursiveGlobalMechanismEstimator(
            mechanism_model,
            basis,
            delta_scale,
            ridge=args.mechanism_latent_ridge,
            forgetting_factor=args.mechanism_forgetting_factor,
        )
        args.mechanism_latent_scale = (
            payload["latent_scale"].cpu().tolist()
        )
        mechanism_training_latents = payload[
            "training_latents"
        ].cpu()
        mechanism_latent_scale = payload["latent_scale"].cpu()
    else:
        mechanism_training_latents = None
        mechanism_latent_scale = None
    replay_states = cognition_warmup(
        source_policy, basis, estimator, args, device,
    )
    if args.operator_representation != "mechanism":
        estimator = configure_online_cognition(
            args.cognition_update_mode,
            basis,
            source_context,
            estimator,
            args,
        )
    environment = CognitiveFiLMHopper(
        source_policy,
        basis,
        source_context,
        estimator,
        delta_scale,
        args,
        update_cognition=True,
        seed_offset=500,
        signature_states=replay_states,
    )
    vector = DummyVecEnv([lambda: environment])
    normalized = VecNormalize(
        vector,
        training=True,
        norm_obs=False,
        norm_reward=True,
        gamma=args.gamma,
    )
    source_state = source_policy.model.policy.state_dict()
    mechanism_policy_shifts = None
    if (
        args.decision_adapter == "mechanism_affine"
        and args.initialize_policy_mechanisms
    ):
        mechanism_policy_shifts = build_mechanism_policy_shifts(
            source_state,
            args.mechanism_actor_models,
            mechanism_training_latents,
            mechanism_latent_scale,
            args.modulation_scale,
        )
    algorithm = (
        SourceAnchoredCognitivePPO
        if args.source_kl_coefficient > 0.0
        else PPO
    )
    policy_class = {
        "film": CognitiveFiLMPolicy,
        "lora": CognitiveLoRAPolicy,
        "mechanism_affine": MechanismAffinePolicy,
    }[args.decision_adapter]
    model = algorithm(
        policy_class,
        normalized,
        learning_rate=args.learning_rate,
        n_steps=args.rollout_steps,
        batch_size=args.minibatch_size,
        n_epochs=args.update_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_ratio,
        target_kl=args.update_target_kl,
        ent_coef=args.entropy_coefficient,
        **(
            {"source_kl_coefficient": args.source_kl_coefficient}
            if args.source_kl_coefficient > 0.0
            else {}
        ),
        seed=args.seed,
        device=device,
        policy_kwargs={
            "source_policy_state": source_state,
            "source_observation_mean": source_policy.mean,
            "source_observation_variance": source_policy.variance,
            "modulation_scale": args.modulation_scale,
            "operator_dimension": args.operator_dimension,
            **(
                {"cognitive_lora_rank": args.cognitive_lora_rank}
                if args.decision_adapter == "lora"
                else {}
            ),
            **(
                {
                    "mechanism_policy_shifts": (
                        mechanism_policy_shifts
                    )
                }
                if args.decision_adapter == "mechanism_affine"
                else {}
            ),
        },
        verbose=0,
    )
    initial = {
        "target_transitions": args.cognition_warmup,
        "stage": (
            "mechanism_policy_initialization"
            if mechanism_policy_shifts is not None
            else "zero_modulation"
        ),
        **evaluate(
            model,
            basis,
            source_context,
            environment.target_context,
            delta_scale,
            args,
            environment.global_operator_signature,
        ),
    }
    print(initial, flush=True)
    model_pretraining = finite_horizon_cognitive_pretrain(
        model,
        replay_states,
        basis,
        source_context,
        environment.target_context,
        delta_scale,
        args,
    )
    after_model_pretraining = None
    if args.model_pretrain_steps > 0:
        after_model_pretraining = {
            "target_transitions": args.cognition_warmup,
            "stage": "finite_horizon_cognitive_pretraining",
            **evaluate(
                model,
                basis,
                source_context,
                environment.target_context,
                delta_scale,
                args,
                environment.global_operator_signature,
            ),
        }
        print(after_model_pretraining, flush=True)
    callback = RecoveryCallback(
        environment,
        (basis, source_context, delta_scale),
        args,
    )
    model.learn(
        total_timesteps=args.decision_transitions,
        callback=callback,
    )
    final = {
        "target_transitions": (
            args.cognition_warmup + args.decision_transitions
        ),
        "stage": "final_after_update",
        **evaluate(
            model,
            basis,
            source_context,
            environment.target_context,
            delta_scale,
            args,
            environment.global_operator_signature,
        ),
    }
    history = [
        initial,
        *(
            [after_model_pretraining]
            if after_model_pretraining is not None
            else []
        ),
        *callback.history,
    ]
    history.append(final)
    output = {
        "experiment": "HopperCognitiveFiLMAdaptation",
        "seed": args.seed,
        "device": str(device),
        "target": args.target,
        "decision_objective": args.decision_objective,
        "decision_adapter": args.decision_adapter,
        "physical_parameters_visible_to_learner": False,
        "cognition_reward_free": True,
        "actor_update_must_pass_through_cognition": True,
        "policy_mechanism_initialization": (
            mechanism_policy_shifts is not None
        ),
        "cognition_update_mode": args.cognition_update_mode,
        "cognition_actor_context_tau": (
            args.cognition_actor_context_tau
        ),
        "cognition_transfer_mode": args.cognition_transfer_mode,
        "cognition_posterior_transport": (
            args.cognition_posterior_transport
        ),
        "cognition_transfer_rate_summary": (
            {
                "mean": float(
                    np.mean(environment.cognition_transfer_rates)
                ),
                "minimum": float(
                    np.min(environment.cognition_transfer_rates)
                ),
                "maximum": float(
                    np.max(environment.cognition_transfer_rates)
                ),
            }
            if environment.cognition_transfer_rates
            else None
        ),
        "actor_cognition_context_gap": float(
            (
                environment.cognitive_context.coefficients
                - environment.target_context.coefficients
            ).norm()
        ),
        "cognition_refresh_statistics": (
            {
                "attempts": environment.estimator.refresh_attempts,
                "accepted": environment.estimator.refresh_count,
                "last_novelty": (
                    environment.estimator.last_novelty
                ),
                "last_confidence": (
                    environment.estimator.last_confidence
                ),
                "last_accepted_step": (
                    environment.estimator.last_accepted_step
                ),
                "last_validation_improvement": (
                    environment.estimator
                    .last_validation_improvement
                ),
                "evidence_batches": len(
                    environment.estimator.evidence_batches
                ),
                "sufficient_transition_count": (
                    environment.estimator
                    .sufficient_transition_count
                ),
            }
            if isinstance(
                environment.estimator,
                ValidatedResidualModeEstimator,
            )
            else None
        ),
        "config": vars(args),
        "model_pretraining": model_pretraining,
        "history": history,
    }
    Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json_out).write_text(
        json.dumps(output, indent=2) + "\n",
        encoding="utf-8",
    )
    model.save(args.model_out)
    normalized.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1811)
    parser.add_argument(
        "--target",
        choices=tuple(name for name in SHIFTS if name != "source"),
        default="payload_125",
    )
    parser.add_argument(
        "--decision-objective",
        choices=("tracking", "task", "hybrid"),
        default="tracking",
    )
    parser.add_argument("--tracking-weight", type=float, default=0.1)
    parser.add_argument("--cognition-warmup", type=int, default=1024)
    parser.add_argument("--cognition-batch", type=int, default=512)
    parser.add_argument(
        "--cognition-update-mode",
        choices=(
            "frozen",
            "full",
            "fixed_low_rank",
            "adaptive_low_rank",
        ),
        default="full",
    )
    parser.add_argument("--cognition-rank", type=int, default=4)
    parser.add_argument(
        "--cognition-coordinate-ridge", type=float, default=0.1,
    )
    parser.add_argument(
        "--cognition-forgetting-factor", type=float, default=0.9999,
    )
    parser.add_argument(
        "--cognition-refresh-threshold", type=float, default=0.5,
    )
    parser.add_argument(
        "--cognition-residual-ridge", type=float, default=1.0,
    )
    parser.add_argument(
        "--cognition-mode-step", type=float, default=1.0,
    )
    parser.add_argument(
        "--cognition-validation-fraction", type=float, default=0.25,
    )
    parser.add_argument(
        "--cognition-functional-proximal-weight",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--cognition-posterior-transport",
        choices=("reset", "replay", "sufficient"),
        default="reset",
    )
    parser.add_argument(
        "--cognition-actor-context-tau", type=float, default=1.0,
    )
    parser.add_argument(
        "--cognition-transfer-mode",
        choices=("fixed", "confidence"),
        default="fixed",
    )
    parser.add_argument("--warmup-noise", type=float, default=0.2)
    parser.add_argument("--decision-transitions", type=int, default=16384)
    parser.add_argument("--rollout-steps", type=int, default=1024)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--update-epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--entropy-coefficient", type=float, default=0.0)
    parser.add_argument("--source-kl-coefficient", type=float, default=0.0)
    parser.add_argument("--update-target-kl", type=float, default=None)
    parser.add_argument("--modulation-scale", type=float, default=0.25)
    parser.add_argument(
        "--decision-adapter",
        choices=("film", "lora", "mechanism_affine"),
        default="film",
    )
    parser.add_argument("--cognitive-lora-rank", type=int, default=4)
    parser.add_argument(
        "--operator-representation",
        choices=(
            "raw",
            "canonical",
            "trajectory_canonical",
            "mechanism",
        ),
        default="raw",
    )
    parser.add_argument(
        "--canonical-operator-damping", type=float, default=1e-3,
    )
    parser.add_argument(
        "--trajectory-signature-window", type=int, default=512,
    )
    parser.add_argument(
        "--mechanism-checkpoint",
        default="results/hopper_global_mechanism_latent_seed1811.pt",
    )
    parser.add_argument(
        "--mechanism-latent-ridge", type=float, default=1e-2,
    )
    parser.add_argument(
        "--mechanism-forgetting-factor", type=float, default=1.0,
    )
    parser.add_argument(
        "--mechanism-actor-models",
        nargs="+",
        default=(
            "results/hopper_payload125_mechanism_actor_frozennorm_80k_seed1811.zip",
            "results/hopper_friction070_mechanism_actor_frozennorm_80k_seed1811.zip",
            "results/hopper_actuator080_mechanism_actor_frozennorm_80k_seed1811.zip",
        ),
    )
    parser.add_argument(
        "--initialize-policy-mechanisms",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--model-pretrain-steps", type=int, default=0)
    parser.add_argument("--model-horizon", type=int, default=3)
    parser.add_argument("--model-batch-size", type=int, default=256)
    parser.add_argument("--model-learning-rate", type=float, default=1e-3)
    parser.add_argument("--model-anchor-weight", type=float, default=0.01)
    parser.add_argument("--model-gradient-clip", type=float, default=5.0)
    parser.add_argument("--model-report-every", type=int, default=25)
    parser.add_argument("--evaluate-every", type=int, default=4096)
    parser.add_argument("--evaluation-episodes", type=int, default=5)
    parser.add_argument(
        "--source-model",
        default="results/hopper_source_sb3_ppo_continued_seed1811.zip",
    )
    parser.add_argument(
        "--source-norm",
        default="results/hopper_source_sb3_vecnorm_continued_seed1811.pkl",
    )
    parser.add_argument(
        "--cognition-checkpoint",
        default="results/hopper_source_centered_protokan_seed1811.pt",
    )
    parser.add_argument(
        "--model-out",
        default="results/hopper_cognitive_film_payload125_seed1811",
    )
    parser.add_argument(
        "--json-out",
        default="results/hopper_cognitive_film_payload125_seed1811.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
