"""Cognition-conditioned hidden modulation of a frozen source Actor."""

from __future__ import annotations

import torch
from torch import nn


class HopperCognitiveFiLMExtractor(nn.Module):
    """Preserve a source policy and modulate both hidden layers via cognition.

    The source actor weights remain frozen.  The only trainable path into its
    action is a bias-free FiLM adapter driven by the evaluated dynamics
    operator.  Zero adapter weights exactly reproduce the source actor.
    """

    def __init__(
        self,
        source_state,
        observation_mean,
        observation_variance,
        *,
        state_dim: int = 11,
        operator_dim: int = 44,
        hidden_dim: int = 64,
        modulation_scale: float = 0.25,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.operator_dim = operator_dim
        self.latent_dim_pi = hidden_dim
        self.latent_dim_vf = hidden_dim
        self.modulation_scale = float(modulation_scale)
        self.register_buffer(
            "observation_mean",
            torch.as_tensor(observation_mean, dtype=torch.float32),
        )
        self.register_buffer(
            "observation_variance",
            torch.as_tensor(observation_variance, dtype=torch.float32),
        )
        self.source_first = nn.Linear(state_dim, hidden_dim)
        self.source_second = nn.Linear(hidden_dim, hidden_dim)
        self.value_first = nn.Linear(
            state_dim + operator_dim, hidden_dim,
        )
        self.value_second = nn.Linear(hidden_dim, hidden_dim)
        self.load_source_weights(source_state)
        self.source_first.requires_grad_(False)
        self.source_second.requires_grad_(False)
        self.adapter = nn.Sequential(
            nn.Linear(operator_dim, hidden_dim, bias=False),
            nn.Tanh(),
            nn.Linear(hidden_dim, 4 * hidden_dim, bias=False),
        )
        self.reset_adapter()

    @torch.no_grad()
    def load_source_weights(self, source_state):
        self.source_first.load_state_dict({
            "weight": source_state[
                "mlp_extractor.policy_net.0.weight"
            ],
            "bias": source_state[
                "mlp_extractor.policy_net.0.bias"
            ],
        })
        self.source_second.load_state_dict({
            "weight": source_state[
                "mlp_extractor.policy_net.2.weight"
            ],
            "bias": source_state[
                "mlp_extractor.policy_net.2.bias"
            ],
        })
        self.value_first.weight.zero_()
        self.value_first.weight[:, :self.state_dim].copy_(
            source_state["mlp_extractor.value_net.0.weight"],
        )
        self.value_first.bias.copy_(
            source_state["mlp_extractor.value_net.0.bias"],
        )
        self.value_second.weight.copy_(
            source_state["mlp_extractor.value_net.2.weight"],
        )
        self.value_second.bias.copy_(
            source_state["mlp_extractor.value_net.2.bias"],
        )

    @torch.no_grad()
    def reset_adapter(self):
        """Restore exact source-policy behavior before target adaptation."""
        nn.init.orthogonal_(self.adapter[0].weight)
        nn.init.zeros_(self.adapter[2].weight)

    def normalized_state(self, observation):
        state = observation[..., :self.state_dim]
        return (
            (state - self.observation_mean)
            / (self.observation_variance + 1e-8).sqrt()
        ).clamp(-10.0, 10.0)

    def normalized_operator(self, observation):
        return observation[
            ..., self.state_dim:self.state_dim + self.operator_dim
        ]

    def forward_actor(self, observation):
        state = self.normalized_state(observation)
        operator = self.normalized_operator(observation)
        modulation = self.adapter(operator)
        gamma_1, beta_1, gamma_2, beta_2 = modulation.chunk(4, dim=-1)
        hidden = torch.tanh(self.source_first(state))
        hidden = (
            hidden
            * (1.0 + self.modulation_scale * torch.tanh(gamma_1))
            + self.modulation_scale * beta_1
        )
        hidden = torch.tanh(self.source_second(hidden))
        return (
            hidden
            * (1.0 + self.modulation_scale * torch.tanh(gamma_2))
            + self.modulation_scale * beta_2
        )

    def forward_critic(self, observation):
        critic_input = torch.cat(
            (
                self.normalized_state(observation),
                self.normalized_operator(observation),
            ),
            dim=-1,
        )
        return torch.tanh(
            self.value_second(
                torch.tanh(self.value_first(critic_input)),
            ),
        )

    def forward(self, observation):
        return (
            self.forward_actor(observation),
            self.forward_critic(observation),
        )


class HopperCognitiveLoRAExtractor(HopperCognitiveFiLMExtractor):
    """Use one cognition hypernetwork to modulate all Actor layers."""

    def __init__(
        self,
        source_state,
        observation_mean,
        observation_variance,
        *,
        state_dim: int = 11,
        operator_dim: int = 44,
        hidden_dim: int = 64,
        action_dim: int = 3,
        rank: int = 4,
        modulation_scale: float = 0.25,
    ):
        super().__init__(
            source_state,
            observation_mean,
            observation_variance,
            state_dim=state_dim,
            operator_dim=operator_dim,
            hidden_dim=hidden_dim,
            modulation_scale=modulation_scale,
        )
        self.action_dim = action_dim
        self.rank = rank
        self.latent_dim_pi = hidden_dim + action_dim
        del self.adapter
        self.operator_to_coefficients = nn.Sequential(
            nn.Linear(operator_dim, hidden_dim, bias=False),
            nn.Tanh(),
            nn.Linear(hidden_dim, 3 * rank, bias=False),
        )
        self.first_left = nn.Parameter(
            torch.empty(hidden_dim, rank)
        )
        self.first_right = nn.Parameter(
            torch.empty(state_dim, rank)
        )
        self.second_left = nn.Parameter(
            torch.empty(hidden_dim, rank)
        )
        self.second_right = nn.Parameter(
            torch.empty(hidden_dim, rank)
        )
        self.action_left = nn.Parameter(
            torch.empty(action_dim, rank)
        )
        self.action_right = nn.Parameter(
            torch.empty(hidden_dim, rank)
        )
        self.reset_adapter()

    @torch.no_grad()
    def reset_adapter(self):
        if not hasattr(self, "operator_to_coefficients"):
            return
        for matrix in (
            self.first_left,
            self.first_right,
            self.second_left,
            self.second_right,
            self.action_left,
            self.action_right,
        ):
            nn.init.orthogonal_(matrix)
        nn.init.orthogonal_(
            self.operator_to_coefficients[0].weight
        )
        nn.init.zeros_(
            self.operator_to_coefficients[2].weight
        )

    def low_rank_effect(self, value, coefficient, left, right):
        projected = value @ right
        return (
            self.modulation_scale
            * ((projected * coefficient) @ left.T)
        )

    def forward_actor(self, observation):
        state = self.normalized_state(observation)
        operator = self.normalized_operator(observation)
        coefficients = self.operator_to_coefficients(operator)
        first, second, action = coefficients.chunk(3, dim=-1)
        first_pre = self.source_first(state)
        first_pre = first_pre + self.low_rank_effect(
            state,
            first,
            self.first_left,
            self.first_right,
        )
        hidden = torch.tanh(first_pre)
        second_pre = self.source_second(hidden)
        second_pre = second_pre + self.low_rank_effect(
            hidden,
            second,
            self.second_left,
            self.second_right,
        )
        hidden = torch.tanh(second_pre)
        action_correction = self.low_rank_effect(
            hidden,
            action,
            self.action_left,
            self.action_right,
        )
        return torch.cat((hidden, action_correction), dim=-1)


class HopperMechanismAffineExtractor(HopperCognitiveFiLMExtractor):
    """Directly transport every source Actor layer along mechanism axes."""

    def __init__(
        self,
        source_state,
        observation_mean,
        observation_variance,
        *,
        state_dim: int = 11,
        mechanism_dim: int = 3,
        hidden_dim: int = 64,
        action_dim: int = 3,
        modulation_scale: float = 0.25,
    ):
        super().__init__(
            source_state,
            observation_mean,
            observation_variance,
            state_dim=state_dim,
            operator_dim=mechanism_dim,
            hidden_dim=hidden_dim,
            modulation_scale=modulation_scale,
        )
        self.action_dim = action_dim
        self.latent_dim_pi = hidden_dim + action_dim
        del self.adapter
        self.first_weight_shift = nn.Parameter(
            torch.zeros(mechanism_dim, hidden_dim, state_dim),
        )
        self.first_bias_shift = nn.Parameter(
            torch.zeros(mechanism_dim, hidden_dim),
        )
        self.second_weight_shift = nn.Parameter(
            torch.zeros(mechanism_dim, hidden_dim, hidden_dim),
        )
        self.second_bias_shift = nn.Parameter(
            torch.zeros(mechanism_dim, hidden_dim),
        )
        self.action_weight_shift = nn.Parameter(
            torch.zeros(mechanism_dim, action_dim, hidden_dim),
        )
        self.action_bias_shift = nn.Parameter(
            torch.zeros(mechanism_dim, action_dim),
        )

    @torch.no_grad()
    def reset_adapter(self):
        if not hasattr(self, "first_weight_shift"):
            return
        for parameter in (
            self.first_weight_shift,
            self.first_bias_shift,
            self.second_weight_shift,
            self.second_bias_shift,
            self.action_weight_shift,
            self.action_bias_shift,
        ):
            parameter.zero_()

    @torch.no_grad()
    def load_policy_shifts(self, shifts):
        for name in (
            "first_weight_shift",
            "first_bias_shift",
            "second_weight_shift",
            "second_bias_shift",
            "action_weight_shift",
            "action_bias_shift",
        ):
            getattr(self, name).copy_(shifts[name])

    def affine_shift(self, latent, value, weight, bias):
        return self.modulation_scale * (
            torch.einsum("bk,koi,bi->bo", latent, weight, value)
            + latent @ bias
        )

    def forward_actor(self, observation):
        state = self.normalized_state(observation)
        mechanism = self.normalized_operator(observation)
        first_pre = self.source_first(state) + self.affine_shift(
            mechanism,
            state,
            self.first_weight_shift,
            self.first_bias_shift,
        )
        hidden = torch.tanh(first_pre)
        second_pre = self.source_second(hidden) + self.affine_shift(
            mechanism,
            hidden,
            self.second_weight_shift,
            self.second_bias_shift,
        )
        hidden = torch.tanh(second_pre)
        action_correction = self.affine_shift(
            mechanism,
            hidden,
            self.action_weight_shift,
            self.action_bias_shift,
        )
        return torch.cat((hidden, action_correction), dim=-1)
