"""Online residual control on a distilled cognition-gated Hopper policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO, TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from torch.nn import functional as F

from cpbn.global_mechanism_kan import (
    GlobalMechanismKANDynamics,
    RecursiveGlobalMechanismEstimator,
)
from cpbn.centered_advantage_critic import CenteredAdvantageTD3Policy
from cpbn.conservative_policy_selection import (
    paired_return_lower_bound,
)
from cpbn.policy_mechanism_decoder import PolicyMechanismDecoder
from cpbn.trust_region_td3 import TrustRegionTD3
from scripts.diagnose_hopper_global_physics_context import collect_transitions
from scripts.prescreen_hopper_physics_shifts import SHIFTS, make_shifted_env
from scripts.validate_hopper_joint_online_adaptation import (
    FrozenSourcePolicy,
    load_cognition,
)


class DistilledMechanismResidualHopper(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        source,
        decoder,
        estimator,
        training_coordinates,
        args,
        *,
        seed_offset,
        update_cognition,
    ):
        super().__init__()
        self.environment = make_shifted_env(
            SHIFTS[args.target], args.seed + seed_offset,
        )()
        self.source = source
        self.decoder = decoder
        self.estimator = estimator
        self.training_coordinates = training_coordinates
        self.args = args
        self.update_cognition = update_cognition
        self.action_space = gym.spaces.Box(
            -1.0, 1.0, shape=(3,), dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, shape=(14,), dtype=np.float32,
        )
        self.raw_observation = None
        self.buffer = []
        self._coefficients = None
        self.refresh_coefficients()

    @torch.no_grad()
    def refresh_coefficients(self):
        coordinate = (
            self.estimator.latent() / self.args.mechanism_latent_scale
        )
        self._coefficients = torch.linalg.lstsq(
            self.training_coordinates.T,
            coordinate,
        ).solution

    @torch.no_grad()
    def coefficients(self):
        return self._coefficients

    @torch.no_grad()
    def base_action(self, observation):
        state = torch.as_tensor(
            observation,
            dtype=torch.float32,
            device=self.args.device,
        ).unsqueeze(0)
        correction = self.decoder(
            state, self.coefficients().unsqueeze(0),
        )[0]
        return (
            self.source.action(observation) + correction
        ).clamp(-1.0, 1.0)

    @torch.no_grad()
    def physical_residual(self, observation, mechanism_action):
        mechanism_action = torch.as_tensor(
            mechanism_action,
            dtype=torch.float32,
            device=self.args.device,
        )
        if self.args.residual_parameterization == "direct":
            return mechanism_action
        state = torch.as_tensor(
            observation,
            dtype=torch.float32,
            device=self.args.device,
        ).unsqueeze(0)
        effects = self.decoder.mechanism_effects(state)[0]
        tangent = effects.T * self.coefficients().unsqueeze(0)
        gram = tangent.T @ tangent
        eigenvalues, eigenvectors = torch.linalg.eigh(
            gram
            + self.args.tangent_damping
            * torch.eye(
                gram.shape[0],
                dtype=gram.dtype,
                device=gram.device,
            )
        )
        inverse_root = (
            eigenvectors
            @ torch.diag(eigenvalues.clamp_min(1e-8).rsqrt())
            @ eigenvectors.T
        )
        return (
            tangent @ inverse_root @ mechanism_action
        ).clamp(-1.0, 1.0)

    @torch.no_grad()
    def transformed(self, observation):
        state = torch.as_tensor(
            observation,
            dtype=torch.float32,
            device=self.args.device,
        )
        normalized = (
            (state - self.source.mean)
            / (self.source.variance + 1e-8).sqrt()
        ).clamp(-10.0, 10.0)
        return torch.cat(
            (normalized, self.coefficients().clamp(-10.0, 10.0)),
        ).cpu().numpy().astype(np.float32)

    def reset(self, *, seed=None, options=None):
        observation, info = self.environment.reset(
            seed=seed, options=options,
        )
        self.raw_observation = observation
        return self.transformed(observation), info

    def step(self, residual_action):
        base = self.base_action(self.raw_observation)
        residual = self.physical_residual(
            self.raw_observation, residual_action,
        )
        action = (
            base + self.args.residual_scale * residual
        ).clamp(-1.0, 1.0)
        following, reward, terminated, truncated, info = (
            self.environment.step(action.cpu().numpy())
        )
        if self.update_cognition:
            nominal = self.source.action(self.raw_observation)
            self.buffer.append(
                (
                    self.raw_observation.copy(),
                    (action - nominal).cpu().numpy(),
                    following - self.raw_observation,
                )
            )
            if len(self.buffer) >= self.args.cognition_batch:
                state, innovation, delta = zip(*self.buffer)
                self.estimator.update(
                    torch.as_tensor(
                        np.asarray(state),
                        dtype=torch.float32,
                        device=self.args.device,
                    ),
                    torch.as_tensor(
                        np.asarray(innovation),
                        dtype=torch.float32,
                        device=self.args.device,
                    ),
                    torch.as_tensor(
                        np.asarray(delta),
                        dtype=torch.float32,
                        device=self.args.device,
                    ),
                    evidence_weight=self.args.online_cognition_weight,
                )
                self.refresh_coefficients()
                self.buffer.clear()
        self.raw_observation = following
        info["base_action"] = base.cpu().numpy()
        return (
            self.transformed(following),
            reward,
            terminated,
            truncated,
            info,
        )

    def close(self):
        self.environment.close()


@torch.no_grad()
def collect_historical_critic_dataset(source, decoder, args, device):
    """Collect reward-labeled residual rollouts in known mechanism worlds."""
    observations = []
    actions = []
    returns = []
    rng = np.random.default_rng(args.seed + 23001)
    mechanism_count = decoder.mechanisms.__len__()
    for environment_index, environment_name in enumerate(
        args.historical_mechanism_environments
    ):
        if environment_index >= mechanism_count:
            raise ValueError(
                "More historical environments than decoder mechanisms.",
            )
        coordinate = torch.zeros(mechanism_count, device=device)
        coordinate[environment_index] = 1.0
        environment = make_shifted_env(
            SHIFTS[environment_name],
            args.seed + 23000 + environment_index,
        )()
        state, _ = environment.reset(
            seed=args.seed + 23000 + environment_index,
        )
        episode_observations = []
        episode_actions = []
        episode_rewards = []
        collected = 0
        while collected < args.historical_critic_transitions:
            tensor_state = torch.as_tensor(
                state, dtype=torch.float32, device=device,
            ).unsqueeze(0)
            correction = decoder(
                tensor_state, coordinate.unsqueeze(0),
            )[0]
            base = (
                source.action(state) + correction
            ).clamp(-1.0, 1.0)
            residual = np.clip(
                rng.normal(
                    0.0,
                    args.historical_critic_noise,
                    size=3,
                ),
                -1.0,
                1.0,
            ).astype(np.float32)
            physical_action = (
                base
                + args.residual_scale
                * torch.as_tensor(
                    residual,
                    dtype=torch.float32,
                    device=device,
                )
            ).clamp(-1.0, 1.0)
            normalized_state = (
                (
                    tensor_state[0] - source.mean
                )
                / (source.variance + 1e-8).sqrt()
            ).clamp(-10.0, 10.0)
            transformed = torch.cat(
                (normalized_state, coordinate),
            ).cpu().numpy().astype(np.float32)
            following, reward, terminated, truncated, _ = (
                environment.step(physical_action.cpu().numpy())
            )
            episode_observations.append(transformed)
            episode_actions.append(residual)
            episode_rewards.append(float(reward))
            collected += 1
            state = following
            if (
                terminated
                or truncated
                or collected == args.historical_critic_transitions
            ):
                discounted = 0.0
                episode_returns = []
                for value in reversed(episode_rewards):
                    discounted = value + args.gamma * discounted
                    episode_returns.append(discounted)
                observations.extend(episode_observations)
                actions.extend(episode_actions)
                returns.extend(reversed(episode_returns))
                episode_observations.clear()
                episode_actions.clear()
                episode_rewards.clear()
                if collected < args.historical_critic_transitions:
                    state, _ = environment.reset()
        environment.close()
    return {
        "observations": torch.as_tensor(
            np.asarray(observations),
            dtype=torch.float32,
            device=device,
        ),
        "actions": torch.as_tensor(
            np.asarray(actions),
            dtype=torch.float32,
            device=device,
        ),
        "returns": torch.as_tensor(
            np.asarray(returns),
            dtype=torch.float32,
            device=device,
        ).unsqueeze(1),
    }


def pretrain_historical_critic(model, dataset, args, device):
    """Supervise both TD3 critics with historical Monte-Carlo returns."""
    generator = torch.Generator(device=device).manual_seed(
        args.seed + 23011,
    )
    sample_count = dataset["observations"].shape[0]
    history = []
    for step in range(1, args.historical_critic_gradient_steps + 1):
        indices = torch.randint(
            sample_count,
            (
                min(
                    args.minibatch_size,
                    sample_count,
                ),
            ),
            generator=generator,
            device=device,
        )
        predicted = model.critic(
            dataset["observations"][indices],
            dataset["actions"][indices],
        )
        target = dataset["returns"][indices]
        loss = sum(F.mse_loss(q_value, target) for q_value in predicted)
        model.critic.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        model.critic.optimizer.step()
        if (
            step == 1
            or step == args.historical_critic_gradient_steps
            or step % args.historical_critic_report_every == 0
        ):
            record = {
                "step": step,
                "loss": float(loss.detach()),
            }
            history.append(record)
            print(
                {"stage": "historical_critic", **record},
                flush=True,
            )
    polyak_update(
        model.critic.parameters(),
        model.critic_target.parameters(),
        1.0,
    )
    return history


@torch.no_grad()
def evaluate(
    model,
    environment,
    args,
    *,
    episodes=None,
    seed_offset=10000,
):
    target = make_shifted_env(
        SHIFTS[args.target], args.seed + seed_offset,
    )()
    returns, lengths = [], []
    healthy = 0
    residuals = []
    physical_residuals = []
    episode_count = episodes or args.evaluation_episodes
    for episode in range(episode_count):
        observation, _ = target.reset(
            seed=args.seed + seed_offset + episode,
        )
        total = 0.0
        length = 0
        while True:
            transformed = environment.transformed(observation)
            if model is None:
                residual = np.zeros(3, dtype=np.float32)
            else:
                residual, _ = model.predict(
                    transformed, deterministic=True,
                )
            base = environment.base_action(observation)
            physical_residual = environment.physical_residual(
                observation, residual,
            )
            action = (
                base
                + args.residual_scale
                * physical_residual
            ).clamp(-1.0, 1.0)
            residuals.extend(np.abs(residual))
            physical_residuals.extend(
                torch.abs(physical_residual).cpu().tolist(),
            )
            observation, reward, terminated, truncated, _ = (
                target.step(action.cpu().numpy())
            )
            total += float(reward)
            length += 1
            if terminated or truncated:
                healthy += int(truncated and not terminated)
                break
        returns.append(total)
        lengths.append(length)
    target.close()
    residual_array = np.asarray(residuals, dtype=np.float32).reshape(-1, 3)
    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_episode_length": float(np.mean(lengths)),
        "healthy_completion_rate": healthy / episode_count,
        "episode_returns": returns,
        "episode_lengths": lengths,
        "residual_abs_mean": float(np.mean(residuals)),
        "residual_mean_per_dimension": (
            residual_array.mean(axis=0).tolist()
        ),
        "residual_std_per_dimension": (
            residual_array.std(axis=0).tolist()
        ),
        "residual_saturation_rate": float(
            (np.abs(residual_array) > 0.95).mean(),
        ),
        "physical_residual_abs_mean": float(
            np.mean(physical_residuals),
        ),
    }


def main(args):
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    print({"stage": "setup", "device": str(device)}, flush=True)
    source = FrozenSourcePolicy(
        args.source_model, args.source_norm, device, args.seed,
    )
    basis, source_context, _, delta_scale = load_cognition(args, device)
    mechanism_payload = torch.load(
        args.mechanism_checkpoint,
        map_location=device,
        weights_only=True,
    )
    mechanism_model = GlobalMechanismKANDynamics(
        source_context,
        mechanism_payload["mechanisms"].to(device),
    )
    estimator = RecursiveGlobalMechanismEstimator(
        mechanism_model,
        basis,
        delta_scale,
        ridge=args.mechanism_latent_ridge,
    )
    warmup = collect_transitions(
        source,
        args.target,
        args.cognition_warmup,
        args,
        device,
        13000,
    )
    estimator.update(
        warmup["state"], warmup["innovation"], warmup["delta"],
    )
    args.mechanism_latent_scale = mechanism_payload[
        "latent_scale"
    ].to(device)
    decoder_payload = torch.load(
        args.decoder_checkpoint,
        map_location=device,
        weights_only=True,
    )
    decoder = PolicyMechanismDecoder(
        source.mean, source.variance,
        mechanism_dim=mechanism_model.mechanisms.shape[0],
    ).to(device)
    decoder.load_state_dict(decoder_payload["decoder"])
    decoder.eval()
    environment = DistilledMechanismResidualHopper(
        source,
        decoder,
        estimator,
        decoder_payload["training_coordinates"].to(device),
        args,
        seed_offset=500,
        update_cognition=True,
    )
    initial = {
        "target_transitions": args.cognition_warmup,
        "mechanism_coefficients": (
            environment.coefficients().cpu().tolist()
        ),
        **evaluate(None, environment, args),
    }
    print({"stage": "initial", **initial}, flush=True)
    vector = DummyVecEnv([lambda: environment])
    normalized = VecNormalize(
        vector,
        training=True,
        norm_obs=False,
        norm_reward=args.reward_normalization == "running",
        gamma=args.gamma,
    )
    historical_critic = []
    if args.decision_algorithm == "ppo":
        model = PPO(
            "MlpPolicy",
            normalized,
            learning_rate=args.learning_rate,
            n_steps=args.rollout_steps,
            batch_size=args.minibatch_size,
            n_epochs=args.update_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            seed=args.seed,
            device=device,
            verbose=0,
        )
        torch.nn.init.zeros_(model.policy.action_net.weight)
        torch.nn.init.zeros_(model.policy.action_net.bias)
        model.policy.log_std.data.fill_(args.initial_log_std)
    else:
        noise = NormalActionNoise(
            mean=np.zeros(3, dtype=np.float32),
            sigma=args.td3_exploration_noise
            * np.ones(3, dtype=np.float32),
        )
        algorithm_class = (
            TrustRegionTD3
            if args.decision_algorithm == "td3_trust"
            else TD3
        )
        model = algorithm_class(
            (
                CenteredAdvantageTD3Policy
                if args.centered_advantage_critic
                else "MlpPolicy"
            ),
            normalized,
            learning_rate=args.learning_rate,
            buffer_size=args.replay_buffer_size,
            learning_starts=args.learning_starts,
            batch_size=args.minibatch_size,
            tau=args.target_tau,
            gamma=args.gamma,
            train_freq=(args.train_frequency, "step"),
            gradient_steps=args.gradient_steps,
            action_noise=noise,
            policy_delay=args.policy_delay,
            target_policy_noise=args.target_policy_noise,
            target_noise_clip=args.target_noise_clip,
            seed=args.seed,
            device=device,
            verbose=0,
            **(
                {
                    "source_trust_coefficient": (
                        args.source_trust_coefficient
                    ),
                    "uncertainty_coefficient": (
                        args.uncertainty_coefficient
                    ),
                    "behavior_coefficient": (
                        args.behavior_coefficient
                    ),
                    "adaptive_q_coefficient": (
                        args.adaptive_q_coefficient
                    ),
                    "baseline_action_probability": (
                        args.baseline_action_probability
                    ),
                }
                if args.decision_algorithm == "td3_trust"
                else {}
            ),
        )
        actor_last = next(
            layer
            for layer in reversed(model.actor.mu)
            if isinstance(layer, torch.nn.Linear)
        )
        torch.nn.init.zeros_(actor_last.weight)
        torch.nn.init.zeros_(actor_last.bias)
        model.actor_target.load_state_dict(model.actor.state_dict())
        if args.historical_critic_transitions > 0:
            if args.reward_normalization != "none":
                raise ValueError(
                    "Historical Monte-Carlo critic targets require "
                    "--reward-normalization none.",
                )
            historical_dataset = collect_historical_critic_dataset(
                source, decoder, args, device,
            )
            print(
                {
                    "stage": "historical_dataset",
                    "transitions": int(
                        historical_dataset["observations"].shape[0],
                    ),
                    "return_mean": float(
                        historical_dataset["returns"].mean(),
                    ),
                    "return_std": float(
                        historical_dataset["returns"].std(),
                    ),
                },
                flush=True,
            )
            historical_critic = pretrain_historical_critic(
                model, historical_dataset, args, device,
            )
    model.learn(total_timesteps=args.decision_transitions)
    selection = {
        "episodes_per_policy": 0,
        "candidate_selected": True,
        "paired_improvement_mean": None,
        "paired_improvement_lcb": None,
        "transitions": 0,
    }
    if args.policy_selection_episodes > 0:
        selection_base = evaluate(
            None,
            environment,
            args,
            episodes=args.policy_selection_episodes,
            seed_offset=20000,
        )
        selection_candidate = evaluate(
            model,
            environment,
            args,
            episodes=args.policy_selection_episodes,
            seed_offset=20000,
        )
        comparison = paired_return_lower_bound(
            selection_base["episode_returns"],
            selection_candidate["episode_returns"],
            confidence_multiplier=(
                args.policy_selection_confidence
            ),
        )
        selection = {
            "episodes_per_policy": args.policy_selection_episodes,
            "candidate_selected": comparison["accepted"],
            "paired_improvement_mean": comparison["mean"],
            "paired_improvement_lcb": comparison["lower_bound"],
            "paired_improvement_standard_error": (
                comparison["standard_error"]
            ),
            "base_returns": selection_base["episode_returns"],
            "candidate_returns": (
                selection_candidate["episode_returns"]
            ),
            "transitions": int(
                sum(selection_base["episode_lengths"])
                + sum(selection_candidate["episode_lengths"])
            ),
        }
    total_target_transitions = (
        args.cognition_warmup
        + args.decision_transitions
        + selection["transitions"]
    )
    final_base = {
        "target_transitions": total_target_transitions,
        "mechanism_coefficients": (
            environment.coefficients().cpu().tolist()
        ),
        **evaluate(None, environment, args),
    }
    final_candidate = {
        "target_transitions": total_target_transitions,
        "mechanism_coefficients": (
            environment.coefficients().cpu().tolist()
        ),
        **evaluate(model, environment, args),
    }
    final = (
        final_candidate
        if selection["candidate_selected"]
        else final_base
    )
    print({"stage": "final", **final}, flush=True)
    output = {
        "experiment": "HopperDistilledMechanismOnlineResidual",
        "seed": args.seed,
        "device": str(device),
        "target": args.target,
        "physical_parameters_visible_to_learner": False,
        "cognition_reward_free": True,
        "decision_uses_real_reward": True,
        "decision_algorithm": args.decision_algorithm,
        "historical_critic_pretraining": historical_critic,
        "policy_selection": selection,
        "initial": initial,
        "final_cognition_base_only": final_base,
        "final_candidate": final_candidate,
        "final": final,
        "config": {
            key: (
                str(value)
                if isinstance(value, torch.device)
                else value.detach().cpu().tolist()
                if isinstance(value, torch.Tensor)
                else value
            )
            for key, value in vars(args).items()
        },
    }
    Path(args.json_out).write_text(
        json.dumps(output, indent=2) + "\n",
        encoding="utf-8",
    )
    model.save(args.model_out)
    normalized.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1811)
    parser.add_argument("--target", default="combo_medium")
    parser.add_argument("--cognition-warmup", type=int, default=512)
    parser.add_argument("--cognition-batch", type=int, default=512)
    parser.add_argument(
        "--online-cognition-weight", type=float, default=0.1,
    )
    parser.add_argument("--exploration-noise", type=float, default=0.2)
    parser.add_argument("--mechanism-latent-ridge", type=float, default=1e-2)
    parser.add_argument("--decision-transitions", type=int, default=2048)
    parser.add_argument(
        "--decision-algorithm",
        choices=("ppo", "td3", "td3_trust"),
        default="ppo",
    )
    parser.add_argument("--rollout-steps", type=int, default=512)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--update-epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--initial-log-std", type=float, default=-1.5)
    parser.add_argument("--replay-buffer-size", type=int, default=100000)
    parser.add_argument("--learning-starts", type=int, default=256)
    parser.add_argument("--train-frequency", type=int, default=1)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument(
        "--reward-normalization",
        choices=("running", "none"),
        default="running",
    )
    parser.add_argument("--target-tau", type=float, default=0.005)
    parser.add_argument("--policy-delay", type=int, default=2)
    parser.add_argument("--target-policy-noise", type=float, default=0.1)
    parser.add_argument("--target-noise-clip", type=float, default=0.2)
    parser.add_argument(
        "--td3-exploration-noise", type=float, default=0.1,
    )
    parser.add_argument(
        "--source-trust-coefficient", type=float, default=0.1,
    )
    parser.add_argument(
        "--uncertainty-coefficient", type=float, default=0.1,
    )
    parser.add_argument(
        "--behavior-coefficient", type=float, default=0.0,
    )
    parser.add_argument(
        "--adaptive-q-coefficient", type=float, default=2.5,
    )
    parser.add_argument(
        "--centered-advantage-critic",
        action="store_true",
    )
    parser.add_argument(
        "--baseline-action-probability", type=float, default=0.0,
    )
    parser.add_argument(
        "--policy-selection-episodes", type=int, default=0,
    )
    parser.add_argument(
        "--policy-selection-confidence", type=float, default=1.0,
        help="One-sided standard-error multiplier.",
    )
    parser.add_argument(
        "--historical-mechanism-environments",
        nargs="+",
        default=("payload_125", "friction_070", "actuator_080"),
    )
    parser.add_argument(
        "--historical-critic-transitions", type=int, default=0,
        help="Transitions per known mechanism environment.",
    )
    parser.add_argument(
        "--historical-critic-noise", type=float, default=0.2,
    )
    parser.add_argument(
        "--historical-critic-gradient-steps", type=int, default=1000,
    )
    parser.add_argument(
        "--historical-critic-report-every", type=int, default=250,
    )
    parser.add_argument("--residual-scale", type=float, default=0.25)
    parser.add_argument(
        "--residual-parameterization",
        choices=("direct", "mechanism_tangent"),
        default="direct",
    )
    parser.add_argument("--tangent-damping", type=float, default=1e-3)
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
        "--mechanism-checkpoint",
        default="results/hopper_global_mechanism_latent_seed1811.pt",
    )
    parser.add_argument(
        "--decoder-checkpoint",
        default="results/hopper_policy_mechanism_decoder_seed1811.pt",
    )
    parser.add_argument(
        "--model-out",
        default="results/hopper_distilled_mechanism_online_seed1811",
    )
    parser.add_argument(
        "--json-out",
        default="results/hopper_distilled_mechanism_online_seed1811.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
