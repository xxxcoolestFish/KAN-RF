"""Online residual control on a distilled cognition-gated Hopper policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from cpbn.global_mechanism_kan import (
    GlobalMechanismKANDynamics,
    RecursiveGlobalMechanismEstimator,
)
from cpbn.policy_mechanism_decoder import PolicyMechanismDecoder
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

    @torch.no_grad()
    def coefficients(self):
        coordinate = (
            self.estimator.latent() / self.args.mechanism_latent_scale
        )
        return torch.linalg.lstsq(
            self.training_coordinates.T,
            coordinate,
        ).solution

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
        residual = torch.as_tensor(
            residual_action,
            dtype=torch.float32,
            device=self.args.device,
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
def evaluate(model, environment, args):
    target = make_shifted_env(
        SHIFTS[args.target], args.seed + 10000,
    )()
    returns, lengths = [], []
    healthy = 0
    residuals = []
    for episode in range(args.evaluation_episodes):
        observation, _ = target.reset(
            seed=args.seed + 10000 + episode,
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
            action = (
                base
                + args.residual_scale
                * torch.as_tensor(
                    residual,
                    dtype=torch.float32,
                    device=args.device,
                )
            ).clamp(-1.0, 1.0)
            residuals.extend(np.abs(residual))
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
    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_episode_length": float(np.mean(lengths)),
        "healthy_completion_rate": healthy / args.evaluation_episodes,
        "residual_abs_mean": float(np.mean(residuals)),
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
        norm_reward=True,
        gamma=args.gamma,
    )
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
    model.learn(total_timesteps=args.decision_transitions)
    final_base = {
        "target_transitions": (
            args.cognition_warmup + args.decision_transitions
        ),
        "mechanism_coefficients": (
            environment.coefficients().cpu().tolist()
        ),
        **evaluate(None, environment, args),
    }
    final = {
        "target_transitions": (
            args.cognition_warmup + args.decision_transitions
        ),
        "mechanism_coefficients": (
            environment.coefficients().cpu().tolist()
        ),
        **evaluate(model, environment, args),
    }
    print({"stage": "final", **final}, flush=True)
    output = {
        "experiment": "HopperDistilledMechanismOnlineResidual",
        "seed": args.seed,
        "device": str(device),
        "target": args.target,
        "physical_parameters_visible_to_learner": False,
        "cognition_reward_free": True,
        "decision_uses_real_reward": True,
        "initial": initial,
        "final_cognition_base_only": final_base,
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
    parser.add_argument("--rollout-steps", type=int, default=512)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--update-epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--initial-log-std", type=float, default=-1.5)
    parser.add_argument("--residual-scale", type=float, default=0.25)
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
