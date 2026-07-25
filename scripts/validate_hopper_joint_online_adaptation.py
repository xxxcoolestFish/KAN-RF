"""Joint online ProtoKAN cognition and residual PPO on Hopper."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from cpbn.generic_affine_kan import (
    AffineKANContext,
    CompactInteractionKANDictionary,
    RecursiveAffineKANEstimator,
)
from scripts.prescreen_hopper_physics_shifts import (
    ENVS,
    SHIFTS,
    make_shifted_env,
)


class FrozenSourcePolicy:
    def __init__(self, model_path, norm_path, device, seed, env="hopper"):
        temporary = DummyVecEnv(
            [make_shifted_env(SHIFTS["source"], seed + 9000, env)],
        )
        norm = VecNormalize.load(norm_path, temporary)
        self.mean = torch.as_tensor(
            norm.obs_rms.mean, dtype=torch.float32, device=device,
        )
        self.variance = torch.as_tensor(
            norm.obs_rms.var, dtype=torch.float32, device=device,
        )
        temporary.close()
        self.model = PPO.load(model_path, device=device)
        self.device = device

    @torch.no_grad()
    def action(self, observation):
        observation = torch.as_tensor(
            observation, dtype=torch.float32, device=self.device,
        )
        normalized = (
            (observation - self.mean)
            / (self.variance + 1e-8).sqrt()
        ).clamp(-10.0, 10.0)
        action, _ = self.model.predict(
            normalized.cpu().numpy(),
            deterministic=True,
        )
        return torch.as_tensor(
            action, dtype=torch.float32, device=self.device,
        )

    def value_effect_metric(self, observation, isotropic_floor):
        """Automatic task relevance metric from the frozen source Critic."""
        with torch.enable_grad():
            raw = torch.as_tensor(
                observation,
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0).detach().requires_grad_(True)
            normalized = (
                (raw - self.mean)
                / (self.variance + 1e-8).sqrt()
            ).clamp(-10.0, 10.0)
            value = self.model.policy.predict_values(normalized)
            gradient = torch.autograd.grad(value.sum(), raw)[0].detach()
        direction = gradient / gradient.norm(
            dim=-1, keepdim=True,
        ).clamp_min(1e-6)
        identity = torch.eye(
            direction.shape[-1],
            dtype=direction.dtype,
            device=direction.device,
        ).unsqueeze(0)
        return (
            direction.unsqueeze(-1) @ direction.unsqueeze(-2)
            + isotropic_floor * identity
        )


def load_cognition(args, device):
    payload = torch.load(
        args.cognition_checkpoint,
        map_location=device,
        weights_only=True,
    )
    basis = CompactInteractionKANDictionary(
        payload["state_scale"],
        torch.ones(int(payload.get("action_dim", 3)), device=device),
        pair_modes=int(payload["pair_modes"]),
    ).to(device)
    source = AffineKANContext(
        payload["source_coefficients"].to(device),
    )
    source.source_controllable_bias = payload.get(
        "source_controllable_bias",
        torch.zeros(basis.action_dim),
    ).to(device)
    estimator = RecursiveAffineKANEstimator(
        basis,
        source,
        ridge=float(payload["cognition_ridge"]),
        forgetting_factor=float(payload["forgetting_factor"]),
    )
    estimator.precision.copy_(
        payload["estimator_precision"].to(device),
    )
    estimator.right.copy_(payload["estimator_right"].to(device))
    estimator.base_precision.copy_(
        payload["estimator_base_precision"].to(device),
    )
    estimator.base_right.copy_(
        payload["estimator_base_right"].to(device),
    )
    basis.policy_centered = bool(payload.get("policy_centered", False))
    return (
        basis,
        source,
        estimator,
        payload["delta_scale"].to(device),
    )


@torch.no_grad()
def cognitive_action_and_features(
    observation,
    source_policy,
    basis,
    source_context,
    target_context,
    delta_scale,
    damping,
    effect_metric_mode="identity",
    metric_isotropic_floor=0.05,
    return_details=False,
):
    state = torch.as_tensor(
        observation,
        dtype=torch.float32,
        device=basis.grid.device,
    ).unsqueeze(0)
    source_action = source_policy.action(observation).unsqueeze(0)
    effect_metric = (
        source_policy.value_effect_metric(
            observation, metric_isotropic_floor,
        )
        if effect_metric_mode == "critic"
        else None
    )
    if getattr(basis, "policy_centered", False):
        zero_innovation = torch.zeros_like(source_action)
        source_effect = source_context.acceleration(
            basis, state, zero_innovation,
        )
        target_effect = target_context.acceleration(
            basis, state, zero_innovation,
        )
        correction = target_context.transport_action(
            basis,
            state,
            source_effect,
            zero_innovation,
            regularization=damping,
            effect_metric=effect_metric,
        )
        base_action = (source_action + correction).clamp(-1.0, 1.0)
    else:
        source_effect = source_context.acceleration(
            basis, state, source_action,
        )
        target_effect = target_context.acceleration(
            basis, state, source_action,
        )
        base_action = target_context.transport_action(
            basis,
            state,
            source_effect,
            source_action,
            regularization=damping,
            effect_metric=effect_metric,
        ).clamp(-1.0, 1.0)
    normalized_state = (
        state / basis.state_scale
    ).clamp(-2.0, 2.0)
    effect_gap = (
        (target_effect - source_effect) / delta_scale
    ).clamp(-10.0, 10.0)
    action_gap = (base_action - source_action).clamp(-2.0, 2.0)
    features = torch.cat(
        (normalized_state, effect_gap, action_gap),
        dim=-1,
    )
    result = (
        base_action[0].cpu().numpy(),
        features[0].cpu().numpy().astype(np.float32),
    )
    if not return_details:
        return result
    _, source_gain = source_context.drift_and_gain(basis, state)
    details = {
        "state": state,
        "source_action": source_action,
        "source_effect": source_effect,
        "source_gain": source_gain,
        "effect_metric": effect_metric,
    }
    return (*result, details)


@torch.no_grad()
def decode_cached_effect_residual(
    residual,
    details,
    basis,
    target_context,
    args,
):
    residual_coordinate = torch.as_tensor(
        residual,
        dtype=torch.float32,
        device=basis.grid.device,
    ).unsqueeze(0)
    effect_delta = (
        details["source_gain"] @ (
            args.residual_scale * residual_coordinate
        ).unsqueeze(-1)
    ).squeeze(-1)
    state = details["state"]
    source_action = details["source_action"]
    desired_effect = details["source_effect"] + effect_delta
    if getattr(basis, "policy_centered", False):
        zero_innovation = torch.zeros_like(source_action)
        correction = target_context.transport_action(
            basis,
            state,
            desired_effect,
            zero_innovation,
            regularization=args.pullback_damping,
            effect_metric=details["effect_metric"],
        )
        action = source_action + correction
    else:
        action = target_context.transport_action(
            basis,
            state,
            desired_effect,
            source_action,
            regularization=args.pullback_damping,
            effect_metric=details["effect_metric"],
        )
    return action.clamp(-1.0, 1.0)[0].cpu().numpy()


@torch.no_grad()
def cognitive_effect_residual_action(
    observation,
    residual,
    source_policy,
    basis,
    source_context,
    target_context,
    args,
):
    _, _, details = cognitive_action_and_features(
        observation,
        source_policy,
        basis,
        source_context,
        target_context,
        torch.ones(
            source_context.coefficients.shape[-1],
            device=basis.grid.device,
        ),
        args.pullback_damping,
        args.effect_metric,
        args.metric_isotropic_floor,
        return_details=True,
    )
    return decode_cached_effect_residual(
        residual, details, basis, target_context, args,
    )


class CognitiveResidualHopper(gym.Env):
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
        transition_sink=None,
    ):
        super().__init__()
        self.environment = make_shifted_env(
            SHIFTS[args.target],
            args.seed + seed_offset,
            args.env,
        )()
        self.source_policy = source_policy
        self.basis = basis
        self.source_context = source_context
        self.estimator = estimator
        self.target_context = estimator.context()
        self.delta_scale = delta_scale
        self.args = args
        self.update_cognition = update_cognition
        self.transition_sink = transition_sink
        self.action_space = gym.spaces.Box(
            -1.0, 1.0, shape=(basis.action_dim,), dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(
            -np.inf,
            np.inf,
            shape=(2 * basis.state_dim + basis.action_dim,),
            dtype=np.float32,
        )
        self.raw_observation = None
        self.buffer = []
        self.cached_base_action = None
        self.cached_details = None

    def _features(self, observation):
        base_action, features, details = cognitive_action_and_features(
            observation,
            self.source_policy,
            self.basis,
            self.source_context,
            self.target_context,
            self.delta_scale,
            self.args.pullback_damping,
            getattr(self.args, "effect_metric", "identity"),
            getattr(self.args, "metric_isotropic_floor", 0.05),
            return_details=True,
        )
        self.cached_base_action = base_action
        self.cached_details = details
        return base_action, features

    def reset(self, *, seed=None, options=None):
        observation, info = self.environment.reset(
            seed=seed, options=options,
        )
        self.raw_observation = observation
        _, features = self._features(observation)
        return features, info

    def step(self, residual_action):
        base_action = self.cached_base_action
        if base_action is None:
            base_action, _ = self._features(self.raw_observation)
        if self.args.residual_space == "effect":
            physical_action = decode_cached_effect_residual(
                residual_action,
                self.cached_details,
                self.basis,
                self.target_context,
                self.args,
            )
        else:
            physical_action = np.clip(
                base_action
                + self.args.residual_scale
                * np.asarray(residual_action, dtype=np.float32),
                -1.0,
                1.0,
            )
        following, reward, terminated, truncated, info = (
            self.environment.step(physical_action)
        )
        if self.update_cognition or self.transition_sink is not None:
            nominal_action = self.source_policy.action(
                self.raw_observation,
            ).cpu().numpy()
            regressor_action = (
                physical_action - nominal_action
                if getattr(self.basis, "policy_centered", False)
                else physical_action
            )
            transition = (
                self.raw_observation.copy(),
                regressor_action.copy(),
                following - self.raw_observation,
            )
            if self.transition_sink is not None:
                self.transition_sink.append(transition)
            if self.update_cognition:
                self.buffer.append(
                    transition,
                )
        if self.update_cognition:
            if len(self.buffer) >= self.args.cognition_batch:
                state, action, delta = zip(*self.buffer)
                self.estimator.update(
                    torch.as_tensor(
                        np.asarray(state),
                        dtype=torch.float32,
                        device=self.basis.grid.device,
                    ),
                    torch.as_tensor(
                        np.asarray(action),
                        dtype=torch.float32,
                        device=self.basis.grid.device,
                    ),
                    torch.as_tensor(
                        np.asarray(delta),
                        dtype=torch.float32,
                        device=self.basis.grid.device,
                    ),
                )
                self.target_context = self.estimator.context()
                self.buffer.clear()
        self.raw_observation = following
        _, features = self._features(following)
        info["cognitive_base_action"] = base_action
        info["physical_action"] = physical_action
        return features, reward, terminated, truncated, info

    def close(self):
        self.environment.close()


@torch.no_grad()
def cognition_warmup(
    source_policy,
    basis,
    estimator,
    args,
    device,
):
    environment = make_shifted_env(
        SHIFTS[args.target], args.seed + 200, args.env,
    )()
    observation, _ = environment.reset(seed=args.seed + 200)
    states, actions, deltas = [], [], []
    replay_states = []
    generator = torch.Generator(device=device).manual_seed(args.seed + 3)
    for _ in range(args.cognition_warmup):
        replay_states.append(observation.copy())
        nominal_action = source_policy.action(observation)
        action = nominal_action
        action = (
            action
            + args.warmup_noise
            * torch.randn(
                action.shape,
                device=device,
                generator=generator,
            )
        ).clamp(-1.0, 1.0)
        following, _, terminated, truncated, _ = environment.step(
            action.cpu().numpy(),
        )
        states.append(observation.copy())
        regressor_action = (
            action - nominal_action
            if getattr(basis, "policy_centered", False)
            else action
        )
        actions.append(regressor_action.cpu().numpy())
        deltas.append(following - observation)
        if len(states) >= args.cognition_batch:
            estimator.update(
                torch.as_tensor(
                    np.asarray(states),
                    dtype=torch.float32,
                    device=device,
                ),
                torch.as_tensor(
                    np.asarray(actions),
                    dtype=torch.float32,
                    device=device,
                ),
                torch.as_tensor(
                    np.asarray(deltas),
                    dtype=torch.float32,
                    device=device,
                ),
            )
            states, actions, deltas = [], [], []
        if terminated or truncated:
            observation, _ = environment.reset()
        else:
            observation = following
    environment.close()
    return np.asarray(replay_states, dtype=np.float32)


def evaluate(
    model,
    source_policy,
    basis,
    source_context,
    target_context,
    delta_scale,
    args,
    episodes,
):
    environment = make_shifted_env(
        SHIFTS[args.target], args.seed + 10000, args.env,
    )()
    returns = []
    lengths = []
    healthy = 0
    base_action_abs = []
    residual_abs = []
    for episode in range(episodes):
        observation, _ = environment.reset(seed=args.seed + 10000 + episode)
        total = 0.0
        length = 0
        while True:
            base_action, features, details = cognitive_action_and_features(
                observation,
                source_policy,
                basis,
                source_context,
                target_context,
                delta_scale,
                args.pullback_damping,
                getattr(args, "effect_metric", "identity"),
                getattr(args, "metric_isotropic_floor", 0.05),
                return_details=True,
            )
            if model is None:
                residual = np.zeros(basis.action_dim, dtype=np.float32)
            else:
                residual, _ = model.predict(
                    features, deterministic=True,
                )
            if args.residual_space == "effect":
                action = decode_cached_effect_residual(
                    residual,
                    details,
                    basis,
                    target_context,
                    args,
                )
            else:
                action = np.clip(
                    base_action + args.residual_scale * residual,
                    -1.0, 1.0,
                )
            observation, reward, terminated, truncated, _ = (
                environment.step(action)
            )
            total += float(reward)
            length += 1
            base_action_abs.append(np.abs(base_action))
            residual_abs.append(np.abs(residual))
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
        "healthy_completion_rate": healthy / episodes,
        "base_action_abs_p95": float(np.quantile(base_action_abs, 0.95)),
        "residual_abs_mean": float(np.mean(residual_abs)),
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
        source_policy, basis, source_context, delta_scale = self.components
        metrics = evaluate(
            self.model,
            source_policy,
            basis,
            source_context,
            self.train_environment.target_context,
            delta_scale,
            self.args,
            self.args.evaluation_episodes,
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
    torch.manual_seed(args.seed)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    print(f"device={device}", flush=True)
    source_policy = FrozenSourcePolicy(
        args.source_model,
        args.source_norm,
        device,
        args.seed,
        env=args.env,
    )
    basis, source_context, estimator, delta_scale = load_cognition(
        args, device,
    )
    cognition_warmup(
        source_policy, basis, estimator, args, device,
    )
    train_environment = CognitiveResidualHopper(
        source_policy,
        basis,
        source_context,
        estimator,
        delta_scale,
        args,
        update_cognition=True,
        seed_offset=500,
    )
    initial = evaluate(
        None,
        source_policy,
        basis,
        source_context,
        train_environment.target_context,
        delta_scale,
        args,
        args.evaluation_episodes,
    )
    initial_record = {
        "target_transitions": args.cognition_warmup,
        **initial,
    }
    print(initial_record, flush=True)
    vector = DummyVecEnv([lambda: train_environment])
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
        clip_range=args.clip_ratio,
        ent_coef=args.entropy_coefficient,
        seed=args.seed,
        device=device,
        verbose=0,
    )
    torch.nn.init.zeros_(model.policy.action_net.weight)
    torch.nn.init.zeros_(model.policy.action_net.bias)
    model.policy.log_std.data.fill_(args.initial_log_std)
    callback = RecoveryCallback(
        train_environment,
        (source_policy, basis, source_context, delta_scale),
        args,
    )
    model.learn(
        total_timesteps=args.decision_transitions,
        callback=callback,
    )
    final = evaluate(
        model,
        source_policy,
        basis,
        source_context,
        train_environment.target_context,
        delta_scale,
        args,
        args.evaluation_episodes,
    )
    final_record = {
        "target_transitions": (
            args.cognition_warmup + args.decision_transitions
        ),
        **final,
    }
    history = [initial_record, *callback.history]
    if not history or history[-1]["target_transitions"] != (
        final_record["target_transitions"]
    ):
        history.append(final_record)
    output = {
        "experiment": "HopperJointProtoKANResidualPPO",
        "seed": args.seed,
        "device": str(device),
        "target": args.target,
        "physical_parameters_visible_to_learner": False,
        "cognition_reward_free": True,
        "decision_uses_real_reward": True,
        "residual_space": args.residual_space,
        "config": vars(args),
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
        "--env", choices=tuple(ENVS), default="hopper",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
    )
    parser.add_argument(
        "--target",
        choices=tuple(name for name in SHIFTS if name != "source"),
        default="combo_mild",
    )
    parser.add_argument("--cognition-warmup", type=int, default=1024)
    parser.add_argument("--cognition-batch", type=int, default=128)
    parser.add_argument("--warmup-noise", type=float, default=0.05)
    parser.add_argument("--decision-transitions", type=int, default=65536)
    parser.add_argument("--rollout-steps", type=int, default=1024)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--update-epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--entropy-coefficient", type=float, default=0.0)
    parser.add_argument("--initial-log-std", type=float, default=-1.5)
    parser.add_argument("--residual-scale", type=float, default=0.25)
    parser.add_argument(
        "--residual-space",
        choices=("action", "effect"),
        default="action",
    )
    parser.add_argument("--pullback-damping", type=float, default=0.05)
    parser.add_argument(
        "--effect-metric",
        choices=("identity", "critic"),
        default="critic",
    )
    parser.add_argument("--metric-isotropic-floor", type=float, default=0.05)
    parser.add_argument("--evaluate-every", type=int, default=8192)
    parser.add_argument("--evaluation-episodes", type=int, default=10)
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
        default="results/hopper_source_protokan_cognition_seed1811.pt",
    )
    parser.add_argument(
        "--model-out",
        default="results/hopper_joint_residual_ppo_seed1811",
    )
    parser.add_argument(
        "--json-out",
        default="results/hopper_joint_residual_ppo_seed1811.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
