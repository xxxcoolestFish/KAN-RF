"""Shared-cognition, two-timescale parallel Hopper adaptation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from scripts.validate_hopper_joint_online_adaptation import (
    CognitiveResidualHopper,
    FrozenSourcePolicy,
    cognition_warmup,
    evaluate,
    load_cognition,
)
from scripts.diagnose_hopper_pullback_effect import (
    fit_orthogonal_control_transform,
)


def update_shared_cognition(estimator, transitions, args, device):
    for start in range(0, len(transitions), args.cognition_batch):
        batch = transitions[start:start + args.cognition_batch]
        state, action, delta = zip(*batch)
        estimator.update(
            torch.as_tensor(
                np.asarray(state),
                dtype=torch.float32,
                device=device,
            ),
            torch.as_tensor(
                np.asarray(action),
                dtype=torch.float32,
                device=device,
            ),
            torch.as_tensor(
                np.asarray(delta),
                dtype=torch.float32,
                device=device,
            ),
        )


def publish_context(environments, estimator=None, context=None):
    if context is None:
        context = estimator.context()
    for environment in environments:
        environment.target_context = context
        environment.cached_base_action = None
        environment.cached_details = None
    return context


def main(args):
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    print(f"device={device}", flush=True)
    source_policy = FrozenSourcePolicy(
        args.source_model,
        args.source_norm,
        device,
        args.seed,
    )
    basis, source_context, estimator, delta_scale = load_cognition(
        args, device,
    )
    if args.cognition_update == "orthogonal_transform":
        context, _ = fit_orthogonal_control_transform(
            source_policy,
            basis,
            source_context,
            args,
            device,
        )
    else:
        cognition_warmup(
            source_policy, basis, estimator, args, device,
        )
        context = estimator.context()
    shared_transitions = []
    environments = [
        CognitiveResidualHopper(
            source_policy,
            basis,
            source_context,
            estimator,
            delta_scale,
            args,
            update_cognition=False,
            seed_offset=500 + index,
            transition_sink=shared_transitions,
        )
        for index in range(args.parallel_envs)
    ]
    context = publish_context(environments, context=context)
    initial = evaluate(
        None,
        source_policy,
        basis,
        source_context,
        context,
        delta_scale,
        args,
        args.evaluation_episodes,
    )
    history = [{
        "phase": 0,
        "target_transitions": args.cognition_warmup,
        **initial,
    }]
    print(history[-1], flush=True)

    vector = DummyVecEnv([
        (lambda environment=environment: environment)
        for environment in environments
    ])
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

    transitions_per_phase = args.parallel_envs * args.rollout_steps
    completed = 0
    phase = 0
    while completed < args.decision_transitions:
        phase += 1
        shared_transitions.clear()
        model._last_obs = None
        model.learn(
            total_timesteps=transitions_per_phase,
            reset_num_timesteps=phase == 1,
        )
        observed = len(shared_transitions)
        if observed != transitions_per_phase:
            raise RuntimeError(
                f"expected {transitions_per_phase} transitions, got {observed}",
            )
        if args.cognition_update == "recursive_full":
            update_shared_cognition(
                estimator, shared_transitions, args, device,
            )
            context = publish_context(environments, estimator=estimator)
        completed += observed
        if (
            phase % args.evaluate_every_phases == 0
            or completed >= args.decision_transitions
        ):
            metrics = evaluate(
                model,
                source_policy,
                basis,
                source_context,
                context,
                delta_scale,
                args,
                args.evaluation_episodes,
            )
            record = {
                "phase": phase,
                "phase_transitions": observed,
                "target_transitions": args.cognition_warmup + completed,
                **metrics,
            }
            history.append(record)
            print(record, flush=True)

    output = {
        "experiment": "HopperSharedCognitionTwoTimescalePPO",
        "seed": args.seed,
        "device": str(device),
        "target": args.target,
        "parallel_envs": args.parallel_envs,
        "residual_space": args.residual_space,
        "physical_parameters_visible_to_learner": False,
        "cognition_reward_free": True,
        "cognition_online_between_every_policy_phase": (
            args.cognition_update == "recursive_full"
        ),
        "cognition_update": args.cognition_update,
        "shared_cognition_across_environments": True,
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
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--target", default="combo_mild")
    parser.add_argument("--parallel-envs", type=int, default=4)
    parser.add_argument("--cognition-warmup", type=int, default=256)
    parser.add_argument("--cognition-batch", type=int, default=64)
    parser.add_argument("--warmup-noise", type=float, default=0.05)
    parser.add_argument(
        "--warmup-exploration",
        choices=("gaussian_clipped", "symmetric"),
        default="gaussian_clipped",
    )
    parser.add_argument(
        "--cognition-update",
        choices=("recursive_full", "orthogonal_transform"),
        default="recursive_full",
    )
    parser.add_argument("--transform-ridge", type=float, default=10.0)
    parser.add_argument("--decision-transitions", type=int, default=2048)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--minibatch-size", type=int, default=128)
    parser.add_argument("--update-epochs", type=int, default=4)
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
        default="effect",
    )
    parser.add_argument("--pullback-damping", type=float, default=0.05)
    parser.add_argument(
        "--effect-metric",
        choices=("identity", "critic"),
        default="identity",
    )
    parser.add_argument("--metric-isotropic-floor", type=float, default=0.05)
    parser.add_argument("--evaluation-episodes", type=int, default=3)
    parser.add_argument("--evaluate-every-phases", type=int, default=1)
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
        default="results/hopper_two_timescale_effect_seed1811",
    )
    parser.add_argument(
        "--json-out",
        default="results/hopper_two_timescale_effect_seed1811.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
