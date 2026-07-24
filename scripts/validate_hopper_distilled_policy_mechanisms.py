"""Distill historical Actors into one cognition-gated policy mechanism net."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

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


@torch.no_grad()
def collect_expert_states(policy, environment_name, count, args, seed):
    environment = make_shifted_env(SHIFTS[environment_name], seed)()
    observation, _ = environment.reset(seed=seed)
    states = []
    for _ in range(count):
        states.append(observation.copy())
        action = policy.action(observation)
        following, _, terminated, truncated, _ = environment.step(
            action.cpu().numpy(),
        )
        observation = (
            environment.reset()[0]
            if terminated or truncated
            else following
        )
    environment.close()
    return np.asarray(states, dtype=np.float32)


def train_decoder(decoder, source, experts, states, args, device):
    optimizer = torch.optim.Adam(
        decoder.parameters(), lr=args.learning_rate,
    )
    generator = torch.Generator(device=device).manual_seed(args.seed + 31)
    tensors = [
        torch.as_tensor(state, device=device)
        for state in states
    ]
    targets = []
    with torch.no_grad():
        for state, expert in zip(states, experts):
            target = []
            for observation in state:
                target.append(
                    (
                        expert.action(observation)
                        - source.action(observation)
                    ).cpu().numpy()
                )
            targets.append(torch.as_tensor(np.asarray(target), device=device))
    history = []
    for step in range(1, args.gradient_steps + 1):
        loss = torch.zeros((), device=device)
        for index, (state, target) in enumerate(zip(tensors, targets)):
            sample = torch.randint(
                state.shape[0],
                (min(args.batch_size, state.shape[0]),),
                generator=generator,
                device=device,
            )
            effect = decoder.mechanism_effects(state[sample])[:, index, :]
            loss = loss + (effect - target[sample]).square().mean()
        loss = loss / len(tensors)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % args.report_every == 0:
            record = {"step": step, "loss": float(loss.detach())}
            history.append(record)
            print({"stage": "distill", **record}, flush=True)
    return history


@torch.no_grad()
def evaluate(source, decoder, coefficients, target, args, device):
    environment = make_shifted_env(
        SHIFTS[target], args.seed + 10000,
    )()
    returns, lengths = [], []
    healthy = 0
    corrections = []
    coefficient_batch = coefficients.unsqueeze(0)
    for episode in range(args.evaluation_episodes):
        observation, _ = environment.reset(
            seed=args.seed + 10000 + episode,
        )
        total = 0.0
        length = 0
        while True:
            state = torch.as_tensor(
                observation, dtype=torch.float32, device=device,
            ).unsqueeze(0)
            correction = decoder(state, coefficient_batch)[0]
            action = (
                source.action(observation) + correction
            ).clamp(-1.0, 1.0)
            corrections.extend(torch.abs(correction).cpu().tolist())
            observation, reward, terminated, truncated, _ = (
                environment.step(action.cpu().numpy())
            )
            total += float(reward)
            length += 1
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
        "correction_abs_mean": float(np.mean(corrections)),
        "correction_abs_p95": float(np.quantile(corrections, 0.95)),
    }


def main(args):
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print({"stage": "setup", "device": str(device)}, flush=True)
    source = FrozenSourcePolicy(
        args.source_model, args.source_norm, device, args.seed,
    )
    experts = [
        FrozenSourcePolicy(model, norm, device, args.seed + index + 1)
        for index, (model, norm) in enumerate(
            zip(args.mechanism_actor_models, args.mechanism_actor_norms)
        )
    ]
    states = [
        collect_expert_states(
            expert,
            environment,
            args.expert_transitions,
            args,
            args.seed + 11000 + index,
        )
        for index, (expert, environment) in enumerate(
            zip(experts, args.mechanism_environments)
        )
    ]
    decoder = PolicyMechanismDecoder(
        source.mean, source.variance,
        mechanism_dim=len(experts),
    ).to(device)
    history = train_decoder(
        decoder, source, experts, states, args, device,
    )

    basis, source_context, _, delta_scale = load_cognition(args, device)
    payload = torch.load(
        args.mechanism_checkpoint,
        map_location=device,
        weights_only=True,
    )
    mechanism_model = GlobalMechanismKANDynamics(
        source_context, payload["mechanisms"].to(device),
    )
    estimator = RecursiveGlobalMechanismEstimator(
        mechanism_model,
        basis,
        delta_scale,
        ridge=args.mechanism_latent_ridge,
    )
    adaptation = collect_transitions(
        source,
        args.target,
        args.cognition_warmup,
        args,
        device,
        12000,
    )
    estimator.update(
        adaptation["state"],
        adaptation["innovation"],
        adaptation["delta"],
    )
    scale = payload["latent_scale"].to(device)
    target_coordinate = estimator.latent() / scale
    training_coordinates = payload["training_latents"].to(device) / scale
    coefficients = torch.linalg.lstsq(
        training_coordinates.T,
        target_coordinate,
    ).solution
    transported = evaluate(
        source, decoder, coefficients, args.target, args, device,
    )
    frozen = evaluate(
        source,
        decoder,
        torch.zeros_like(coefficients),
        args.target,
        args,
        device,
    )
    output = {
        "experiment": "HopperDistilledPolicyMechanismGate",
        "seed": args.seed,
        "device": str(device),
        "target": args.target,
        "physical_parameters_visible_to_learner": False,
        "historical_policy_coefficients": coefficients.cpu().tolist(),
        "distillation": history,
        "frozen_source": frozen,
        "distilled_transport": transported,
        "improvement": transported["mean_return"] - frozen["mean_return"],
        "config": vars(args),
    }
    Path(args.checkpoint_out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "decoder": decoder.state_dict(),
            "training_coordinates": training_coordinates.cpu(),
        },
        args.checkpoint_out,
    )
    Path(args.json_out).write_text(
        json.dumps(output, indent=2) + "\n",
        encoding="utf-8",
    )
    print({"stage": "summary", **output}, flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1811)
    parser.add_argument("--target", default="combo_medium")
    parser.add_argument(
        "--mechanism-environments",
        nargs="+",
        default=("payload_125", "friction_070", "actuator_080"),
    )
    parser.add_argument("--expert-transitions", type=int, default=2048)
    parser.add_argument("--gradient-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--report-every", type=int, default=200)
    parser.add_argument("--cognition-warmup", type=int, default=512)
    parser.add_argument("--exploration-noise", type=float, default=0.2)
    parser.add_argument("--evaluation-episodes", type=int, default=5)
    parser.add_argument("--mechanism-latent-ridge", type=float, default=1e-2)
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
        "--mechanism-actor-models",
        nargs="+",
        default=(
            "results/hopper_payload125_mechanism_actor_frozennorm_80k_seed1811.zip",
            "results/hopper_friction070_mechanism_actor_frozennorm_80k_seed1811.zip",
            "results/hopper_actuator080_mechanism_actor_frozennorm_80k_seed1811.zip",
        ),
    )
    parser.add_argument(
        "--mechanism-actor-norms",
        nargs="+",
        default=(
            "results/hopper_payload125_mechanism_actor_frozennorm_16k_norm_seed1811.pkl",
            "results/hopper_friction070_mechanism_actor_frozennorm_16k_norm_seed1811.pkl",
            "results/hopper_actuator080_mechanism_actor_frozennorm_16k_norm_seed1811.pkl",
        ),
    )
    parser.add_argument(
        "--checkpoint-out",
        default="results/hopper_policy_mechanism_decoder_seed1811.pt",
    )
    parser.add_argument(
        "--json-out",
        default="results/hopper_distilled_policy_mechanisms_seed1811.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
