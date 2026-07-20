"""Second cognition attempt: multistep and local-secant predictive fitting."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

import torch

from cpbn import OracleAcrobotDynamics, random_states
from cpbn.cognition import ProtoKANDynamics
from cpbn.time_varying_tube import (
    TimeVaryingTubeSet,
    apply_tangent_error,
    plan_continuous_cem_route,
    tangent_error,
)
from scripts.validate_learned_cognitive_tubes import (
    execute_hidden_controller,
    jacobian_metrics,
    prediction_metrics,
    replay_route,
    sample_batch,
)


def curriculum_horizon(step, total, maximum):
    levels = [value for value in (1, 2, 4, 8, 16, 24) if value <= maximum]
    index = min(len(levels) - 1, step * len(levels) // total)
    return levels[index]


def local_secant_loss(model, source, count, generator, noise):
    state, action, next_state = sample_batch(source, count, generator, 0.85)
    state_error = torch.randn(count, 4, generator=generator) * noise
    state_error[:, 2:] *= 0.75
    action_error = torch.randn(count, 1, generator=generator) * (2.0 * noise)
    neighbor_state = apply_tangent_error(state, state_error)
    neighbor_action = (action + action_error).clamp(-1.0, 1.0)
    with torch.no_grad():
        neighbor_next = source(neighbor_state, neighbor_action)
        true_response = tangent_error(neighbor_next, next_state)
    model_next = model(state, action)
    model_neighbor_next = model(neighbor_state, neighbor_action)
    model_response = tangent_error(model_neighbor_next, model_next)
    return (model_response - true_response).square().mean()


def train_cognition(model, source, args):
    generator = torch.Generator().manual_seed(args.seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.cognitive_lr)
    history = []
    for step in range(args.cognitive_steps):
        horizon = curriculum_horizon(
            step, args.cognitive_steps, args.maximum_training_horizon,
        )
        true_state = random_states(args.cognitive_batch, generator)
        model_state = true_state.clone()
        prediction_loss = torch.zeros(())
        weight_sum = 0.0
        for index in range(horizon):
            action = torch.rand(
                args.cognitive_batch, 1, generator=generator,
            ) * 2.0 - 1.0
            with torch.no_grad():
                true_state = source(true_state, action)
            model_state = model(model_state, action)
            weight = 1.0 / (index + 1) ** 0.5
            prediction_loss = prediction_loss + weight * tangent_error(
                model_state, true_state,
            ).square().mean()
            weight_sum += weight
        prediction_loss = prediction_loss / weight_sum
        secant = local_secant_loss(
            model, source, args.secant_batch, generator, args.secant_noise,
        )
        repulsion = model.network.repulsion_loss(tau=0.08)
        loss = (
            prediction_loss
            + args.secant_weight * secant
            + args.repulsion_weight * repulsion
        )
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if step == 0 or (step + 1) % args.log_every == 0:
            record = {
                "step": step + 1,
                "training_horizon": horizon,
                "multistep_prediction_loss": float(prediction_loss.detach()),
                "local_secant_loss": float(secant.detach()),
                "repulsion_loss": float(repulsion.detach()),
            }
            history.append(record)
            print(json.dumps(record), flush=True)
    return history


def run(args):
    torch.manual_seed(args.seed)
    source = OracleAcrobotDynamics()
    cognition = ProtoKANDynamics(args.hidden_dim, args.n_prototypes)
    training = train_cognition(cognition, source, args)
    prediction = prediction_metrics(
        cognition, source, args.validation_count,
        args.validation_horizon, args.test_seed,
    )
    jacobian = jacobian_metrics(
        cognition, source, args.jacobian_count, args.test_seed + 100,
    )
    route = plan_continuous_cem_route(
        cognition,
        segment_count=args.route_segments,
        segment_steps=args.segment_steps,
        population=args.cem_population,
        elite_count=args.cem_elite,
        iterations=args.cem_iterations,
        seed=args.seed + 200,
    )
    route_replay = replay_route(route, source)
    tubes = TimeVaryingTubeSet(
        cognition, route,
        construction_samples=args.construction_samples,
        quantile=args.tube_quantile,
        seed=args.seed + 300,
    )
    predicted_certifier = tubes.evaluate_hidden_lqr(
        args.trials_per_edge, args.test_seed + 200,
    )
    real_certifier = execute_hidden_controller(
        tubes, source, args.trials_per_edge, args.test_seed + 300,
    )
    gates = {
        "one_step_tangent_rmse_below_0_02": (
            prediction["one_step_tangent_rmse"] < 0.02
        ),
        "action_jacobian_cosine_above_0_90": (
            jacobian["action_jacobian_mean_cosine"] > 0.90
        ),
        "learned_route_succeeds_in_real_source": route_replay["route_succeeds"],
        "real_tube_minimum_completion_above_0_90": (
            real_certifier["minimum_edge_completion"] >= 0.90
        ),
    }
    return {
        "training_history": training,
        "prediction": prediction,
        "jacobian": jacobian,
        "learned_route": asdict(route.diagnostics),
        "real_replay_of_learned_route": route_replay,
        "tube_construction": asdict(tubes.diagnostics),
        "predicted_dynamics_certifier": predicted_certifier,
        "real_source_certifier": real_certifier,
        "gates": gates,
        "passed": all(gates.values()),
        "cognitive_parameter_count": sum(
            parameter.numel() for parameter in cognition.parameters()
        ),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--n-prototypes", type=int, default=8)
    parser.add_argument("--cognitive-steps", type=int, default=1600)
    parser.add_argument("--cognitive-batch", type=int, default=128)
    parser.add_argument("--maximum-training-horizon", type=int, default=16)
    parser.add_argument("--secant-batch", type=int, default=128)
    parser.add_argument("--secant-noise", type=float, default=0.025)
    parser.add_argument("--secant-weight", type=float, default=0.5)
    parser.add_argument("--cognitive-lr", type=float, default=2e-3)
    parser.add_argument("--repulsion-weight", type=float, default=1e-5)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--validation-count", type=int, default=1024)
    parser.add_argument("--validation-horizon", type=int, default=24)
    parser.add_argument("--jacobian-count", type=int, default=24)
    parser.add_argument("--route-segments", type=int, default=20)
    parser.add_argument("--segment-steps", type=int, default=24)
    parser.add_argument("--cem-population", type=int, default=1024)
    parser.add_argument("--cem-elite", type=int, default=64)
    parser.add_argument("--cem-iterations", type=int, default=10)
    parser.add_argument("--construction-samples", type=int, default=512)
    parser.add_argument("--tube-quantile", type=float, default=0.99)
    parser.add_argument("--trials-per-edge", type=int, default=64)
    parser.add_argument("--test-seed", type=int, default=20260805)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-out", type=str, default="")
    return parser.parse_args()


def main():
    args = parse_args()
    output = {
        "experiment": "MultistepSecantProtoKANCognitiveTubeValidation",
        "config": vars(args),
        "result": run(args),
    }
    text = json.dumps(output, ensure_ascii=False, indent=2)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
