"""Validate whether a source-only ProtoKAN can construct executable tubes."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

import torch
import torch.nn.functional as F

from cpbn import OracleAcrobotDynamics, random_states, tip_height
from cpbn.cognition import ProtoKANDynamics
from cpbn.time_varying_tube import (
    TimeVaryingTubeSet,
    apply_tangent_error,
    plan_continuous_cem_route,
    tangent_error,
)


def sample_batch(dynamics, count, generator, action_limit=1.0):
    state = random_states(count, generator)
    action = (
        torch.rand(count, 1, generator=generator) * 2.0 - 1.0
    ) * action_limit
    with torch.no_grad():
        next_state = dynamics(state, action)
    return state, action, next_state


def train_cognition(model, source, args):
    generator = torch.Generator().manual_seed(args.seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.cognitive_lr)
    history = []
    for step in range(args.cognitive_steps):
        state, action, next_state = sample_batch(
            source, args.cognitive_batch, generator,
        )
        prediction_loss = model.prediction_loss(state, action, next_state)
        repulsion = model.network.repulsion_loss(tau=0.08)
        loss = prediction_loss + args.repulsion_weight * repulsion
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if step == 0 or (step + 1) % args.log_every == 0:
            record = {
                "step": step + 1,
                "prediction_loss": float(prediction_loss.detach()),
                "repulsion_loss": float(repulsion.detach()),
            }
            history.append(record)
            print(json.dumps(record), flush=True)
    return history


@torch.no_grad()
def prediction_metrics(model, source, count, horizon, seed):
    generator = torch.Generator().manual_seed(seed)
    state, action, next_state = sample_batch(source, count, generator)
    prediction = model(state, action)
    one_step_error = tangent_error(prediction, next_state).norm(dim=-1)

    true_state = random_states(count, generator)
    model_state = true_state.clone()
    rollout_errors = []
    for _ in range(horizon):
        action = torch.rand(count, 1, generator=generator) * 2.0 - 1.0
        true_state = source(true_state, action)
        model_state = model(model_state, action)
        rollout_errors.append(tangent_error(model_state, true_state).norm(dim=-1))
    rollout_errors = torch.stack(rollout_errors, dim=1)
    return {
        "one_step_raw_mse": float(F.mse_loss(prediction, next_state)),
        "one_step_tangent_rmse": float(one_step_error.square().mean().sqrt()),
        "one_step_tangent_p95": float(torch.quantile(one_step_error, 0.95)),
        "rollout_horizon": horizon,
        "rollout_mean_tangent_error": float(rollout_errors.mean()),
        "rollout_final_tangent_rmse": float(
            rollout_errors[:, -1].square().mean().sqrt()
        ),
    }


def local_jacobians(dynamics, state, action):
    with torch.no_grad():
        next_center = dynamics(state.view(1, 6), action.view(1, 1)).squeeze(0)
    zero_state = torch.zeros(4, requires_grad=True)
    zero_action = torch.zeros(1, requires_grad=True)

    def local_map(state_error, action_error):
        perturbed = apply_tangent_error(state, state_error).view(1, 6)
        local_action = (action + action_error).view(1, 1)
        predicted = dynamics(perturbed, local_action)
        return tangent_error(predicted, next_center.view(1, 6)).squeeze(0)

    return tuple(
        value.detach() for value in torch.autograd.functional.jacobian(
            local_map, (zero_state, zero_action), vectorize=True,
        )
    )


def jacobian_metrics(model, source, count, seed):
    generator = torch.Generator().manual_seed(seed)
    state = random_states(count, generator)
    action = torch.rand(count, 1, generator=generator) * 1.6 - 0.8
    state_cosine, action_cosine = [], []
    state_relative, action_relative = [], []
    for index in range(count):
        true_a, true_b = local_jacobians(source, state[index], action[index])
        model_a, model_b = local_jacobians(model, state[index], action[index])
        state_cosine.append(F.cosine_similarity(
            true_a.flatten(), model_a.flatten(), dim=0,
        ))
        action_cosine.append(F.cosine_similarity(
            true_b.flatten(), model_b.flatten(), dim=0,
        ))
        state_relative.append(
            (true_a - model_a).norm() / true_a.norm().clamp_min(1e-8)
        )
        action_relative.append(
            (true_b - model_b).norm() / true_b.norm().clamp_min(1e-8)
        )
    state_cosine = torch.stack(state_cosine)
    action_cosine = torch.stack(action_cosine)
    state_relative = torch.stack(state_relative)
    action_relative = torch.stack(action_relative)
    return {
        "sample_count": count,
        "state_jacobian_mean_cosine": float(state_cosine.mean()),
        "state_jacobian_mean_relative_error": float(state_relative.mean()),
        "action_jacobian_mean_cosine": float(action_cosine.mean()),
        "action_jacobian_min_cosine": float(action_cosine.min()),
        "action_jacobian_mean_relative_error": float(action_relative.mean()),
    }


@torch.no_grad()
def replay_route(route, execution_dynamics):
    state = route.states[0].view(1, 6).clone()
    maximum_height = -float("inf")
    success_step = -1
    state_errors = []
    for segment in range(route.diagnostics.segment_count):
        action = route.actions[segment].view(1, 1)
        for local_step in range(route.diagnostics.segment_steps):
            state = execution_dynamics(state, action)
            step = segment * route.diagnostics.segment_steps + local_step + 1
            height = float(tip_height(state))
            maximum_height = max(maximum_height, height)
            if success_step < 0 and height >= 1.0:
                success_step = step
            state_errors.append(
                tangent_error(state, route.states[step].view(1, 6)).norm(dim=-1)
            )
    state_errors = torch.cat(state_errors)
    return {
        "maximum_height": maximum_height,
        "success_step": success_step,
        "mean_state_error": float(state_errors.mean()),
        "maximum_state_error": float(state_errors.max()),
        "route_succeeds": success_step >= 0,
    }


@torch.no_grad()
def execute_hidden_controller(tubes, execution_dynamics, trials, seed):
    generator = torch.Generator().manual_seed(seed)
    edge = torch.arange(tubes.edge_count).repeat_interleave(trials)
    state = tubes.sample_initial(edge, generator)
    stayed_inside = torch.ones(edge.shape[0], dtype=torch.bool)
    for step in range(tubes.horizon):
        center = tubes.centers[edge, step]
        error = tangent_error(state, center)
        gain = tubes._construction_gains[edge, step]
        action = tubes._construction_actions[edge].unsqueeze(-1) + torch.bmm(
            gain, error.unsqueeze(-1),
        ).squeeze(-1)
        state = execution_dynamics(state, action.clamp(-1.0, 1.0))
        phase = torch.full_like(edge, step + 1)
        stayed_inside &= tubes.normalized_distance(state, edge, phase) <= 1.0
    final_phase = torch.full_like(edge, tubes.horizon)
    completed = tubes.normalized_distance(state, edge, final_phase) <= 1.0
    per_edge = completed.view(tubes.edge_count, trials).float().mean(dim=1)
    return {
        "completion_rate": float(completed.float().mean()),
        "per_edge_completion": per_edge.tolist(),
        "minimum_edge_completion": float(per_edge.min()),
        "full_tube_adherence": float(stayed_inside.float().mean()),
    }


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
    parser.add_argument("--cognitive-steps", type=int, default=1200)
    parser.add_argument("--cognitive-batch", type=int, default=512)
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
        "experiment": "SourceProtoKANCognitiveTubeValidation",
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
