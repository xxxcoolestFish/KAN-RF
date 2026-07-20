"""Validate the actor-free Bellman layer with exact source dynamics.

This is deliberately a bounded diagnostic, not the final learned-cognition
experiment.  It separates action-solver correctness from value propagation.
"""

from __future__ import annotations

import argparse
import copy
import json

import torch
import torch.nn.functional as F

from cpbn import (
    GOAL,
    SOURCE_FACTOR,
    ImplicitBellmanAction,
    OracleAcrobotDynamics,
    ValueNetwork,
    bellman_return,
    grid_best_action,
    random_states,
    reset_down_states,
    task_reward,
    tip_height,
)


def stationarity_metrics(value, dynamics, state, goal, action, gamma, floor):
    action = action.detach().clone().requires_grad_(True)
    objective, _, _, _ = bellman_return(
        value, dynamics, state, goal, action, gamma,
    )
    gradient = torch.autograd.grad(objective.sum(), action, create_graph=True)[0]
    curvature = torch.autograd.grad(gradient.sum(), action)[0]
    at_lower = action.detach() <= -0.999
    at_upper = action.detach() >= 0.999
    interior = ~(at_lower | at_upper)
    projected_residual = torch.where(
        interior,
        gradient.detach().abs(),
        torch.where(
            at_lower,
            gradient.detach().clamp_min(0.0),
            (-gradient.detach()).clamp_min(0.0),
        ),
    )
    kkt = projected_residual <= 1e-3
    return {
        "mean_abs_projected_stationarity": float(projected_residual.mean()),
        "kkt_fraction": float(kkt.float().mean()),
        "locally_concave_fraction": float((curvature < -floor).float().mean()),
        "mean_curvature": float(curvature.mean()),
        "minimum_curvature": float(curvature.min()),
        "maximum_curvature": float(curvature.max()),
        "boundary_fraction": float((at_lower | at_upper).float().mean()),
    }


def action_landscape_audit(value, dynamics, goal, action_layer, args, seed):
    generator = torch.Generator().manual_seed(seed)
    state = random_states(args.audit_count, generator)
    goal_batch = goal.expand(args.audit_count, -1)
    initial = torch.rand(args.audit_count, 1, generator=generator) * 2.0 - 1.0
    implicit = action_layer(value, dynamics, state, goal_batch, initial)
    implicit_objective, _, _, _ = bellman_return(
        value, dynamics, state, goal_batch, implicit, args.gamma,
    )
    grid_action, grid_objective = grid_best_action(
        value, dynamics, state, goal_batch, args.grid_points, args.gamma,
    )
    return {
        "grid_points": args.grid_points,
        "mean_grid_regret": float((grid_objective - implicit_objective).mean()),
        "maximum_grid_regret": float((grid_objective - implicit_objective).max()),
        "mean_action_difference_from_grid": float((grid_action - implicit).abs().mean()),
        "grid_boundary_fraction": float((grid_action.abs() >= 0.999).float().mean()),
        **stationarity_metrics(
            value, dynamics, state, goal_batch, implicit, args.gamma,
            args.curvature_floor,
        ),
    }


@torch.no_grad()
def evaluate(value, dynamics, goal, action_layer, args, seed):
    generator = torch.Generator().manual_seed(seed)
    state = reset_down_states(args.test_count, generator=generator)
    goal_batch = goal.expand(args.test_count, -1)
    previous_action = torch.zeros(args.test_count, 1)
    maximum = torch.full((args.test_count,), -float("inf"))
    success = torch.zeros(args.test_count, dtype=torch.bool)
    total_reward = torch.zeros(args.test_count)
    for _ in range(args.eval_steps):
        with torch.enable_grad():
            action = action_layer(
                value, dynamics, state, goal_batch, previous_action,
            )
        next_state = dynamics(state, action)
        reward, done = task_reward(state, next_state, action)
        maximum = torch.maximum(maximum, tip_height(next_state))
        success |= done
        total_reward += reward
        state, previous_action = next_state, action
    return {
        "success_count": int(success.sum()),
        "success_rate": float(success.float().mean()),
        "mean_max_height": float(maximum.mean()),
        "max_height": float(maximum.max()),
        "mean_undiscounted_reward": float(total_reward.mean()),
    }


def train(args):
    torch.manual_seed(args.seed)
    dynamics = OracleAcrobotDynamics(args.factor)
    goal = GOAL.view(1, -1)
    value = ValueNetwork(args.hidden_dim)
    torch.nn.init.zeros_(value.net[-1].weight)
    torch.nn.init.zeros_(value.net[-1].bias)
    target_value = copy.deepcopy(value)
    for parameter in target_value.parameters():
        parameter.requires_grad = False
    optimizer = torch.optim.Adam(value.parameters(), lr=args.value_lr)
    action_layer = ImplicitBellmanAction(
        iterations=args.solver_iterations,
        gradient_step=args.gradient_step,
        max_step=args.max_solver_step,
        curvature_floor=args.curvature_floor,
        gamma=args.gamma,
    )
    generator = torch.Generator().manual_seed(args.seed + 1000)
    history = []

    for iteration in range(args.value_iterations):
        state = random_states(args.batch_size, generator)
        goal_batch = goal.expand(args.batch_size, -1)
        initial = torch.rand(
            args.batch_size, 1, generator=generator,
        ) * 2.0 - 1.0
        target_action = action_layer(
            target_value, dynamics, state, goal_batch, initial,
        )
        with torch.no_grad():
            target, _, _, _ = bellman_return(
                target_value, dynamics, state, goal_batch, target_action,
                args.gamma,
            )
        prediction = value(state, goal_batch)
        loss = F.smooth_l1_loss(prediction, target)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(value.parameters(), 5.0)
        optimizer.step()
        with torch.no_grad():
            for target_parameter, parameter in zip(
                target_value.parameters(), value.parameters(), strict=True,
            ):
                target_parameter.lerp_(parameter, args.target_tau)

        if iteration == 0 or (iteration + 1) % args.eval_every == 0:
            history.append(
                {
                    "iteration": iteration + 1,
                    "value_loss": float(loss.detach()),
                    "target_mean": float(target.mean()),
                    "target_std": float(target.std(unbiased=False)),
                    "value_mean": float(prediction.detach().mean()),
                    "evaluation": evaluate(
                        value, dynamics, goal, action_layer, args,
                        args.eval_seed + iteration,
                    ),
                    "action_landscape": action_landscape_audit(
                        value, dynamics, goal, action_layer, args,
                        args.audit_seed + iteration,
                    ),
                }
            )
    return {
        "factor": tuple(args.factor),
        "history": history,
        "final_evaluation": evaluate(
            value, dynamics, goal, action_layer, args, args.eval_seed + 10000,
        ),
        "final_action_landscape": action_landscape_audit(
            value, dynamics, goal, action_layer, args, args.audit_seed + 10000,
        ),
        "trainable_value_parameters": sum(p.numel() for p in value.parameters()),
        "trainable_actor_parameters": 0,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--factor", type=float, nargs=4, default=list(SOURCE_FACTOR))
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--value-iterations", type=int, default=120)
    parser.add_argument("--value-lr", type=float, default=1e-3)
    parser.add_argument("--target-tau", type=float, default=0.10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--solver-iterations", type=int, default=10)
    parser.add_argument("--gradient-step", type=float, default=0.20)
    parser.add_argument("--max-solver-step", type=float, default=0.25)
    parser.add_argument("--curvature-floor", type=float, default=1e-3)
    parser.add_argument("--eval-every", type=int, default=30)
    parser.add_argument("--test-count", type=int, default=16)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--audit-count", type=int, default=64)
    parser.add_argument("--grid-points", type=int, default=129)
    parser.add_argument("--eval-seed", type=int, default=20260731)
    parser.add_argument("--audit-seed", type=int, default=20260801)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-out", type=str, default="")
    return parser.parse_args()


def main():
    args = parse_args()
    output = {
        "experiment": "OracleImplicitBellmanActionLayer",
        "scope": "source_environment_oracle_dynamics_only",
        "config": vars(args),
        "result": train(args),
    }
    text = json.dumps(output, ensure_ascii=False, indent=2)
    print(text)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")


if __name__ == "__main__":
    main()
