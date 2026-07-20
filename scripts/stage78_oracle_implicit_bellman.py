"""Oracle validation of an actor-free implicit Bellman action layer.

This stage intentionally stops before learned-cognition transfer.  It asks a
single question: with exact target dynamics, can a value network and a
one-dimensional implicit greedy layer learn a useful Acrobot controller?

The action layer has no independent actor parameters.  It solves the local
Bellman stationarity equation using analytic autograd derivatives through the
exact dynamics and audits the solution against a dense grid that is never
used to execute actions or train the value network.
"""

from __future__ import annotations

import argparse
import copy
import json

import torch
import torch.nn.functional as F

from physics_transfer.multifactor_data import _random_states
from scripts import stage43_standard_ppo_baseline as ppo
from scripts.stage15_real_outcome_replay import PRETRAIN_FACTOR
from scripts.stage23_multistep_terminal_value import GOAL


def oracle_bellman_return(value, state, goal, action, factor, gamma):
    """One-step reward plus bootstrapped value under exact dynamics."""
    factor_tensor = torch.as_tensor(
        factor, dtype=state.dtype, device=state.device,
    ).view(1, 4).expand(state.shape[0], -1)
    next_state = ppo.step(
        state, action, factor_tensor[:, 0], factor_tensor[:, 1],
        factor_tensor[:, 2], factor_tensor[:, 3],
    )
    reward, done = ppo.reward_fn(state, next_state, action)
    continuation = value(next_state, goal)
    bellman_return = (
        reward + gamma * continuation * (~done).to(state.dtype)
    )
    return bellman_return, next_state, reward, done


def implicit_action(
    value, state, goal, factor, gamma, iterations, gradient_step,
    max_step, curvature_floor, initial_action=None,
):
    """Solve a local projected Bellman stationarity equation in 1-D action."""
    if initial_action is None:
        action = torch.zeros(state.shape[0], 1, dtype=state.dtype)
    else:
        action = initial_action.detach().clone().clamp(-1.0, 1.0)
    for _ in range(iterations):
        action.requires_grad_(True)
        objective, _, _, _ = oracle_bellman_return(
            value, state, goal, action, factor, gamma,
        )
        gradient = torch.autograd.grad(
            objective.sum(), action, create_graph=True,
        )[0]
        curvature = torch.autograd.grad(
            gradient.sum(), action,
        )[0]
        # Newton is used only at locally concave points.  Elsewhere a bounded
        # projected ascent step moves toward a better stationary region.
        newton_step = -gradient / curvature.clamp_max(-curvature_floor)
        ascent_step = gradient_step * gradient
        delta = torch.where(
            curvature < -curvature_floor, newton_step, ascent_step,
        ).clamp(-max_step, max_step)
        action = (action.detach() + delta.detach()).clamp(-1.0, 1.0)
    return action.detach()


@torch.no_grad()
def grid_best_action(value, state, goal, factor, gamma, grid_points):
    """Audit-only approximate global action; never used by the controller."""
    grid = torch.linspace(-1.0, 1.0, grid_points)
    count = state.shape[0]
    tiled_state = state[:, None, :].expand(-1, grid_points, -1).reshape(-1, 6)
    tiled_goal = goal[:, None, :].expand(-1, grid_points, -1).reshape(-1, 6)
    tiled_action = grid.view(1, -1, 1).expand(count, -1, -1).reshape(-1, 1)
    objective, _, _, _ = oracle_bellman_return(
        value, tiled_state, tiled_goal, tiled_action, factor, gamma,
    )
    objective = objective.view(count, grid_points)
    best_index = objective.argmax(dim=1)
    return grid[best_index].unsqueeze(-1), objective.max(dim=1).values


def stationarity_metrics(value, state, goal, action, factor, gamma,
                         curvature_floor):
    action = action.detach().clone().requires_grad_(True)
    objective, _, _, _ = oracle_bellman_return(
        value, state, goal, action, factor, gamma,
    )
    gradient = torch.autograd.grad(
        objective.sum(), action, create_graph=True,
    )[0]
    curvature = torch.autograd.grad(gradient.sum(), action)[0]
    at_lower = action.detach() <= -0.999
    at_upper = action.detach() >= 0.999
    interior = ~(at_lower | at_upper)
    kkt = (
        (interior & (gradient.detach().abs() <= 1e-3))
        | (at_lower & (gradient.detach() <= 1e-3))
        | (at_upper & (gradient.detach() >= -1e-3))
    )
    return {
        "mean_abs_projected_stationarity": float(
            torch.where(interior, gradient.detach().abs(), torch.zeros_like(gradient))
            .mean()
        ),
        "kkt_fraction": float(kkt.float().mean()),
        "locally_concave_fraction": float(
            (curvature.detach() < -curvature_floor).float().mean()
        ),
        "mean_curvature": float(curvature.detach().mean()),
        "minimum_curvature": float(curvature.detach().min()),
        "maximum_curvature": float(curvature.detach().max()),
        "boundary_fraction": float((at_lower | at_upper).float().mean()),
    }


def action_landscape_audit(value, factor, goal, args, seed):
    generator = torch.Generator().manual_seed(seed)
    state = _random_states(args.audit_count, generator)
    goal_batch = goal.expand(args.audit_count, -1)
    initial = torch.rand(
        args.audit_count, 1, generator=generator,
    ) * 2.0 - 1.0
    implicit = implicit_action(
        value, state, goal_batch, factor, args.gamma,
        args.solver_iterations, args.gradient_step, args.max_solver_step,
        args.curvature_floor, initial,
    )
    with torch.no_grad():
        implicit_objective, _, _, _ = oracle_bellman_return(
            value, state, goal_batch, implicit, factor, args.gamma,
        )
        grid_action, grid_objective = grid_best_action(
            value, state, goal_batch, factor, args.gamma, args.grid_points,
        )
    return {
        "grid_points": args.grid_points,
        "mean_grid_regret": float(
            (grid_objective - implicit_objective).mean()
        ),
        "maximum_grid_regret": float(
            (grid_objective - implicit_objective).max()
        ),
        "mean_action_difference_from_grid": float(
            (grid_action - implicit).abs().mean()
        ),
        "grid_boundary_fraction": float(
            (grid_action.abs() >= 0.999).float().mean()
        ),
        **stationarity_metrics(
            value, state, goal_batch, implicit, factor, args.gamma,
            args.curvature_floor,
        ),
    }


@torch.no_grad()
def evaluate(value, factor, goal, args, seed):
    torch.manual_seed(seed)
    state = ppo.reset_down_states(args.test_count)
    goal_batch = goal.expand(args.test_count, -1)
    previous_action = torch.zeros(args.test_count, 1)
    maximum = torch.full((args.test_count,), -float("inf"))
    success = torch.zeros(args.test_count, dtype=torch.bool)
    total_reward = torch.zeros(args.test_count)
    for _ in range(args.eval_steps):
        # Action optimization needs input gradients even though evaluation does
        # not retain a computation graph for value parameters.
        with torch.enable_grad():
            action = implicit_action(
                value, state, goal_batch, factor, args.gamma,
                args.solver_iterations, args.gradient_step,
                args.max_solver_step, args.curvature_floor,
                previous_action,
            )
        factor_tensor = torch.as_tensor(factor).view(1, 4).expand(
            args.test_count, -1,
        )
        next_state = ppo.step(
            state, action, factor_tensor[:, 0], factor_tensor[:, 1],
            factor_tensor[:, 2], factor_tensor[:, 3],
        )
        reward, done = ppo.reward_fn(state, next_state, action)
        height = ppo.tip_height(next_state)
        maximum = torch.maximum(maximum, height)
        success |= done
        total_reward += reward
        state = next_state
        previous_action = action
    return {
        "success_count": int(success.sum()),
        "success_rate": float(success.float().mean()),
        "mean_max_height": float(maximum.mean()),
        "max_height": float(maximum.max()),
        "mean_undiscounted_reward": float(total_reward.mean()),
    }


def train(args):
    torch.manual_seed(args.seed)
    factor = tuple(args.factor)
    goal = GOAL.view(1, -1)
    value = ppo.ValueCritic(args.hidden_dim)
    torch.nn.init.zeros_(value.net[-1].weight)
    torch.nn.init.zeros_(value.net[-1].bias)
    target_value = copy.deepcopy(value)
    for parameter in target_value.parameters():
        parameter.requires_grad = False
    optimizer = torch.optim.Adam(value.parameters(), lr=args.value_lr)
    generator = torch.Generator().manual_seed(args.seed + 1000)
    history = []
    for iteration in range(args.value_iterations):
        state = _random_states(args.batch_size, generator)
        goal_batch = goal.expand(args.batch_size, -1)
        initial = torch.rand(
            args.batch_size, 1, generator=generator,
        ) * 2.0 - 1.0
        target_action = implicit_action(
            target_value, state, goal_batch, factor, args.gamma,
            args.solver_iterations, args.gradient_step,
            args.max_solver_step, args.curvature_floor, initial,
        )
        with torch.no_grad():
            target, _, _, _ = oracle_bellman_return(
                target_value, state, goal_batch, target_action,
                factor, args.gamma,
            )
        prediction = value(state, goal_batch)
        loss = F.smooth_l1_loss(prediction, target)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(value.parameters(), 5.0)
        optimizer.step()
        with torch.no_grad():
            for target_parameter, parameter in zip(
                target_value.parameters(), value.parameters(),
            ):
                target_parameter.lerp_(parameter, args.target_tau)

        if iteration == 0 or (iteration + 1) % args.eval_every == 0:
            history.append({
                "iteration": iteration + 1,
                "value_loss": float(loss.detach()),
                "target_mean": float(target.mean()),
                "target_std": float(target.std(unbiased=False)),
                "value_mean": float(prediction.detach().mean()),
                "evaluation": evaluate(
                    value, factor, goal, args,
                    args.eval_seed + iteration,
                ),
                "action_landscape": action_landscape_audit(
                    value, factor, goal, args,
                    args.audit_seed + iteration,
                ),
            })
    return {
        "factor": factor,
        "history": history,
        "final_evaluation": evaluate(
            value, factor, goal, args, args.eval_seed + 10000,
        ),
        "final_action_landscape": action_landscape_audit(
            value, factor, goal, args, args.audit_seed + 10000,
        ),
        "trainable_value_parameters": sum(
            parameter.numel() for parameter in value.parameters()
        ),
        "trainable_actor_parameters": 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--factor", type=float, nargs=4,
                        default=list(PRETRAIN_FACTOR[0]))
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
    args = parser.parse_args()
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
