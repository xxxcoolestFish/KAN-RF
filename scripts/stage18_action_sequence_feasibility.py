"""Check whether the fixed evaluation states are reachable at all.

This is an oracle feasibility diagnostic, not a deployable controller.  It
optimizes one bounded action sequence per initial state directly through the
known simulator.  If a state remains below the success threshold after
multiple restarts, a 20/20 claim under the current horizon/action limits is
not justified without changing the task definition or control budget.
"""

from __future__ import annotations

import argparse
import json

import torch

from physics_transfer.multifactor_data import _random_states
from physics_transfer.variants import step
from scripts.stage13_online_task_loss_adaptation import tip_height
from scripts.stage15_real_outcome_replay import PRETRAIN_FACTOR


def geometric_height(state):
    # Equivalent to tip_height, but avoids atan2 branch points during
    # differentiable action-sequence optimization.
    c1, s1, c2, s2 = state[:, 0], state[:, 1], state[:, 2], state[:, 3]
    return -c1 - (c1 * c2 - s1 * s2)


def optimize_batch(initial_states, factor, horizon, steps, restarts, seed, lr):
    count = initial_states.shape[0]
    factor_tensor = torch.tensor(factor, dtype=initial_states.dtype).view(1, 4).expand(count, -1)
    best_heights = torch.full((count,), -float("inf"))
    best_restart = torch.zeros(count, dtype=torch.long)

    for restart in range(restarts):
        torch.manual_seed(seed + restart)
        logits = torch.randn(count, horizon, 1) * 0.25
        logits.requires_grad_()
        optimizer = torch.optim.Adam([logits], lr=lr)
        for _ in range(steps):
            state = initial_states
            heights = []
            actions = []
            for index in range(horizon):
                action = torch.tanh(logits[:, index])
                state = step(
                    state, action,
                    factor_tensor[:, 0], factor_tensor[:, 1],
                    factor_tensor[:, 2], factor_tensor[:, 3],
                )
                actions.append(action)
                heights.append(geometric_height(state))
            height_stack = torch.stack(heights, dim=1)
            action_stack = torch.stack(actions, dim=1)
            # Smooth approximation to max height, with small control and
            # action-change penalties to avoid numerical impulses.
            soft_max = 0.08 * torch.logsumexp(height_stack / 0.08, dim=1)
            smooth = (action_stack[:, 1:] - action_stack[:, :-1]).square().mean(dim=(1, 2))
            loss = (-soft_max + 1e-4 * action_stack.square().mean(dim=(1, 2)) + 1e-4 * smooth).mean()
            optimizer.zero_grad(); loss.backward(); optimizer.step()

        with torch.no_grad():
            state = initial_states
            heights = []
            for index in range(horizon):
                action = torch.tanh(logits[:, index])
                state = step(
                    state, action,
                    factor_tensor[:, 0], factor_tensor[:, 1],
                    factor_tensor[:, 2], factor_tensor[:, 3],
                )
                heights.append(geometric_height(state))
            maximum = torch.stack(heights, dim=1).max(dim=1).values
            improved = maximum > best_heights
            best_heights = torch.where(improved, maximum, best_heights)
            best_restart = torch.where(improved, torch.full_like(best_restart, restart), best_restart)

    return best_heights, best_restart


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-count", type=int, default=20)
    parser.add_argument("--test-seed", type=int, default=20260718)
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--restarts", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generator = torch.Generator().manual_seed(args.test_seed)
    initial_states = _random_states(args.test_count, generator=generator)
    factor = PRETRAIN_FACTOR[0]
    maximum, restart = optimize_batch(
        initial_states, factor, args.horizon, args.steps,
        args.restarts, args.seed, args.lr,
    )
    success = maximum >= 1.0
    print(json.dumps({
        "environment": factor,
        "test_count": args.test_count,
        "test_seed": args.test_seed,
        "horizon": args.horizon,
        "optimization_steps": args.steps,
        "restarts": args.restarts,
        "feasible_count_found": int(success.sum().item()),
        "feasible_rate_found": float(success.float().mean().item()),
        "optimized_max_height": maximum.tolist(),
        "best_restart": restart.tolist(),
        "note": "Feasibility found by direct sequence optimization is a lower bound on reachability, not a proof of impossibility.",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

