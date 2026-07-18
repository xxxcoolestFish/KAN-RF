"""Diagnose whether the decision policy can represent and learn a good policy.

This experiment keeps one fixed set of initial states and separates three
questions that are mixed together by online self-replay:

1. how good the one-time mapped initialization is;
2. how good the direct true-dynamics MPC teacher is;
3. whether a fully trainable decision network can imitate that teacher.

The cognitive model is used only to obtain the one-time fixed operator.  The
oracle teacher uses the known simulator only as a diagnostic upper bound.
"""

from __future__ import annotations

import argparse
import json

import torch
from torch import nn
import torch.nn.functional as F

from physics_transfer.multifactor_data import _random_states
from physics_transfer.variants import step
from scripts.stage13_online_task_loss_adaptation import (
    RuntimeTaskDecision,
    initialize_decision,
    initial_operator,
    tip_height,
)
from scripts.stage15_real_outcome_replay import PRETRAIN_FACTOR, pretrain
from scripts.stage7_mpc_decision_adaptation import true_mpc_teacher


class FullyTrainableDecision(RuntimeTaskDecision):
    """Runtime decision whose task and base physics parameters are trainable."""

    def __init__(self, initialized):
        super().__init__(initialized)
        # RuntimeTaskDecision stores this as a buffer because the normal online
        # experiment freezes it.  The capacity diagnostic deliberately removes
        # that restriction.
        initial_basis = self.base_basis.detach().clone()
        del self._buffers["base_basis"]
        self.base_basis = nn.Parameter(initial_basis)
        for parameter in self.parameters():
            parameter.requires_grad = True


def make_fixed_states(count: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return _random_states(count, generator=generator)


def rollout(decision, q_const, initial_states, factor, teacher=False, steps=64):
    state = initial_states.detach().clone()
    factor_tensor = torch.tensor(factor, dtype=state.dtype).view(1, 4).expand(state.shape[0], -1)
    heights = []
    for _ in range(steps):
        if teacher:
            action = true_mpc_teacher(state, factor)
        else:
            action = decision(state, q_const.expand(state.shape[0], -1))["action"]
        state = step(
            state, action,
            factor_tensor[:, 0], factor_tensor[:, 1],
            factor_tensor[:, 2], factor_tensor[:, 3],
        )
        heights.append(tip_height(state))
    max_height = torch.stack(heights, dim=1).max(dim=1).values
    success = max_height >= 1.0
    return {
        "success_count": int(success.sum().item()),
        "success_rate": float(success.float().mean().item()),
        "mean_max_height": float(max_height.mean().item()),
        "max_height": max_height.tolist(),
    }


def train_oracle_policy(decision, q_const, factor, steps, batch_size, seed):
    """Fit all decision parameters to the true one-step 5-step MPC teacher."""
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(decision.parameters(), lr=2e-3)
    losses = []
    for _ in range(steps):
        states = _random_states(batch_size)
        with torch.no_grad():
            target = true_mpc_teacher(states, factor)
        prediction = decision(states, q_const.expand(batch_size, -1))["action"]
        loss = F.mse_loss(prediction, target)
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(decision.parameters(), 5.0)
        optimizer.step()
        losses.append(loss.item())
    return {
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "mean_last_50_loss": sum(losses[-50:]) / min(50, len(losses)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cognitive-steps", type=int, default=100)
    parser.add_argument("--base-steps", type=int, default=100)
    parser.add_argument("--mapper-steps", type=int, default=100)
    parser.add_argument("--oracle-steps", type=int, default=500)
    parser.add_argument("--oracle-batch-size", type=int, default=128)
    parser.add_argument("--test-count", type=int, default=20)
    parser.add_argument("--test-seed", type=int, default=20260718)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cognitive = pretrain(args.cognitive_steps, 16, args.seed)
    initialized = initialize_decision(cognitive, 16, args.base_steps, args.mapper_steps, args.seed)
    q_const = initial_operator(cognitive, 16, args.seed).detach()
    test_states = make_fixed_states(args.test_count, args.test_seed)
    factor = PRETRAIN_FACTOR[0]

    mapped = RuntimeTaskDecision(initialized)
    mapped_result = rollout(mapped, q_const, test_states, factor)
    teacher_result = rollout(None, None, test_states, factor, teacher=True)

    oracle = FullyTrainableDecision(initialized)
    oracle_fit = train_oracle_policy(
        oracle, q_const, factor, args.oracle_steps, args.oracle_batch_size, args.seed + 9000
    )
    oracle_result = rollout(oracle, q_const, test_states, factor)

    print(json.dumps({
        "environment": factor,
        "test_count": args.test_count,
        "test_seed": args.test_seed,
        "same_fixed_initial_states": True,
        "mapped_initialization": mapped_result,
        "true_mpc_teacher_upper_bound": teacher_result,
        "fully_trainable_oracle_imitation": {
            "fit": oracle_fit,
            "evaluation": oracle_result,
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

