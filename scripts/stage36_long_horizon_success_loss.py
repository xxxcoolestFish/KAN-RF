"""Long-horizon policy training with a differentiable terminal success loss.

The source dynamics are used directly as a diagnostic upper bound.  The
experiment sweeps the maximum training horizon and compares the full
cognitive-parameter policy with a plain state-goal MLP.
"""

from __future__ import annotations

import argparse
import copy
import json

import torch
import torch.nn.functional as F

from physics_transfer.full_parameter_transport import FullParameterTransport
from physics_transfer.multifactor_data import _random_states
from physics_transfer.sensitivity_policy import SensitivityMandatoryPolicy
from physics_transfer.variants import step
from scripts.stage15_real_outcome_replay import PRETRAIN_FACTOR
from scripts.stage23_multistep_terminal_value import (
    GOAL,
    SimpleCognitiveKAN,
    smooth_tip_height,
)
from scripts.stage27_parameter_transport import pretrain_cognitive
from scripts.stage34_real_transition_goal_loss import goal_potential
from scripts.stage35_plain_policy_ablation import PlainPolicy


def smooth_max(values: torch.Tensor, temperature: float = 0.08) -> torch.Tensor:
    """Stable soft maximum, approximately bounded by the true maximum."""
    return temperature * torch.logsumexp(
        values / temperature, dim=1,
    ) - temperature * torch.log(
        torch.as_tensor(values.shape[1], dtype=values.dtype, device=values.device),
    )


def horizon_levels(max_horizon: int) -> list[int]:
    return [h for h in (1, 2, 4, 8, 16) if h <= max_horizon]


def exact_transition(current: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    factor = torch.tensor(
        PRETRAIN_FACTOR[0], dtype=current.dtype, device=current.device,
    ).view(1, 4).expand(current.shape[0], -1)
    return step(current, action, factor[:, 0], factor[:, 1], factor[:, 2], factor[:, 3])


def terminal_success_loss(
    current: torch.Tensor,
    heights: list[torch.Tensor],
    actions: list[torch.Tensor],
    goal: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    height_stack = torch.stack(heights, dim=1)
    soft_height = smooth_max(height_stack)
    success_probability = torch.sigmoid(10.0 * (soft_height - 1.0))
    terminal_potential = goal_potential(current, goal)
    per_step_potential = torch.stack(
        [goal_potential(state, goal) for state in []], dim=1,
    ) if False else None
    action_stack = torch.stack(actions, dim=1)
    smooth = (
        (action_stack[:, 1:] - action_stack[:, :-1]).square().mean()
        if action_stack.shape[1] > 1 else torch.zeros((), dtype=current.dtype)
    )
    loss = (
        -success_probability.mean()
        + 0.50 * terminal_potential.mean()
        + 0.02 * smooth
    )
    return loss, {
        "soft_height": float(soft_height.detach().mean().item()),
        "success_probability": float(success_probability.detach().mean().item()),
        "terminal_potential": float(terminal_potential.detach().mean().item()),
    }


def fit_policy(
    policy,
    cognitive: SimpleCognitiveKAN,
    goal: torch.Tensor,
    max_horizon: int,
    steps: int,
    batch_size: int,
    seed: int,
) -> dict:
    if hasattr(policy, "cognitive"):
        for parameter in policy.cognitive.parameters():
            parameter.requires_grad = False
        policy.transport.freeze()
    for parameter in cognitive.parameters():
        parameter.requires_grad = False
    parameters = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=2e-3)
    torch.manual_seed(seed + 3600 + max_horizon)
    levels = horizon_levels(max_horizon)
    losses, success_probs, soft_heights = [], [], []
    for index in range(steps):
        horizon = levels[min(
            len(levels) - 1,
            int(index / max(1, steps - 1) * len(levels)),
        )]
        current = _random_states(batch_size)
        heights, actions = [], []
        for _ in range(horizon):
            action = policy(current, goal)
            next_state = exact_transition(current, action)
            heights.append(smooth_tip_height(next_state))
            actions.append(action)
            current = next_state
        loss, stats = terminal_success_loss(current, heights, actions, goal)
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 5.0); optimizer.step()
        losses.append(float(loss.item()))
        success_probs.append(stats["success_probability"])
        soft_heights.append(stats["soft_height"])
    return {
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "mean_last_20_loss": sum(losses[-20:]) / min(20, len(losses)),
        "mean_last_20_soft_success": sum(success_probs[-20:]) / min(20, len(success_probs)),
        "mean_last_20_soft_height": sum(soft_heights[-20:]) / min(20, len(soft_heights)),
        "max_horizon": max_horizon,
    }


@torch.no_grad()
def evaluate(policy, states: torch.Tensor, goal: torch.Tensor, steps: int) -> dict:
    current = states.clone()
    factor = torch.tensor(
        PRETRAIN_FACTOR[0], dtype=current.dtype, device=current.device,
    ).view(1, 4).expand(states.shape[0], -1)
    maxima = torch.full(
        (states.shape[0],), -float("inf"), dtype=states.dtype,
    )
    actions = []
    for _ in range(steps):
        action = policy(current, goal)
        actions.append(action)
        current = step(current, action, factor[:, 0], factor[:, 1], factor[:, 2], factor[:, 3])
        maxima = torch.maximum(maxima, smooth_tip_height(current))
    success = maxima >= 1.0
    return {
        "success_count": int(success.sum().item()),
        "success_rate": float(success.float().mean().item()),
        "mean_max_height": float(maxima.mean().item()),
        "mean_final_goal_potential": float(goal_potential(current, goal).mean().item()),
        "mean_abs_action": float(torch.stack(actions, dim=1).abs().mean().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cognitive-steps", type=int, default=200)
    parser.add_argument("--policy-steps", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-horizons", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--rollout-steps", type=int, default=48)
    parser.add_argument("--test-count", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--test-seed", type=int, default=20260719)
    parser.add_argument("--json-out", type=str, default="")
    args = parser.parse_args()
    goal = GOAL.view(1, -1)
    records = []
    for seed in args.seeds:
        torch.manual_seed(seed)
        cognitive = SimpleCognitiveKAN()
        cognitive_fit = pretrain_cognitive(cognitive, args.cognitive_steps,
                                            args.batch_size, seed)
        template = SensitivityMandatoryPolicy(
            cognitive, FullParameterTransport(cognitive),
        )
        states = _random_states(
            args.test_count,
            generator=torch.Generator().manual_seed(args.test_seed),
        )
        seed_record = {"seed": seed, "cognitive_fit": cognitive_fit, "horizons": {}}
        for max_horizon in args.max_horizons:
            for architecture in ("full", "plain"):
                policy = (copy.deepcopy(template) if architecture == "full"
                          else PlainPolicy())
                fit = fit_policy(
                    policy, cognitive, goal, max_horizon, args.policy_steps,
                    args.batch_size, seed,
                )
                seed_record["horizons"][f"{architecture}_H{max_horizon}"] = {
                    "fit": fit,
                    "source": evaluate(policy, states, goal, args.rollout_steps),
                }
        records.append(seed_record)
    output = {
        "architecture": "LongHorizonTerminalSuccessLoss",
        "experiment": "exact_source_dynamics_horizon_sweep",
        "config": vars(args),
        "seeds": records,
    }
    text = json.dumps(output, ensure_ascii=False, indent=2)
    print(text)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")


if __name__ == "__main__":
    main()
