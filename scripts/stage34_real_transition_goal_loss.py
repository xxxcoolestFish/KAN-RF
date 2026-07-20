"""Compare model-rollout and real-transition policy training.

This experiment isolates two causes of the current decision failure:

1. training the policy through the learned cognitive model versus the exact
   source-environment transition;
2. the legacy height-only task cost versus an explicit goal-state potential.

The cognitive network is still pre-trained on transition prediction and its
full parameter vector is coupled into the decision policy.  Only the policy
training transition and decision loss are changed.
"""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

from physics_transfer.full_parameter_transport import FullParameterTransport
from physics_transfer.multifactor_data import _random_states
from physics_transfer.sensitivity_policy import SensitivityMandatoryPolicy
from physics_transfer.variants import step
from scripts.stage15_real_outcome_replay import PRETRAIN_FACTOR
from scripts.stage2_lowrank_loss_ablation import HELDOUT
from scripts.stage23_multistep_terminal_value import (
    GOAL,
    SimpleCognitiveKAN,
    smooth_tip_height,
    task_cost,
)
from scripts.stage27_parameter_transport import pretrain_cognitive, prediction_error


def _expand_goal(goal: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
    if goal.shape[0] == 1 and state.shape[0] != 1:
        return goal.expand(state.shape[0], -1)
    return goal


def goal_potential(state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
    """Distance to the explicit target state in the normalized state space."""
    goal = _expand_goal(goal, state)
    angle_error = (state[:, :4] - goal[:, :4]).square().mean(dim=-1)
    velocity_error = (state[:, 4:] - goal[:, 4:]).square().mean(dim=-1)
    return angle_error + 0.25 * velocity_error


def goal_directed_cost(
    current: torch.Tensor,
    next_state: torch.Tensor,
    action: torch.Tensor,
    goal: torch.Tensor,
) -> torch.Tensor:
    """Task cost with an explicit target-state direction.

    The target-state potential is zero only at ``goal``.  The height term
    preserves the original swing-up success objective, while velocity and
    effort discourage unstable or unnecessarily large actions.
    """
    del current  # retained in the signature for a common rollout interface
    potential = goal_potential(next_state, goal)
    height_shortfall = F.relu(1.0 - smooth_tip_height(next_state)).square()
    velocity = 0.05 * next_state[:, 4:].square().sum(dim=-1)
    effort = 0.01 * action[:, 0].square()
    return potential + 0.5 * height_shortfall + velocity + effort


def transition(
    current: torch.Tensor,
    action: torch.Tensor,
    cognitive: SimpleCognitiveKAN,
    mode: str,
    factor: tuple[float, float, float, float],
) -> torch.Tensor:
    if mode == "model":
        return cognitive(current, action)
    factor_tensor = torch.tensor(
        factor, dtype=current.dtype, device=current.device,
    ).view(1, 4).expand(current.shape[0], -1)
    return step(
        current,
        action,
        factor_tensor[:, 0],
        factor_tensor[:, 1],
        factor_tensor[:, 2],
        factor_tensor[:, 3],
    )


def fit_policy(
    policy: SensitivityMandatoryPolicy,
    transition_mode: str,
    loss_mode: str,
    goal: torch.Tensor,
    steps: int,
    batch_size: int,
    max_horizon: int,
    seed: int,
    factor: tuple[float, float, float, float],
) -> dict:
    for parameter in policy.cognitive.parameters():
        parameter.requires_grad = False
    policy.transport.freeze()
    parameters = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=2e-3)
    torch.manual_seed(seed + 3400)
    losses, horizons = [], []
    levels = [h for h in (1, 2, 4, 8) if h <= max_horizon]
    for index in range(steps):
        horizon = levels[min(
            len(levels) - 1,
            int(index / max(1, steps - 1) * len(levels)),
        )]
        current = _random_states(batch_size)
        costs, actions = [], []
        for _ in range(horizon):
            action = policy(current, goal)
            next_state = transition(
                current, action, policy.cognitive, transition_mode, factor,
            )
            if loss_mode == "legacy":
                cost = task_cost(next_state, action)
            else:
                cost = goal_directed_cost(current, next_state, action, goal)
            costs.append(cost)
            actions.append(action)
            current = next_state
        discounted = torch.stack(
            [(0.95 ** t) * value for t, value in enumerate(costs)], dim=1,
        ).sum(dim=1)
        action_stack = torch.stack(actions, dim=1)
        smooth = (
            (action_stack[:, 1:] - action_stack[:, :-1]).square().mean()
            if horizon > 1 else torch.zeros((), dtype=current.dtype)
        )
        if loss_mode == "legacy":
            terminal = torch.zeros_like(discounted)
        else:
            terminal = 0.5 * goal_potential(current, goal)
        loss = (discounted + (0.95 ** horizon) * terminal).mean() + 0.02 * smooth
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 5.0)
        optimizer.step()
        losses.append(float(loss.item()))
        horizons.append(horizon)
    return {
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "mean_last_20_loss": sum(losses[-20:]) / min(20, len(losses)),
        "final_horizon": horizons[-1],
    }


@torch.no_grad()
def evaluate(
    policy: SensitivityMandatoryPolicy,
    states: torch.Tensor,
    factor: tuple[float, float, float, float],
    goal: torch.Tensor,
    rollout_steps: int,
) -> dict:
    current = states.detach().clone()
    factor_tensor = torch.tensor(
        factor, dtype=current.dtype, device=current.device,
    ).view(1, 4).expand(current.shape[0], -1)
    maxima = torch.full(
        (current.shape[0],), -float("inf"), dtype=current.dtype,
    )
    actions = []
    for _ in range(rollout_steps):
        action = policy(current, goal)
        actions.append(action)
        current = step(
            current,
            action,
            factor_tensor[:, 0],
            factor_tensor[:, 1],
            factor_tensor[:, 2],
            factor_tensor[:, 3],
        )
        maxima = torch.maximum(maxima, smooth_tip_height(current))
    success = maxima >= 1.0
    return {
        "success_count": int(success.sum().item()),
        "success_rate": float(success.float().mean().item()),
        "mean_max_height": float(maxima.mean().item()),
        "mean_final_goal_potential": float(goal_potential(current, goal).mean().item()),
        "mean_abs_action": float(torch.stack(actions, dim=1).abs().mean().item()),
    }


@dataclass(frozen=True)
class Variant:
    transition: str
    loss: str

    @property
    def name(self) -> str:
        return f"{self.transition}_transition__{self.loss}_loss"


def run_seed(args: argparse.Namespace, seed: int) -> dict:
    torch.manual_seed(seed)
    cognitive = SimpleCognitiveKAN()
    cognitive_fit = pretrain_cognitive(
        cognitive, args.cognitive_steps, args.batch_size, seed,
    )
    template = SensitivityMandatoryPolicy(
        cognitive, FullParameterTransport(cognitive),
    )
    goal = GOAL.view(1, -1)
    variants = [
        Variant("model", "legacy"),
        Variant("model", "goal"),
        Variant("real", "legacy"),
        Variant("real", "goal"),
    ]
    generator = torch.Generator().manual_seed(args.test_seed)
    states = _random_states(args.test_count, generator=generator)
    factors = {
        "source": PRETRAIN_FACTOR[0],
        "heldout_1": HELDOUT[0],
        "heldout_2": HELDOUT[1],
    }
    output = {
        "seed": seed,
        "cognitive_fit": cognitive_fit,
        "cognitive_prediction": {
            label: prediction_error(cognitive, factor)
            for label, factor in factors.items()
        },
        "variants": {},
    }
    for variant in variants:
        policy = copy.deepcopy(template)
        fit = fit_policy(
            policy,
            variant.transition,
            variant.loss,
            goal,
            args.policy_steps,
            args.batch_size,
            args.max_horizon,
            seed,
            PRETRAIN_FACTOR[0],
        )
        output["variants"][variant.name] = {
            "fit": fit,
            "evaluation": {
                label: evaluate(policy, states, factor, goal, args.rollout_steps)
                for label, factor in factors.items()
            },
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cognitive-steps", type=int, default=200)
    parser.add_argument("--policy-steps", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-horizon", type=int, default=4)
    parser.add_argument("--rollout-steps", type=int, default=48)
    parser.add_argument("--test-count", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--test-seed", type=int, default=20260719)
    parser.add_argument("--json-out", type=str, default="")
    args = parser.parse_args()
    outputs = [run_seed(args, seed) for seed in args.seeds]
    result = {
        "architecture": "FullParameterCoupling",
        "experiment": "real_transition_vs_cognitive_rollout_and_goal_directed_loss",
        "source_factor": PRETRAIN_FACTOR[0],
        "config": vars(args),
        "seeds": outputs,
    }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")


if __name__ == "__main__":
    main()
