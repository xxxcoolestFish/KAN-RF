"""Compare a plain state-goal policy with the full cognitive coupling policy."""

from __future__ import annotations

import argparse
import copy
import json

import torch
from torch import nn

from physics_transfer.full_parameter_transport import FullParameterTransport
from physics_transfer.multifactor_data import _random_states
from physics_transfer.sensitivity_policy import SensitivityMandatoryPolicy
from physics_transfer.variants import step
from scripts.stage15_real_outcome_replay import PRETRAIN_FACTOR
from scripts.stage23_multistep_terminal_value import (
    GOAL,
    SimpleCognitiveKAN,
    smooth_tip_height,
    task_cost,
)
from scripts.stage27_parameter_transport import pretrain_cognitive
from scripts.stage34_real_transition_goal_loss import (
    goal_directed_cost,
    goal_potential,
)


class PlainPolicy(nn.Module):
    def __init__(self, hidden_dim: int = 32, action_limit: float = 0.9):
        super().__init__()
        self.action_limit = action_limit
        self.network = nn.Sequential(
            nn.Linear(12, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        if goal.shape[0] == 1 and state.shape[0] != 1:
            goal = goal.expand(state.shape[0], -1)
        return self.action_limit * torch.tanh(
            self.network(torch.cat([state, goal], dim=-1))
        )


def fit(policy, cognitive, goal, loss_mode, steps, batch_size, max_horizon, seed):
    for parameter in cognitive.parameters():
        parameter.requires_grad = False
    if hasattr(policy, "transport"):
        policy.transport.freeze()
    parameters = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=2e-3)
    torch.manual_seed(seed + 3500)
    levels = [h for h in (1, 2, 4, 8) if h <= max_horizon]
    losses = []
    factor = torch.tensor(PRETRAIN_FACTOR[0])
    for index in range(steps):
        horizon = levels[min(len(levels) - 1,
                             int(index / max(1, steps - 1) * len(levels)))]
        current = _random_states(batch_size)
        costs, actions = [], []
        for _ in range(horizon):
            action = policy(current, goal)
            f = factor.to(current).view(1, 4).expand(batch_size, -1)
            next_state = step(current, action, f[:, 0], f[:, 1], f[:, 2], f[:, 3])
            cost = (task_cost(next_state, action) if loss_mode == "legacy"
                    else goal_directed_cost(current, next_state, action, goal))
            costs.append(cost); actions.append(action); current = next_state
        discounted = torch.stack(
            [(0.95 ** t) * value for t, value in enumerate(costs)], dim=1,
        ).sum(dim=1)
        action_stack = torch.stack(actions, dim=1)
        smooth = ((action_stack[:, 1:] - action_stack[:, :-1]).square().mean()
                  if horizon > 1 else torch.zeros((), dtype=current.dtype))
        terminal = (torch.zeros_like(discounted) if loss_mode == "legacy"
                    else 0.5 * goal_potential(current, goal))
        loss = (discounted + (0.95 ** horizon) * terminal).mean() + 0.02 * smooth
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 5.0); optimizer.step()
        losses.append(float(loss.item()))
    return {"first_loss": losses[0], "last_loss": losses[-1],
            "mean_last_20_loss": sum(losses[-20:]) / min(20, len(losses))}


@torch.no_grad()
def evaluate(policy, states, goal, steps):
    current = states.clone()
    f = torch.tensor(PRETRAIN_FACTOR[0]).view(1, 4).expand(states.shape[0], -1)
    maxima = torch.full((states.shape[0],), -float("inf")); actions = []
    for _ in range(steps):
        action = policy(current, goal); actions.append(action)
        current = step(current, action, f[:, 0], f[:, 1], f[:, 2], f[:, 3])
        maxima = torch.maximum(maxima, smooth_tip_height(current))
    success = maxima >= 1.0
    return {
        "success_rate": float(success.float().mean().item()),
        "success_count": int(success.sum().item()),
        "mean_max_height": float(maxima.mean().item()),
        "mean_final_goal_potential": float(goal_potential(current, goal).mean().item()),
        "mean_abs_action": float(torch.stack(actions, 1).abs().mean().item()),
    }


def main():
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
    records = []
    for seed in args.seeds:
        torch.manual_seed(seed)
        cognitive = SimpleCognitiveKAN()
        cognitive_fit = pretrain_cognitive(cognitive, args.cognitive_steps,
                                            args.batch_size, seed)
        template = SensitivityMandatoryPolicy(
            cognitive, FullParameterTransport(cognitive),
        )
        goal = GOAL.view(1, -1)
        states = _random_states(
            args.test_count,
            generator=torch.Generator().manual_seed(args.test_seed),
        )
        item = {"seed": seed, "cognitive_fit": cognitive_fit, "variants": {}}
        for loss_mode in ("legacy", "goal"):
            for architecture in ("full", "plain"):
                policy = (copy.deepcopy(template) if architecture == "full"
                          else PlainPolicy())
                fit_result = fit(
                    policy, cognitive, goal, loss_mode, args.policy_steps,
                    args.batch_size, args.max_horizon, seed,
                )
                item["variants"][f"{architecture}_real__{loss_mode}_loss"] = {
                    "fit": fit_result,
                    "source": evaluate(policy, states, goal, args.rollout_steps),
                }
        records.append(item)
    output = {"architecture": "FullCoupling_vs_PlainPolicy",
              "experiment": "real_transition_parameter_interference_ablation",
              "config": vars(args), "seeds": records}
    text = json.dumps(output, ensure_ascii=False, indent=2)
    print(text)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")


if __name__ == "__main__":
    main()
