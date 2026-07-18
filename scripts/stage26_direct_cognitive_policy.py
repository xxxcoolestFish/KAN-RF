"""Direct policy decoder built on a separately trained cognitive ProtoKAN.

The cognitive model is trained only with transition prediction loss.  The
decision model contains it as a frozen submodule and reads its physical
response to fixed action probes.  The policy head then outputs an action in one
forward pass; no action-gradient optimization is used at decision time.
"""

from __future__ import annotations

import argparse
import json
import math

import torch
import torch.nn.functional as F
from torch import nn

from kanrf._protokan import ProtoKAN
from physics_transfer.multifactor_data import _random_states
from physics_transfer.transition_data import sample_transition_batch
from physics_transfer.variants import step
from scripts.stage15_real_outcome_replay import PRETRAIN_FACTOR
from scripts.stage2_lowrank_loss_ablation import FACTORS, HELDOUT
from scripts.stage23_multistep_terminal_value import (
    GOAL,
    SimpleCognitiveKAN,
    pretrain_cognitive_multistep,
    smooth_tip_height,
    task_cost,
)


class CognitiveProbeEncoder(nn.Module):
    def __init__(self, cognitive, probe_actions=(-1.0, 0.0, 1.0)):
        super().__init__()
        self.cognitive = cognitive
        self.register_buffer("probe_actions", torch.tensor(probe_actions).view(-1, 1))

    def forward(self, state, goal):
        batch_size = state.shape[0]
        probe_count = self.probe_actions.shape[0]
        probe_states = state.unsqueeze(1).expand(-1, probe_count, -1).reshape(-1, 6)
        probe_actions = self.probe_actions.view(1, probe_count, 1).expand(
            batch_size, -1, -1
        ).reshape(-1, 1)
        predicted = self.cognitive(probe_states, probe_actions).reshape(
            batch_size, probe_count, 6
        )
        response_delta = predicted - predicted[:, 1:2]
        return torch.cat([
            state, goal.expand(batch_size, -1), predicted.flatten(1),
            response_delta.flatten(1),
        ], dim=-1)


class DirectCognitivePolicy(nn.Module):
    def __init__(self, cognitive, hidden_dim=32, n_prototypes=8):
        super().__init__()
        self.encoder = CognitiveProbeEncoder(cognitive)
        self.policy = ProtoKAN([48, hidden_dim, 1], n_prototypes=n_prototypes)

    @property
    def cognitive(self):
        return self.encoder.cognitive

    def forward(self, state, goal):
        return torch.tanh(self.policy(self.encoder(state, goal)))


def stats(losses):
    return {"first_loss": float(losses[0]), "last_loss": float(losses[-1]),
            "mean_last_20_loss": float(sum(losses[-20:]) / min(20, len(losses)))}


def horizon_at(index, total, maximum):
    levels = [h for h in (1, 2, 4, 8) if h <= maximum]
    return levels[min(len(levels) - 1, int(index / max(1, total - 1) * len(levels)))]


def train_policy(model, steps, batch_size, max_horizon, seed, goal, gamma):
    torch.manual_seed(seed + 2600)
    for parameter in model.cognitive.parameters():
        parameter.requires_grad = False
    model.cognitive.eval()
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=1e-3)
    losses, horizons = [], []
    for index in range(steps):
        horizon = horizon_at(index, steps, max_horizon)
        current = _random_states(batch_size)
        costs, actions = [], []
        for t in range(horizon):
            action = model(current, goal)
            next_state = model.cognitive(current, action)
            costs.append(task_cost(next_state, action))
            actions.append(action)
            current = next_state
        discounted = torch.stack(
            [gamma ** t * cost for t, cost in enumerate(costs)], dim=1
        ).sum(dim=1)
        terminal = 0.5 * task_cost(current, torch.zeros_like(actions[-1]))
        action_stack = torch.stack(actions, dim=1)
        smooth = (
            (action_stack[:, 1:] - action_stack[:, :-1]).square().mean()
            if horizon > 1 else torch.zeros((), dtype=current.dtype)
        )
        loss = (discounted + gamma ** horizon * terminal).mean() + 0.02 * smooth
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.policy.parameters(), 5.0)
        optimizer.step()
        losses.append(loss.item()); horizons.append(horizon)
    return {**stats(losses), "final_horizon": horizons[-1]}


@torch.no_grad()
def prediction_error(cognitive, factor, batch_size=512):
    batch = sample_transition_batch(batch_size, (factor,))
    prediction = cognitive(batch["state"], batch["action"])
    return {"one_step_smooth_l1": float(F.smooth_l1_loss(
        prediction, batch["next_state"]).item()),
        "one_step_mse": float(F.mse_loss(
            prediction, batch["next_state"]).item())}


@torch.no_grad()
def evaluate(model, states, factor, goal, rollout_steps):
    current = states.detach().clone()
    factor_tensor = torch.tensor(factor, dtype=current.dtype).view(1, 4).expand(current.shape[0], -1)
    maxima = torch.full((current.shape[0],), -float("inf")); actions = []
    for _ in range(rollout_steps):
        action = model(current, goal); actions.append(action)
        current = step(current, action, factor_tensor[:, 0], factor_tensor[:, 1],
                       factor_tensor[:, 2], factor_tensor[:, 3])
        maxima = torch.maximum(maxima, smooth_tip_height(current))
    success = maxima >= 1.0
    return {"success_count": int(success.sum().item()),
            "success_rate": float(success.float().mean().item()),
            "mean_max_height": float(maxima.mean().item()),
            "mean_abs_action": float(torch.stack(actions, 1).abs().mean().item())}


def factor_distance(source, factor):
    scale = (5.0, 0.05, 0.2, 0.2)
    return float(math.sqrt(sum(((a - b) / s) ** 2 for a, b, s in zip(source, factor, scale))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cognitive-steps", type=int, default=300)
    parser.add_argument("--max-cognitive-horizon", type=int, default=8)
    parser.add_argument("--policy-steps", type=int, default=250)
    parser.add_argument("--policy-horizon", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--rollout-steps", type=int, default=64)
    parser.add_argument("--test-count", type=int, default=20)
    parser.add_argument("--test-seed", type=int, default=20260718)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    source, goal = PRETRAIN_FACTOR[0], GOAL.view(1, -1)
    cognitive = SimpleCognitiveKAN()
    cognitive_fit = pretrain_cognitive_multistep(
        cognitive, args.cognitive_steps, 32,
        args.max_cognitive_horizon, args.seed,
    )
    model = DirectCognitivePolicy(cognitive)
    policy_fit = train_policy(
        model, args.policy_steps, args.batch_size, args.policy_horizon,
        args.seed, goal, args.gamma,
    )
    generator = torch.Generator().manual_seed(args.test_seed)
    states = _random_states(args.test_count, generator=generator)
    factors = [("source", source)]
    factors += [(f"factor_{i + 1}", f) for i, f in enumerate(FACTORS[1:])]
    factors += [(f"heldout_{i + 1}", f) for i, f in enumerate(HELDOUT)]
    results = []
    for label, factor in factors:
        results.append({"label": label, "factor": factor,
                        "normalized_distance_from_source": factor_distance(source, factor),
                        "cognitive_prediction": prediction_error(cognitive, factor),
                        "decision": evaluate(model, states, factor, goal, args.rollout_steps)})
    print(json.dumps({"architecture": "DirectCognitiveProbePolicy",
                      "training_factor": source,
                      "training_uses_only_source_factor": True,
                      "cognitive_loss_and_policy_loss_separate": True,
                      "decision_time_action_optimization": "none",
                      "teacher_usage": "none", "cognitive_fit": cognitive_fit,
                      "policy_fit": policy_fit, "results": results,
                      "test_seed": args.test_seed}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
