"""Run the first complete cognitive--decision closed-loop experiment.

The experiment deliberately uses one pipeline rather than an ablation table:

1. pretrain the streaming cognitive model on one factor;
2. initialize the decision task/physics branches from a cognitive operator;
3. let the decision network act in the differentiable environment;
4. update the cognitive model from the observed transition;
5. send a fresh operator to a separate, online decision residual.

The initialization mapper is frozen after step 2.  Runtime transfer is a
separate zero-initialized residual, so the experiment tests the distinction
between one-time parameter initialization and continual physical feedback.
"""

from __future__ import annotations

import argparse
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from physics_transfer.operator_hypernetwork import OperatorMappedDecision
from physics_transfer.multifactor_data import _random_states
from physics_transfer.transition_data import sample_transition_sequence_batch
from physics_transfer.variants import step
from scripts.stage2_lowrank_gap_calibration import HELDOUT
from scripts.stage6_single_env_adaptation import PRETRAIN_FACTOR, pretrain
from scripts.stage7_mpc_decision_adaptation import cognitive_mpc_teacher, operator_query
from scripts.stage10_operator_parameter_mapping import train_mapper, train_state_base


OPERATOR_DIM = 54
SUCCESS_COST = 0.25


class RuntimeOperatorResidual(nn.Module):
    """Frozen initialized policy plus a separately learned online residual."""

    def __init__(self, mapped: OperatorMappedDecision, init_operator: torch.Tensor,
                 residual_scale: float = 0.20):
        super().__init__()
        self.task_trunk = mapped.task_trunk
        self.task_head = mapped.task_head
        with torch.no_grad():
            init_basis = mapped.base_basis + mapped.mapper(init_operator).mean(dim=0)
        self.register_buffer("init_basis", init_basis.detach().clone())
        self.residual_scale = residual_scale
        self.runtime_residual = nn.Sequential(
            nn.Linear(6 + OPERATOR_DIM, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )
        # At deployment the initialized policy is used first; online transfer
        # learns only from subsequent observed transitions.
        nn.init.zeros_(self.runtime_residual[-1].weight)
        nn.init.zeros_(self.runtime_residual[-1].bias)
        for parameter in self.task_trunk.parameters():
            parameter.requires_grad = False
        for parameter in self.task_head.parameters():
            parameter.requires_grad = False

    def forward(self, state: torch.Tensor, operator: torch.Tensor):
        hidden = torch.tanh(self.task_trunk(state))
        task_logits = self.task_head(hidden)
        physics_logits = torch.einsum("pah,bh->ba", self.init_basis, hidden)
        residual = self.residual_scale * torch.tanh(
            self.runtime_residual(torch.cat([state, operator], dim=-1))
        )
        logits = task_logits + physics_logits + residual
        return {
            "action": torch.tanh(logits),
            "logits": logits,
            "residual": residual,
        }


def collect_initial_operator(cognitive, sequence_steps: int, seed: int):
    """Estimate a fixed pretraining operator used only for initialization."""
    torch.manual_seed(seed + 7000)
    operators = []
    with torch.no_grad():
        for _ in range(8):
            batch = sample_transition_sequence_batch(32, sequence_steps, PRETRAIN_FACTOR)
            output = cognitive.forward_sequence(
                batch["state"], batch["action"], batch["next_state"]
            )
            state = batch["state"].reshape(-1, 6)
            latent = output["pre_latents"].reshape(-1, cognitive.latent_dim)
            operators.append(operator_query(cognitive, state, latent))
    return torch.cat(operators, dim=0).mean(dim=0, keepdim=True)


def build_pipeline(args):
    cognitive = pretrain(args.cognitive_steps, args.sequence_steps, args.seed)
    base = train_state_base(cognitive, args.base_steps, args.sequence_steps, args.seed)
    mapped = train_mapper(
        cognitive, base, args.mapper_steps, args.sequence_steps, args.seed, rank=args.rank
    )
    init_operator = collect_initial_operator(cognitive, args.sequence_steps, args.seed)
    decision = RuntimeOperatorResidual(mapped, init_operator)
    return cognitive, decision


def run_episode(cognitive, decision, cognitive_optimizer, decision_optimizer,
                factor, steps: int, seed: int):
    """Run one actual decision-generated trajectory and update both networks."""
    torch.manual_seed(seed)
    state = _random_states(1)
    latent = cognitive.initial_latent(1)
    costs, action_errors, prediction_errors, residual_norms = [], [], [], []
    success = False

    for _ in range(steps):
        # The action is produced before observing the next state.
        with torch.no_grad():
            operator = operator_query(cognitive, state, latent)
            cognitive_target = cognitive_mpc_teacher(cognitive, state, latent)
        decision_output = decision(state, operator)
        action = decision_output["action"]

        # This is the environment transition generated by the decision action.
        factor_tensor = torch.tensor(factor, dtype=state.dtype).repeat(state.shape[0], 1)
        next_state = step(
            state, action.detach(), factor_tensor[:, 0], factor_tensor[:, 1],
            factor_tensor[:, 2], factor_tensor[:, 3],
        )

        # Slow cognitive learning from the newly observed transition.
        prediction = cognitive.predict_next(state, action.detach(), latent)
        cognitive_loss = F.smooth_l1_loss(prediction, next_state)
        cognitive_optimizer.zero_grad()
        cognitive_loss.backward()
        torch.nn.utils.clip_grad_norm_(cognitive.parameters(), 5.0)
        cognitive_optimizer.step()

        # Fast decision adaptation through a separate runtime residual.
        decision_loss = F.mse_loss(action, cognitive_target)
        decision_optimizer.zero_grad()
        decision_loss.backward()
        torch.nn.utils.clip_grad_norm_(decision.runtime_residual.parameters(), 5.0)
        decision_optimizer.step()

        with torch.no_grad():
            latent = cognitive.observe_transition(
                state, action.detach(), next_state, latent
            ).detach()
            current_cost = (
                0.5 * next_state[:, 4].square()
                + 0.5 * next_state[:, 5].square()
                + 0.5 * (1.0 - next_state[:, 0])
                + 0.25 * (1.0 - next_state[:, 2])
            )
            costs.append(current_cost.item())
            action_errors.append(F.mse_loss(action.detach(), cognitive_target).item())
            prediction_errors.append(cognitive_loss.item())
            residual_norms.append(decision_output["residual"].norm().item())
            success = success or current_cost.item() <= SUCCESS_COST
            state = next_state.detach()

    return {
        "success": bool(success),
        "min_cost": min(costs),
        "final_cost": costs[-1],
        "mean_action_mse": sum(action_errors) / len(action_errors),
        "mean_prediction_mse": sum(prediction_errors) / len(prediction_errors),
        "mean_runtime_residual": sum(residual_norms) / len(residual_norms),
    }


def run_phase(cognitive, decision, factor, args, seed_offset: int):
    cognitive_optimizer = torch.optim.Adam(cognitive.parameters(), lr=args.cognitive_online_lr)
    decision_optimizer = torch.optim.Adam(
        decision.runtime_residual.parameters(), lr=args.decision_online_lr
    )
    episodes = []
    for episode in range(args.episodes):
        episodes.append(
            run_episode(
                cognitive, decision, cognitive_optimizer, decision_optimizer,
                factor, args.rollout_steps, args.seed + seed_offset + episode,
            )
        )
    return episodes


def summarize(episodes):
    return {
        "success_count": sum(item["success"] for item in episodes),
        "success_rate": sum(item["success"] for item in episodes) / max(len(episodes), 1),
        "first_success_rate": sum(item["success"] for item in episodes[:3]) / max(min(3, len(episodes)), 1),
        "last_success_rate": sum(item["success"] for item in episodes[-3:]) / max(min(3, len(episodes)), 1),
        "mean_min_cost": sum(item["min_cost"] for item in episodes) / max(len(episodes), 1),
        "mean_action_mse": sum(item["mean_action_mse"] for item in episodes) / max(len(episodes), 1),
        "mean_prediction_mse": sum(item["mean_prediction_mse"] for item in episodes) / max(len(episodes), 1),
        "episodes": episodes,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cognitive-steps", type=int, default=200)
    parser.add_argument("--base-steps", type=int, default=200)
    parser.add_argument("--mapper-steps", type=int, default=200)
    parser.add_argument("--sequence-steps", type=int, default=16)
    parser.add_argument("--rollout-steps", type=int, default=64)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--cognitive-online-lr", type=float, default=2e-4)
    parser.add_argument("--decision-online-lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cognitive, decision = build_pipeline(args)
    same_environment = run_phase(cognitive, decision, PRETRAIN_FACTOR[0], args, 1000)
    changed_environment = run_phase(cognitive, decision, HELDOUT[0], args, 3000)
    print(json.dumps({
        "pretrain_factor": PRETRAIN_FACTOR[0],
        "changed_factor": HELDOUT[0],
        "success_threshold_cost": SUCCESS_COST,
        "same_environment": summarize(same_environment),
        "changed_environment": summarize(changed_environment),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
