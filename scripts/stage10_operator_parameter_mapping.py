"""Probe cognitive-operator to decision-parameter pretraining."""

from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F

from physics_transfer.separated_decision import SeparatedPhysicsDecision
from physics_transfer.operator_hypernetwork import OperatorMappedDecision
from physics_transfer.transition_data import sample_transition_sequence_batch
from scripts.stage2_lowrank_gap_calibration import HELDOUT
from scripts.stage6_single_env_adaptation import PRETRAIN_FACTOR, pretrain
from scripts.stage7_mpc_decision_adaptation import cognitive_mpc_teacher, operator_query, true_mpc_teacher


OPERATOR_DIM = 54


def train_state_base(cognitive, steps: int, sequence_steps: int, seed: int):
    """Pretrain only the task branch on cognitive targets with zero physics input."""
    torch.manual_seed(seed + 5000)
    decision = SeparatedPhysicsDecision(6, 1, OPERATOR_DIM, hidden_dim=24, n_prototypes=8)
    with torch.no_grad(): decision.physics_basis.zero_()
    decision.physics_basis.requires_grad = False
    optimizer = torch.optim.Adam(
        [p for p in decision.parameters() if p.requires_grad], lr=2e-3
    )
    cognitive.eval()
    for _ in range(steps):
        batch = sample_transition_sequence_batch(32, sequence_steps, PRETRAIN_FACTOR)
        with torch.no_grad():
            output = cognitive.forward_sequence(batch["state"], batch["action"], batch["next_state"])
            state = batch["state"].reshape(-1, 6)
            latent = output["pre_latents"].reshape(-1, cognitive.latent_dim)
            target = cognitive_mpc_teacher(cognitive, state, latent)
        prediction = decision(state, torch.zeros(state.shape[0], OPERATOR_DIM))["action"]
        loss = F.mse_loss(prediction, target)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    return decision


def train_mapper(cognitive, base, steps: int, sequence_steps: int, seed: int, rank: int):
    torch.manual_seed(seed + 6000)
    decision = OperatorMappedDecision(base, OPERATOR_DIM, rank=rank)
    optimizer = torch.optim.Adam(decision.mapper.parameters(), lr=2e-3)
    cognitive.eval()
    for _ in range(steps):
        batch = sample_transition_sequence_batch(32, sequence_steps, PRETRAIN_FACTOR)
        with torch.no_grad():
            output = cognitive.forward_sequence(batch["state"], batch["action"], batch["next_state"])
            state = batch["state"].reshape(-1, 6)
            latent = output["pre_latents"].reshape(-1, cognitive.latent_dim)
            operator = operator_query(cognitive, state, latent)
            target = cognitive_mpc_teacher(cognitive, state, latent)
        prediction = decision(state, operator)["action"]
        loss = F.mse_loss(prediction, target) + 1e-3 * decision.adapter_norm(operator).square()
        optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(decision.mapper.parameters(), 5.0); optimizer.step()
    return decision


def evaluate(cognitive, decision, factor, sequence_steps: int):
    cognitive.eval(); decision.eval()
    with torch.no_grad():
        batch = sample_transition_sequence_batch(128, sequence_steps, (factor,))
        output = cognitive.forward_sequence(batch["state"], batch["action"], batch["next_state"])
        state = batch["state"].reshape(-1, 6)
        latent = output["pre_latents"].reshape(-1, cognitive.latent_dim)
        operator = operator_query(cognitive, state, latent)
        cognitive_target = cognitive_mpc_teacher(cognitive, state, latent)
        true_target = true_mpc_teacher(state, factor)
        prediction = decision(state, operator)["action"]
        shuffled = decision(state, operator[torch.randperm(operator.shape[0])])["action"]
        return {
            "cognitive_target_mse": F.mse_loss(prediction, cognitive_target).item(),
            "true_mpc_mse": F.mse_loss(prediction, true_target).item(),
            "shuffle_delta": F.mse_loss(shuffled, cognitive_target).item() - F.mse_loss(prediction, cognitive_target).item(),
            "adapter_norm": decision.adapter_norm(operator).item(),
        }


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--cognitive-steps", type=int, default=100); parser.add_argument("--base-steps", type=int, default=100); parser.add_argument("--mapper-steps", type=int, default=100); parser.add_argument("--sequence-steps", type=int, default=16); parser.add_argument("--rank", type=int, default=1); parser.add_argument("--seed", type=int, default=42); args = parser.parse_args()
    cognitive = pretrain(args.cognitive_steps, args.sequence_steps, args.seed)
    base = train_state_base(cognitive, args.base_steps, args.sequence_steps, args.seed)
    decision = train_mapper(cognitive, base, args.mapper_steps, args.sequence_steps, args.seed, args.rank)
    results = []
    for factor in (PRETRAIN_FACTOR[0], *HELDOUT):
        results.append({"factor": factor, "metrics": evaluate(cognitive, decision, factor, args.sequence_steps)})
    print(json.dumps({"pretrain_factor": PRETRAIN_FACTOR[0], "rank": args.rank, "results": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
