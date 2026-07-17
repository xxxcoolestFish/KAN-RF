"""Jointly align cognitive physics coefficients with a decision task."""

from __future__ import annotations

import argparse
import copy
import json

import torch
import torch.nn.functional as F

from physics_transfer.decision import PhysicsAwareDecision
from physics_transfer.interface import PhysicsTransport
from physics_transfer.losses import cognitive_prediction_loss, token_consistency_loss
from physics_transfer.multifactor_data import sample_multifactor_batch
from scripts.stage2_lowrank_gap_calibration import FACTORS, HELDOUT, _pair
from scripts.stage2_lowrank_semantic import rollout_loss, train as train_semantic
from scripts.stage3_oracle_receiver_probe import conditional_teacher, factor_code


def train_joint(cognitive_model, decision_weight: float, steps: int, seed: int):
    torch.manual_seed(seed + 3000)
    decision = PhysicsAwareDecision(6, 1, physics_slots=4, hidden_dim=24, n_prototypes=8)
    transport = PhysicsTransport(4, 4)
    trainable = list(cognitive_model.parameters()) + list(decision.parameters()) + list(transport.parameters())
    optimizer = torch.optim.Adam(trainable, lr=2e-3)
    for index in range(steps):
        batch = sample_multifactor_batch(16, 8, FACTORS)
        output = cognitive_model(batch["history"], batch["state"], batch["action"])
        standard = cognitive_prediction_loss(output["next_state"], batch["next_state"])
        factor_a = FACTORS[index % len(FACTORS)]
        factor_b = FACTORS[(index + 1) % len(FACTORS)]
        pair = _pair(factor_a, factor_b, batch_size=16)
        out_a = cognitive_model(pair["history_a"], pair["state"], pair["action"])
        out_b = cognitive_model(pair["history_b"], pair["state"], pair["action"])
        absolute = (
            cognitive_prediction_loss(out_a["next_state"], pair["target_a"])
            + cognitive_prediction_loss(out_b["next_state"], pair["target_b"])
        )
        gap = F.mse_loss(
            torch.linalg.vector_norm(out_a["next_state"] - out_b["next_state"], dim=-1),
            torch.linalg.vector_norm(pair["target_a"] - pair["target_b"], dim=-1),
        )
        same_a = sample_multifactor_batch(16, 8, (factor_a,))
        same_b = sample_multifactor_batch(16, 8, (factor_a,))
        same_out_a = cognitive_model(same_a["history"], same_a["state"], same_a["action"])
        same_out_b = cognitive_model(same_b["history"], same_b["state"], same_b["action"])
        token_consistency = token_consistency_loss(
            same_out_a["physics_pooled"], same_out_b["physics_pooled"],
            same_out_a["physics_gates"], same_out_b["physics_gates"],
        )
        coefficient_consistency = F.mse_loss(
            same_out_a["physics_coefficients"], same_out_b["physics_coefficients"]
        )
        receiver = transport(output["physics_coefficients"])
        target_action = conditional_teacher(batch["state"], batch["factors"])
        policy_loss = F.mse_loss(decision(batch["state"], receiver), target_action)
        horizon = rollout_loss(cognitive_model, batch, horizon=4)
        loss = (
            standard + absolute + 50.0 * gap + 0.01 * token_consistency
            + 0.1 * coefficient_consistency + 0.05 * output["basis_gram_error"].square().mean()
            + 0.01 * output["physics_residual"].square().mean()
            + 0.25 * horizon + decision_weight * policy_loss
            + 1e-3 * transport.alignment_loss()
        )
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 5.0)
        optimizer.step()
    return cognitive_model, decision, transport


def evaluate(cognitive_model, decision, transport):
    cognitive_model.eval(); decision.eval(); transport.eval()
    with torch.no_grad():
        batch = sample_multifactor_batch(256, 8, HELDOUT)
        output = cognitive_model(batch["history"], batch["state"], batch["action"])
        target_action = conditional_teacher(batch["state"], batch["factors"])
        receiver = transport(output["physics_coefficients"])
        prediction = decision(batch["state"], receiver)
        shuffled_code = output["physics_coefficients"][torch.randperm(receiver.shape[0])]
        shuffled = decision(batch["state"], transport(shuffled_code))
        mse = F.mse_loss(prediction, target_action).item()
        shuffled_mse = F.mse_loss(shuffled, target_action).item()
        return {
            "one_step_loss": rollout_loss(cognitive_model, batch, 1).item(),
            "four_step_loss": rollout_loss(cognitive_model, batch, 4).item(),
            "policy_mse": mse,
            "shuffled_policy_mse": shuffled_mse,
            "receiver_shuffle_delta": shuffled_mse - mse,
            "receiver_oracle_code_mse": F.mse_loss(
                receiver, factor_code(batch["factors"])
            ).item(),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain-steps", type=int, default=300)
    parser.add_argument("--joint-steps", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    base = train_semantic(0.1, args.pretrain_steps, args.seed)
    results = []
    for weight in (0.1, 1.0):
        model, decision, transport = train_joint(
            copy.deepcopy(base), weight, args.joint_steps, args.seed
        )
        results.append({
            "decision_weight": weight,
            "metrics": evaluate(model, decision, transport),
        })
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
