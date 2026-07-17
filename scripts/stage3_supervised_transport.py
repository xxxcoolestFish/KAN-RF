"""Diagnostic: align cognitive coefficients to oracle coordinates before policy use."""

from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F

from physics_transfer.decision import PhysicsAwareDecision
from physics_transfer.interface import PhysicsTransport
from physics_transfer.multifactor_data import sample_multifactor_batch
from scripts.stage2_lowrank_mixed_horizon import train as train_cognitive
from scripts.stage3_oracle_receiver_probe import (
    HELDOUT_FACTORS,
    TRAIN_FACTORS,
    conditional_teacher,
    factor_code,
)


def train_transport(cognitive_model, align_weight: float, steps: int, seed: int):
    torch.manual_seed(seed + 2000)
    decision = PhysicsAwareDecision(6, 1, physics_slots=4, hidden_dim=24, n_prototypes=8)
    transport = PhysicsTransport(4, 4)
    optimizer = torch.optim.Adam(
        list(decision.parameters()) + list(transport.parameters()), lr=2e-3
    )
    for _ in range(steps):
        batch = sample_multifactor_batch(64, 8, TRAIN_FACTORS)
        with torch.no_grad():
            code = cognitive_model(
                batch["history"], batch["state"], batch["action"]
            )["physics_coefficients"]
        oracle = factor_code(batch["factors"])
        receiver = transport(code)
        target = conditional_teacher(batch["state"], batch["factors"])
        prediction = decision(batch["state"], receiver)
        loss = (
            F.mse_loss(prediction, target)
            + align_weight * F.mse_loss(receiver, oracle)
            + 1e-3 * transport.alignment_loss()
        )
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(decision.parameters()) + list(transport.parameters()), 5.0
        )
        optimizer.step()
    return decision, transport


def evaluate(cognitive_model, decision, transport):
    decision.eval(); transport.eval()
    with torch.no_grad():
        batch = sample_multifactor_batch(1024, 8, HELDOUT_FACTORS)
        code = cognitive_model(
            batch["history"], batch["state"], batch["action"]
        )["physics_coefficients"]
        oracle = factor_code(batch["factors"])
        target = conditional_teacher(batch["state"], batch["factors"])
        receiver = transport(code)
        prediction = decision(batch["state"], receiver)
        shuffled = decision(batch["state"], transport(code[torch.randperm(code.shape[0])]))
        return {
            "mse": F.mse_loss(prediction, target).item(),
            "shuffled_receiver_mse": F.mse_loss(shuffled, target).item(),
            "receiver_shuffle_delta": F.mse_loss(shuffled, target).item()
            - F.mse_loss(prediction, target).item(),
            "transport_oracle_mse": F.mse_loss(receiver, oracle).item(),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cognitive-steps", type=int, default=300)
    parser.add_argument("--decision-steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    cognitive_model = train_cognitive(0.25, args.cognitive_steps, args.seed)
    cognitive_model.eval()
    results = []
    for weight in (0.0, 0.1, 1.0):
        decision, transport = train_transport(
            cognitive_model, weight, args.decision_steps, args.seed
        )
        results.append({
            "align_weight": weight,
            "metrics": evaluate(cognitive_model, decision, transport),
        })
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
