"""Connect the frozen low-rank cognitive coefficients to the decision receiver."""

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


def receiver_code(mode: str, batch, cognitive_model):
    if mode == "oracle":
        return factor_code(batch["factors"])
    if mode == "state_only":
        return torch.zeros(batch["state"].shape[0], 4)
    if mode == "cognitive":
        with torch.no_grad():
            return cognitive_model(
                batch["history"], batch["state"], batch["action"]
            )["physics_coefficients"]
    raise ValueError(f"unknown mode: {mode}")


def train_decision(mode: str, cognitive_model, steps: int, seed: int):
    torch.manual_seed(seed + 1000)
    decision = PhysicsAwareDecision(
        state_dim=6, action_dim=1, physics_slots=4,
        hidden_dim=24, n_prototypes=8,
    )
    transport = PhysicsTransport(4, 4)
    optimizer = torch.optim.Adam(
        list(decision.parameters()) + list(transport.parameters()), lr=2e-3
    )
    for _ in range(steps):
        batch = sample_multifactor_batch(64, 8, TRAIN_FACTORS)
        target = conditional_teacher(batch["state"], batch["factors"])
        code = receiver_code(mode, batch, cognitive_model)
        receiver = transport(code)
        prediction = decision(batch["state"], receiver)
        loss = F.mse_loss(prediction, target) + 1e-3 * transport.alignment_loss()
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(decision.parameters()) + list(transport.parameters()), 5.0
        )
        optimizer.step()
    return decision, transport


def evaluate(mode: str, decision, transport, cognitive_model):
    decision.eval(); transport.eval()
    with torch.no_grad():
        batch = sample_multifactor_batch(1024, 8, HELDOUT_FACTORS)
        target = conditional_teacher(batch["state"], batch["factors"])
        code = receiver_code(mode, batch, cognitive_model)
        prediction = decision(batch["state"], transport(code))
        shuffled_code = code[torch.randperm(code.shape[0])]
        shuffled = decision(batch["state"], transport(shuffled_code))
        mse = F.mse_loss(prediction, target).item()
        shuffled_mse = F.mse_loss(shuffled, target).item()
        return {
            "mse": mse,
            "shuffled_receiver_mse": shuffled_mse,
            "receiver_shuffle_delta": shuffled_mse - mse,
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
    for mode in ("state_only", "oracle", "cognitive"):
        decision, transport = train_decision(
            mode, cognitive_model, args.decision_steps, args.seed
        )
        results.append({
            "mode": mode,
            "metrics": evaluate(mode, decision, transport, cognitive_model),
        })
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
