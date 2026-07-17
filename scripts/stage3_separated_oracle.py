"""Oracle probe for separated task and physical decision branches."""

from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F

from physics_transfer.multifactor_data import sample_multifactor_batch
from physics_transfer.separated_decision import ConcatPhysicsDecision, SeparatedPhysicsDecision
from scripts.stage2_lowrank_gap_calibration import HELDOUT, _pair
from scripts.stage3_oracle_receiver_probe import TRAIN_FACTORS, conditional_teacher, factor_code


def build(mode: str):
    if mode == "concat":
        return ConcatPhysicsDecision(6, 1, 4, hidden_dim=24, n_prototypes=8)
    return SeparatedPhysicsDecision(6, 1, 4, hidden_dim=24, n_prototypes=8)


def forward(model, mode, state, code):
    if mode == "state_only":
        code = torch.zeros_like(code)
    return model(state, code)


def train(mode: str, steps: int, seed: int):
    torch.manual_seed(seed)
    model = build(mode)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    for index in range(steps):
        batch = sample_multifactor_batch(64, 8, TRAIN_FACTORS)
        code = factor_code(batch["factors"])
        target = conditional_teacher(batch["state"], batch["factors"])
        output = forward(model, mode, batch["state"], code)
        loss = F.mse_loss(output["action"], target)
        factor_a = TRAIN_FACTORS[index % len(TRAIN_FACTORS)]
        factor_b = TRAIN_FACTORS[(index + 1) % len(TRAIN_FACTORS)]
        pair = _pair(factor_a, factor_b, batch_size=64)
        fa = torch.tensor(factor_a).float().repeat(64, 1)
        fb = torch.tensor(factor_b).float().repeat(64, 1)
        code_a = factor_code(fa)
        code_b = factor_code(fb)
        target_a = conditional_teacher(pair["state"], fa)
        target_b = conditional_teacher(pair["state"], fb)
        pred_a = forward(model, mode, pair["state"], code_a)["action"]
        pred_b = forward(model, mode, pair["state"], code_b)["action"]
        loss = loss + F.mse_loss(pred_a, target_a) + F.mse_loss(pred_b, target_b)
        loss = loss + 10.0 * F.mse_loss(pred_a - pred_b, target_a - target_b)
        if mode == "separated":
            loss = loss + 0.05 * model.separation_loss()
            loss = loss + 0.05 * model.physics_basis_orthogonality_loss()
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
    return model


def evaluate(model, mode: str):
    model.eval()
    with torch.no_grad():
        batch = sample_multifactor_batch(1024, 8, HELDOUT)
        code = factor_code(batch["factors"])
        target = conditional_teacher(batch["state"], batch["factors"])
        prediction = forward(model, mode, batch["state"], code)["action"]
        shuffled = forward(model, mode, batch["state"], code[torch.randperm(1024)])["action"]
        pair = _pair(HELDOUT[0], HELDOUT[1], batch_size=1024)
        fa = torch.tensor(HELDOUT[0]).float().repeat(1024, 1)
        fb = torch.tensor(HELDOUT[1]).float().repeat(1024, 1)
        target_a = conditional_teacher(pair["state"], fa)
        target_b = conditional_teacher(pair["state"], fb)
        pred_a = forward(model, mode, pair["state"], factor_code(fa))["action"]
        pred_b = forward(model, mode, pair["state"], factor_code(fb))["action"]
        predicted_gap = (pred_a - pred_b).abs().mean().item()
        true_gap = (target_a - target_b).abs().mean().item()
        return {
            "mse": F.mse_loss(prediction, target).item(),
            "shuffled_mse": F.mse_loss(shuffled, target).item(),
            "shuffle_delta": F.mse_loss(shuffled, target).item() - F.mse_loss(prediction, target).item(),
            "predicted_action_gap": predicted_gap,
            "true_action_gap": true_gap,
            "action_gap_ratio": predicted_gap / max(true_gap, 1e-8),
        }


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--steps", type=int, default=300); parser.add_argument("--seed", type=int, default=42); args = parser.parse_args()
    results = []
    for mode in ("state_only", "concat", "separated"):
        model = train(mode, args.steps, args.seed)
        results.append({"mode": mode, "metrics": evaluate(model, mode)})
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
