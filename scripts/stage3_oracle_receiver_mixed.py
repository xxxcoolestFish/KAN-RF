"""Mixed-factor evaluation for the oracle decision receiver."""

from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F

from physics_transfer.multifactor_data import sample_multifactor_batch
from scripts.stage3_oracle_receiver_probe import (
    HELDOUT_FACTORS,
    conditional_teacher,
    factor_code,
    train,
)


def evaluate(model, mode: str):
    model.eval()
    with torch.no_grad():
        batch = sample_multifactor_batch(1024, 8, HELDOUT_FACTORS)
        target = conditional_teacher(batch["state"], batch["factors"])
        code = factor_code(batch["factors"])
        receiver = code if mode == "oracle" else torch.zeros_like(code)
        prediction = model(batch["state"], receiver)
        shuffled_receiver = receiver[torch.randperm(receiver.shape[0])]
        shuffled = model(batch["state"], shuffled_receiver)
        mse = F.mse_loss(prediction, target).item()
        shuffled_mse = F.mse_loss(shuffled, target).item()
        return {
            "mse": mse,
            "shuffled_receiver_mse": shuffled_mse,
            "receiver_shuffle_delta": shuffled_mse - mse,
            "target_std": target.std().item(),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    results = []
    for mode in ("state_only", "oracle"):
        model = train(mode, args.steps, args.seed)
        results.append({"mode": mode, "metrics": evaluate(model, mode)})
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
