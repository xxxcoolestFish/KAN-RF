"""Smooth conditional-control probe for the decision physics receiver."""

from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F

from physics_transfer.decision import PhysicsAwareDecision
from physics_transfer.multifactor_data import sample_multifactor_batch


TRAIN_FACTORS = (
    (7.35, 0.00, 0.80, 0.80), (7.35, 0.08, 1.20, 1.20),
    (12.25, 0.00, 1.20, 1.20), (12.25, 0.08, 0.80, 0.80),
    (14.70, 0.04, 1.00, 1.00),
)
HELDOUT_FACTORS = (
    (9.80, 0.04, 1.10, 0.90), (13.475, 0.06, 0.90, 1.10),
)


def factor_code(factors: torch.Tensor) -> torch.Tensor:
    center = torch.tensor([11.025, 0.04, 1.0, 1.0], dtype=factors.dtype)
    scale = torch.tensor([3.675, 0.04, 0.2, 0.2], dtype=factors.dtype)
    return (factors - center.to(factors.device)) / scale.to(factors.device)


def conditional_teacher(state: torch.Tensor, factors: torch.Tensor) -> torch.Tensor:
    """A smooth control target with state--physics interactions."""
    code = factor_code(factors)
    signal = (
        0.25 * state[:, 4]
        + 0.60 * code[:, 0] * state[:, 4]
        + 0.45 * code[:, 1] * state[:, 5]
        + 0.45 * code[:, 2] * state[:, 1]
        - 0.45 * code[:, 3] * state[:, 3]
    )
    return torch.tanh(signal).unsqueeze(-1)


def make_batch(batch_size: int, factors):
    batch = sample_multifactor_batch(batch_size, 8, factors)
    return batch, conditional_teacher(batch["state"], batch["factors"])


def train(mode: str, steps: int, seed: int):
    torch.manual_seed(seed)
    model = PhysicsAwareDecision(
        state_dim=6, action_dim=1, physics_slots=4,
        hidden_dim=24, n_prototypes=8,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    for _ in range(steps):
        batch, target = make_batch(64, TRAIN_FACTORS)
        code = factor_code(batch["factors"])
        receiver = code if mode == "oracle" else torch.zeros_like(code)
        prediction = model(batch["state"], receiver)
        loss = F.mse_loss(prediction, target)
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
    return model


def evaluate(model, mode: str, factor):
    model.eval()
    with torch.no_grad():
        batch, target = make_batch(512, (factor,))
        code = factor_code(batch["factors"])
        receiver = code if mode == "oracle" else torch.zeros_like(code)
        prediction = model(batch["state"], receiver)
        shuffled = model(batch["state"], receiver.flip(0))
        return {
            "factor": factor,
            "mse": F.mse_loss(prediction, target).item(),
            "shuffled_receiver_mse": F.mse_loss(shuffled, target).item(),
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
        results.append({
            "mode": mode,
            "metrics": [evaluate(model, mode, factor) for factor in HELDOUT_FACTORS],
        })
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
