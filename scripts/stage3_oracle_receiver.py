"""Oracle test for the decision network's explicit physics receiver."""

from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F

from physics_transfer.decision import PhysicsAwareDecision
from physics_transfer.multifactor_data import sample_multifactor_batch
from physics_transfer.variants import step as variant_step


TRAIN_FACTORS = (
    (7.35, 0.00, 0.80, 0.80), (7.35, 0.08, 1.20, 1.20),
    (12.25, 0.00, 1.20, 1.20), (12.25, 0.08, 0.80, 0.80),
    (14.70, 0.04, 1.00, 1.00),
)
HELDOUT_FACTORS = (
    (9.80, 0.04, 1.10, 0.90), (13.475, 0.06, 0.90, 1.10),
)


def factor_code(factors: torch.Tensor) -> torch.Tensor:
    """Normalize oracle factors to comparable receiver coordinates."""
    center = torch.tensor([11.025, 0.04, 1.0, 1.0], dtype=factors.dtype)
    scale = torch.tensor([3.675, 0.04, 0.2, 0.2], dtype=factors.dtype)
    return (factors - center.to(factors.device)) / scale.to(factors.device)


def tip_height(state: torch.Tensor) -> torch.Tensor:
    theta1 = torch.atan2(state[:, 1], state[:, 0])
    theta2 = torch.atan2(state[:, 3], state[:, 2])
    return -torch.cos(theta1) - torch.cos(theta1 + theta2)


def teacher_action(state: torch.Tensor, factors: torch.Tensor,
                   horizon: int = 3) -> torch.Tensor:
    """Choose the constant torque with the best short-horizon tip height."""
    plus = state
    minus = state
    plus_action = torch.ones(state.shape[0], 1)
    minus_action = -torch.ones(state.shape[0], 1)
    for _ in range(horizon):
        plus = variant_step(
            plus, plus_action, factors[:, 0], factors[:, 1],
            factors[:, 2], factors[:, 3]
        )
        minus = variant_step(
            minus, minus_action, factors[:, 0], factors[:, 1],
            factors[:, 2], factors[:, 3]
        )
    return torch.where(
        tip_height(plus).unsqueeze(-1) >= tip_height(minus).unsqueeze(-1),
        plus_action, minus_action,
    )


def make_batch(batch_size: int, factors):
    batch = sample_multifactor_batch(batch_size, 8, factors)
    target = teacher_action(batch["state"], batch["factors"])
    return batch, target


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
            "sign_accuracy": (prediction.sign() == target.sign()).float().mean().item(),
            "shuffled_receiver_mse": F.mse_loss(shuffled, target).item(),
            "positive_target_fraction": (target > 0).float().mean().item(),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    results = []
    for mode in ("state_only", "oracle"):
        model = train(mode, args.steps, args.seed)
        metrics = [evaluate(model, mode, factor) for factor in HELDOUT_FACTORS]
        results.append({"mode": mode, "metrics": metrics})
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
