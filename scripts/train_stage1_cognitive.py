"""Stage 1: train only the adaptive cognitive dynamics representation."""

from __future__ import annotations

import argparse

import torch

from physics_transfer.adaptive import AdaptiveCognitivePredictor
from physics_transfer.data import sample_acrobot_batch
from physics_transfer.losses import cognitive_prediction_loss, gate_sparsity_loss


def run(steps: int = 300, seed: int = 42) -> dict[str, float]:
    torch.manual_seed(seed)
    history_steps = 8
    train_gravities = (7.35, 12.25, 14.7)
    model = AdaptiveCognitivePredictor(
        state_dim=6, action_dim=1, history_dim=history_steps * 7,
        token_count=8, token_dim=8, hidden_dim=24, n_prototypes=8,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)

    for _ in range(steps):
        batch = sample_acrobot_batch(64, history_steps, train_gravities)
        output = model(batch["history"], batch["state"], batch["action"])
        prediction = cognitive_prediction_loss(output["next_state"], batch["next_state"])
        loss = prediction + 1e-3 * gate_sparsity_loss(output["gates"])
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

    with torch.no_grad():
        heldout = sample_acrobot_batch(256, history_steps, (9.8, 11.0, 13.475))
        result = model(heldout["history"], heldout["state"], heldout["action"])
        heldout_loss = cognitive_prediction_loss(
            result["next_state"], heldout["next_state"]
        ).item()
        effective_slots = result["gates"].gt(0.5).float().sum(dim=-1).mean().item()

    metrics = {
        "heldout_prediction_loss": heldout_loss,
        "mean_effective_slots": effective_slots,
    }
    print(metrics)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=300)
    args = parser.parse_args()
    run(steps=args.steps)
