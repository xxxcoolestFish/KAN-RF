"""Stage-1 capacity check when several hidden dynamics factors vary."""

from __future__ import annotations

import argparse
import json

import torch

from physics_transfer.adaptive import AdaptiveCognitivePredictor
from physics_transfer.losses import cognitive_prediction_loss, token_consistency_loss
from physics_transfer.multifactor_data import sample_multifactor_batch


TRAIN_FACTORS = (
    (7.35, 0.00, 0.80, 0.80),
    (7.35, 0.08, 1.20, 1.20),
    (12.25, 0.00, 1.20, 1.20),
    (12.25, 0.08, 0.80, 0.80),
    (14.70, 0.04, 1.00, 1.00),
)
HELDOUT_FACTORS = (
    (9.80, 0.04, 1.10, 0.90),
    (13.475, 0.06, 0.90, 1.10),
)


def run_config(token_count: int, steps: int, seed: int):
    torch.manual_seed(seed)
    history_steps = 8
    model = AdaptiveCognitivePredictor(
        6, 1, history_steps * 7, token_count=token_count,
        token_dim=8, hidden_dim=24, n_prototypes=8,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)

    for index in range(steps):
        batch = sample_multifactor_batch(32, history_steps, TRAIN_FACTORS)
        output = model(batch["history"], batch["state"], batch["action"])
        loss = cognitive_prediction_loss(output["next_state"], batch["next_state"])
        loss = loss + 1e-3 * output["gates"].mean()
        factor = TRAIN_FACTORS[index % len(TRAIN_FACTORS)]
        pair_a = sample_multifactor_batch(32, history_steps, (factor,))
        pair_b = sample_multifactor_batch(32, history_steps, (factor,))
        output_a = model(pair_a["history"], pair_a["state"], pair_a["action"])
        output_b = model(pair_b["history"], pair_b["state"], pair_b["action"])
        loss = loss + 0.01 * token_consistency_loss(
            output_a["pooled"], output_b["pooled"],
            output_a["gates"], output_b["gates"],
        )
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

    with torch.no_grad():
        batch = sample_multifactor_batch(256, history_steps, HELDOUT_FACTORS)
        output = model(batch["history"], batch["state"], batch["action"])
        same_a = sample_multifactor_batch(128, history_steps, (HELDOUT_FACTORS[0],))
        same_b = sample_multifactor_batch(128, history_steps, (HELDOUT_FACTORS[0],))
        out_a = model(same_a["history"], same_a["state"], same_a["action"])
        out_b = model(same_b["history"], same_b["state"], same_b["action"])
        other = sample_multifactor_batch(128, history_steps, (HELDOUT_FACTORS[1],))
        out_other = model(other["history"], other["state"], other["action"])
        result = {
            "token_count": token_count,
            "seed": seed,
            "heldout_prediction_loss": cognitive_prediction_loss(
                output["next_state"], batch["next_state"]
            ).item(),
            "same_factor_consistency": token_consistency_loss(
                out_a["pooled"], out_b["pooled"], out_a["gates"], out_b["gates"]
            ).item(),
            "different_factor_distance": (
                out_a["pooled"].mean(0) - out_other["pooled"].mean(0)
            ).norm().item(),
            "mean_effective_slots": out_a["gates"].gt(0.5).float().sum(-1).mean().item(),
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    results = [run_config(m, args.steps, args.seed) for m in (4, 8, 16)]
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
