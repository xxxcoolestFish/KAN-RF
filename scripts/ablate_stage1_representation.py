"""Controlled stage-1 ablations for adaptive physics representations."""

from __future__ import annotations

import argparse
import json

import torch

from physics_transfer.adaptive import AdaptiveCognitivePredictor
from physics_transfer.data import sample_acrobot_batch
from physics_transfer.losses import cognitive_prediction_loss, token_consistency_loss


TRAIN_GRAVITIES = (7.35, 12.25, 14.7)
HELDOUT_GRAVITIES = (9.8, 11.0, 13.475)


def _evaluate(model, history_steps: int):
    model.eval()
    with torch.no_grad():
        heldout = sample_acrobot_batch(192, history_steps, HELDOUT_GRAVITIES)
        prediction = model(heldout["history"], heldout["state"], heldout["action"])
        same_a = sample_acrobot_batch(128, history_steps, (9.8,))
        same_b = sample_acrobot_batch(128, history_steps, (9.8,))
        out_a = model(same_a["history"], same_a["state"], same_a["action"])
        out_b = model(same_b["history"], same_b["state"], same_b["action"])
        different = sample_acrobot_batch(128, history_steps, (11.0,))
        out_d = model(different["history"], different["state"], different["action"])
        same_distance = token_consistency_loss(
            out_a["pooled"], out_b["pooled"], out_a["gates"], out_b["gates"]
        ).item()
        different_distance = (
            out_a["pooled"].mean(0) - out_d["pooled"].mean(0)
        ).norm().item()
        return {
            "heldout_prediction_loss": cognitive_prediction_loss(
                prediction["next_state"], heldout["next_state"]
            ).item(),
            "same_dynamics_consistency": same_distance,
            "different_dynamics_distance": different_distance,
            "mean_effective_slots": out_a["gates"].gt(0.5).float().sum(-1).mean().item(),
            "mean_gate": out_a["gates"].mean().item(),
        }


def run_config(token_count: int, lambda_sparse: float, lambda_consistency: float,
               steps: int, seed: int, batch_size: int = 32):
    torch.manual_seed(seed)
    history_steps = 8
    model = AdaptiveCognitivePredictor(
        state_dim=6, action_dim=1, history_dim=history_steps * 7,
        token_count=token_count, token_dim=8, hidden_dim=24, n_prototypes=8,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    model.train()

    for step_index in range(steps):
        batch = sample_acrobot_batch(batch_size, history_steps, TRAIN_GRAVITIES)
        output = model(batch["history"], batch["state"], batch["action"])
        loss = cognitive_prediction_loss(output["next_state"], batch["next_state"])
        loss = loss + lambda_sparse * output["gates"].mean()

        if lambda_consistency > 0.0:
            gravity = TRAIN_GRAVITIES[step_index % len(TRAIN_GRAVITIES)]
            pair_a = sample_acrobot_batch(batch_size, history_steps, (gravity,))
            pair_b = sample_acrobot_batch(batch_size, history_steps, (gravity,))
            output_a = model(pair_a["history"], pair_a["state"], pair_a["action"])
            output_b = model(pair_b["history"], pair_b["state"], pair_b["action"])
            loss = loss + lambda_consistency * token_consistency_loss(
                output_a["pooled"], output_b["pooled"],
                output_a["gates"], output_b["gates"],
            )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

    metrics = _evaluate(model, history_steps)
    metrics.update({
        "token_count": token_count,
        "lambda_sparse": lambda_sparse,
        "lambda_consistency": lambda_consistency,
        "seed": seed,
    })
    return metrics


def configs(mode: str):
    if mode == "capacity":
        return [(m, 1e-3, 0.1) for m in (4, 8, 16)]
    if mode == "sparsity":
        return [(8, value, 0.1) for value in (0.0, 1e-4, 1e-3, 1e-2)]
    if mode == "consistency":
        return [(8, 1e-3, value) for value in (0.0, 0.01, 0.1, 1.0)]
    raise ValueError(f"unknown mode: {mode}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("capacity", "sparsity", "consistency"),
                        default="capacity")
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    results = [
        run_config(*config, steps=args.steps, seed=args.seed)
        for config in configs(args.mode)
    ]
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
