"""Calibrate low-rank transport with an explicit counterfactual gap loss."""

from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F

from physics_transfer.losses import cognitive_prediction_loss, token_consistency_loss
from physics_transfer.low_rank_split import LowRankSplitCognitivePredictor
from physics_transfer.multifactor_data import sample_multifactor_batch
from scripts.stage2_lowrank_loss_ablation import FACTORS, HELDOUT, _pair


def train(gap_weight: float, steps: int, seed: int):
    torch.manual_seed(seed)
    model = LowRankSplitCognitivePredictor(
        physics_rank=4, residual_scale=0.1, state_dim=6,
        action_dim=1, history_steps=8, token_count=8, token_dim=8,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    for index in range(steps):
        batch = sample_multifactor_batch(32, 8, FACTORS)
        output = model(batch["history"], batch["state"], batch["action"])
        standard = cognitive_prediction_loss(output["next_state"], batch["next_state"])
        factor_a = FACTORS[index % len(FACTORS)]
        factor_b = FACTORS[(index + 1) % len(FACTORS)]
        pair = _pair(factor_a, factor_b)
        out_a = model(pair["history_a"], pair["state"], pair["action"])
        out_b = model(pair["history_b"], pair["state"], pair["action"])
        absolute = (
            cognitive_prediction_loss(out_a["next_state"], pair["target_a"])
            + cognitive_prediction_loss(out_b["next_state"], pair["target_b"])
        )
        predicted_diff = out_a["next_state"] - out_b["next_state"]
        target_diff = pair["target_a"] - pair["target_b"]
        gap_loss = F.mse_loss(
            torch.linalg.vector_norm(predicted_diff, dim=-1),
            torch.linalg.vector_norm(target_diff, dim=-1),
        )
        same_a = sample_multifactor_batch(32, 8, (factor_a,))
        same_b = sample_multifactor_batch(32, 8, (factor_a,))
        same_out_a = model(same_a["history"], same_a["state"], same_a["action"])
        same_out_b = model(same_b["history"], same_b["state"], same_b["action"])
        consistency = token_consistency_loss(
            same_out_a["physics_pooled"], same_out_b["physics_pooled"],
            same_out_a["physics_gates"], same_out_b["physics_gates"],
        )
        orthogonality = output["basis_gram_error"].square().mean()
        energy = output["physics_residual"].square().mean()
        loss = (
            standard + absolute + gap_weight * gap_loss + 0.01 * consistency
            + 0.05 * orthogonality + 0.01 * energy
        )
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
    return model


def evaluate(model):
    model.eval()
    with torch.no_grad():
        pair = _pair(HELDOUT[0], HELDOUT[1], 128)
        out_a = model(pair["history_a"], pair["state"], pair["action"])
        out_b = model(pair["history_b"], pair["state"], pair["action"])
        predicted_gap = torch.linalg.vector_norm(
            out_a["next_state"] - out_b["next_state"], dim=-1
        ).mean().item()
        true_gap = torch.linalg.vector_norm(
            pair["target_a"] - pair["target_b"], dim=-1
        ).mean().item()
        return {
            "counterfactual_absolute_loss": (
                cognitive_prediction_loss(out_a["next_state"], pair["target_a"])
                + cognitive_prediction_loss(out_b["next_state"], pair["target_b"])
            ).item(),
            "counterfactual_predicted_gap": predicted_gap,
            "counterfactual_true_gap": true_gap,
            "counterfactual_gap_ratio": predicted_gap / max(true_gap, 1e-8),
            "coefficient_gap": torch.linalg.vector_norm(
                out_a["physics_coefficients"] - out_b["physics_coefficients"], dim=-1
            ).mean().item(),
            "physics_consistency": token_consistency_loss(
                out_a["physics_pooled"], out_b["physics_pooled"],
                out_a["physics_gates"], out_b["physics_gates"],
            ).item(),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    results = []
    for weight in (1.0, 10.0, 50.0):
        metrics = evaluate(train(weight, args.steps, args.seed))
        metrics["gap_weight"] = weight
        results.append(metrics)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
