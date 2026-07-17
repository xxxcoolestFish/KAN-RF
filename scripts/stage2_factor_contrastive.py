"""Weakly supervised context contrastive objective for cognitive coefficients.

The loss only uses context equivalence (same/different environment episode);
it never assigns a meaning to a coefficient coordinate or assumes a known
number of physical parameters.
"""

from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F

from physics_transfer.losses import cognitive_prediction_loss, token_consistency_loss
from physics_transfer.low_rank_split import LowRankSplitCognitivePredictor
from physics_transfer.multifactor_data import sample_multifactor_batch
from scripts.stage2_lowrank_gap_calibration import FACTORS, HELDOUT, _pair
from scripts.stage2_lowrank_semantic import rollout_loss


def train(weight: float, steps: int, seed: int, margin: float = 0.5):
    torch.manual_seed(seed)
    model = LowRankSplitCognitivePredictor(
        physics_rank=4, residual_scale=0.1, state_dim=6,
        action_dim=1, history_steps=8, token_count=8, token_dim=8,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    for index in range(steps):
        batch = sample_multifactor_batch(16, 8, FACTORS)
        output = model(batch["history"], batch["state"], batch["action"])
        standard = cognitive_prediction_loss(output["next_state"], batch["next_state"])
        factor_a = FACTORS[index % len(FACTORS)]
        factor_b = FACTORS[(index + 1) % len(FACTORS)]
        pair = _pair(factor_a, factor_b, batch_size=16)
        out_a = model(pair["history_a"], pair["state"], pair["action"])
        out_b = model(pair["history_b"], pair["state"], pair["action"])
        absolute = cognitive_prediction_loss(out_a["next_state"], pair["target_a"]) + cognitive_prediction_loss(out_b["next_state"], pair["target_b"])
        gap = F.mse_loss(
            torch.linalg.vector_norm(out_a["next_state"] - out_b["next_state"], dim=-1),
            torch.linalg.vector_norm(pair["target_a"] - pair["target_b"], dim=-1),
        )
        same_a = sample_multifactor_batch(16, 8, (factor_a,))
        same_b = sample_multifactor_batch(16, 8, (factor_a,))
        same_out_a = model(same_a["history"], same_a["state"], same_a["action"])
        same_out_b = model(same_b["history"], same_b["state"], same_b["action"])
        code_a = same_out_a["physics_coefficients"]
        code_b = same_out_b["physics_coefficients"]
        diff_a = out_a["physics_coefficients"]
        diff_b = out_b["physics_coefficients"]
        same_loss = F.mse_loss(code_a, code_b)
        diff_distance = torch.linalg.vector_norm(
            F.normalize(diff_a, dim=-1) - F.normalize(diff_b, dim=-1), dim=-1
        )
        diff_loss = F.relu(margin - diff_distance).square().mean()
        consistency = token_consistency_loss(
            same_out_a["physics_pooled"], same_out_b["physics_pooled"],
            same_out_a["physics_gates"], same_out_b["physics_gates"],
        )
        horizon = rollout_loss(model, batch, horizon=4)
        loss = standard + absolute + 50.0 * gap + 0.01 * consistency
        loss = loss + weight * (same_loss + diff_loss)
        loss = loss + 0.05 * output["basis_gram_error"].square().mean() + 0.01 * output["physics_residual"].square().mean() + 0.25 * horizon
        optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
    return model


def evaluate(model):
    model.eval()
    with torch.no_grad():
        same_a = sample_multifactor_batch(256, 8, (HELDOUT[0],))
        same_b = sample_multifactor_batch(256, 8, (HELDOUT[0],))
        different = sample_multifactor_batch(256, 8, (HELDOUT[1],))
        code_a = model(same_a["history"], same_a["state"], same_a["action"])["physics_coefficients"]
        code_b = model(same_b["history"], same_b["state"], same_b["action"])["physics_coefficients"]
        code_d = model(different["history"], different["state"], different["action"])["physics_coefficients"]
        same = torch.linalg.vector_norm(code_a - code_b, dim=-1).mean().item()
        diff = torch.linalg.vector_norm(code_a - code_d, dim=-1).mean().item()
        heldout = sample_multifactor_batch(128, 8, HELDOUT)
        return {"same_code_distance": same, "different_code_distance": diff, "separation_margin": diff - same, "one_step_loss": rollout_loss(model, heldout, 1).item(), "four_step_loss": rollout_loss(model, heldout, 4).item()}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--steps", type=int, default=100); parser.add_argument("--seed", type=int, default=42); args = parser.parse_args()
    results = []
    for weight in (1.0, 10.0, 50.0):
        metrics = evaluate(train(weight, args.steps, args.seed)); metrics["contrastive_weight"] = weight; results.append(metrics)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
