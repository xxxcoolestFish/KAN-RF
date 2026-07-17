"""Learn a gauge-free functional latent geometry from same/different contexts."""

from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F

from physics_transfer.losses import token_consistency_loss, cognitive_prediction_loss
from physics_transfer.multifactor_data import sample_multifactor_batch
from scripts.shared_dictionary_utils import dynamics_terms
from scripts.stage2_lowrank_gap_calibration import FACTORS, HELDOUT, _pair
from scripts.stage2_lowrank_semantic import rollout_loss
from physics_transfer.low_rank_split import LowRankSplitCognitivePredictor


def train(contrastive_weight: float, steps: int, seed: int, margin: float = 0.5):
    torch.manual_seed(seed)
    model = LowRankSplitCognitivePredictor(
        physics_rank=4, residual_scale=0.1, state_dim=6,
        action_dim=1, history_steps=8, token_count=8, token_dim=8,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    for index in range(steps):
        batch = sample_multifactor_batch(16, 8, FACTORS)
        pair, output, out_a, out_b, standard, absolute, gap, token, _ = dynamics_terms(model, batch, index)
        factor_a = FACTORS[index % len(FACTORS)]
        same_a = sample_multifactor_batch(16, 8, (factor_a,))
        same_b = sample_multifactor_batch(16, 8, (factor_a,))
        same_out_a = model(same_a["history"], same_a["state"], same_a["action"])
        same_out_b = model(same_b["history"], same_b["state"], same_b["action"])
        same_code_a = F.normalize(same_out_a["physics_coefficients"], dim=-1)
        same_code_b = F.normalize(same_out_b["physics_coefficients"], dim=-1)
        same_loss = F.mse_loss(same_code_a, same_code_b)
        diff_code_a = F.normalize(out_a["physics_coefficients"], dim=-1)
        diff_code_b = F.normalize(out_b["physics_coefficients"], dim=-1)
        diff_distance = torch.linalg.vector_norm(diff_code_a - diff_code_b, dim=-1)
        diff_loss = F.relu(margin - diff_distance).square().mean()
        horizon = rollout_loss(model, batch, horizon=4)
        loss = standard + absolute + 50.0 * gap + 0.01 * token
        loss = loss + contrastive_weight * (same_loss + diff_loss)
        loss = loss + 0.05 * output["basis_gram_error"].square().mean()
        loss = loss + 0.01 * output["physics_residual"].square().mean() + 0.25 * horizon
        optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
    return model


def evaluate(model):
    model.eval()
    with torch.no_grad():
        same_a = sample_multifactor_batch(256, 8, (HELDOUT[0],))
        same_b = sample_multifactor_batch(256, 8, (HELDOUT[0],))
        different = sample_multifactor_batch(256, 8, (HELDOUT[1],))
        out_a = model(same_a["history"], same_a["state"], same_a["action"])
        out_b = model(same_b["history"], same_b["state"], same_b["action"])
        out_d = model(different["history"], different["state"], different["action"])
        code_a = F.normalize(out_a["physics_coefficients"], dim=-1)
        code_b = F.normalize(out_b["physics_coefficients"], dim=-1)
        code_d = F.normalize(out_d["physics_coefficients"], dim=-1)
        same_distance = torch.linalg.vector_norm(code_a - code_b, dim=-1).mean().item()
        different_distance = torch.linalg.vector_norm(code_a - code_d, dim=-1).mean().item()
        heldout = sample_multifactor_batch(128, 8, HELDOUT)
        return {
            "same_latent_distance": same_distance,
            "different_latent_distance": different_distance,
            "separation_margin": different_distance - same_distance,
            "one_step_loss": rollout_loss(model, heldout, 1).item(),
            "four_step_loss": rollout_loss(model, heldout, 4).item(),
        }


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--steps", type=int, default=100); parser.add_argument("--seed", type=int, default=42); parser.add_argument("--margin", type=float, default=0.5); args = parser.parse_args()
    results = []
    for weight in (0.1, 1.0, 5.0):
        model = train(weight, args.steps, args.seed, args.margin)
        metrics = evaluate(model); metrics["contrastive_weight"] = weight; results.append(metrics)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
