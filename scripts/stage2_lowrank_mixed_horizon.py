"""Train low-rank physics transport with a mixed one-/multi-step objective."""

from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F

from physics_transfer.losses import cognitive_prediction_loss, token_consistency_loss
from physics_transfer.low_rank_split import LowRankSplitCognitivePredictor
from physics_transfer.multifactor_data import sample_multifactor_batch
from physics_transfer.variants import step as variant_step
from scripts.stage2_lowrank_gap_calibration import FACTORS, HELDOUT, _pair


def _shift_history(history, state, action):
    sequence = history.view(history.shape[0], 8, 7)
    transition = torch.cat([state, action], dim=-1).unsqueeze(1)
    return torch.cat([sequence[:, 1:], transition], dim=1).flatten(start_dim=1)


def _true_step(state, action, factors):
    return variant_step(
        state, action, factors[:, 0], factors[:, 1],
        factors[:, 2], factors[:, 3]
    )


def rollout_loss(model, batch, horizon=4):
    history = batch["history"]
    predicted_state = batch["state"]
    true_state = batch["state"]
    losses = []
    for _ in range(horizon):
        action = torch.rand(predicted_state.shape[0], 1) * 2.0 - 1.0
        output = model(history, predicted_state, action)
        true_next = _true_step(true_state, action, batch["factors"])
        losses.append(cognitive_prediction_loss(output["next_state"], true_next))
        history = _shift_history(history, predicted_state, action)
        predicted_state = output["next_state"]
        true_state = true_next
    return torch.stack(losses).mean()


def train(horizon_weight: float, steps: int, seed: int):
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
        absolute = (
            cognitive_prediction_loss(out_a["next_state"], pair["target_a"])
            + cognitive_prediction_loss(out_b["next_state"], pair["target_b"])
        )
        gap_loss = F.mse_loss(
            torch.linalg.vector_norm(
                out_a["next_state"] - out_b["next_state"], dim=-1
            ),
            torch.linalg.vector_norm(
                pair["target_a"] - pair["target_b"], dim=-1
            ),
        )
        same_a = sample_multifactor_batch(16, 8, (factor_a,))
        same_b = sample_multifactor_batch(16, 8, (factor_a,))
        same_out_a = model(same_a["history"], same_a["state"], same_a["action"])
        same_out_b = model(same_b["history"], same_b["state"], same_b["action"])
        consistency = token_consistency_loss(
            same_out_a["physics_pooled"], same_out_b["physics_pooled"],
            same_out_a["physics_gates"], same_out_b["physics_gates"],
        )
        horizon = rollout_loss(model, batch, horizon=4)
        loss = (
            standard + absolute + 50.0 * gap_loss + 0.01 * consistency
            + 0.05 * output["basis_gram_error"].square().mean()
            + 0.01 * output["physics_residual"].square().mean()
            + horizon_weight * horizon
        )
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
    return model


def evaluate(model):
    model.eval()
    with torch.no_grad():
        heldout = sample_multifactor_batch(128, 8, HELDOUT)
        return {
            "one_step_loss": rollout_loss(model, heldout, 1).item(),
            "four_step_loss": rollout_loss(model, heldout, 4).item(),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    results = []
    for weight in (0.0, 0.1, 0.25):
        metrics = evaluate(train(weight, args.steps, args.seed))
        metrics["horizon_weight"] = weight
        results.append(metrics)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
