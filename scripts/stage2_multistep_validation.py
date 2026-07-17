"""Stage 2: test whether the cognitive representation supports multi-step prediction."""

from __future__ import annotations

import argparse
import json

import torch

from physics_transfer.adaptive import AdaptiveCognitivePredictor
from physics_transfer.losses import cognitive_prediction_loss, token_consistency_loss
from physics_transfer.multifactor_data import sample_multifactor_batch
from physics_transfer.variants import step as variant_step


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


def _shift_history(history, state, action):
    transition = torch.cat([state, action], dim=-1)
    return torch.cat([history[:, 7:], transition], dim=-1)


def _true_step(state, action, factors):
    return variant_step(
        state, action, factors[:, 0], factors[:, 1],
        factors[:, 2], factors[:, 3],
    )


def rollout_loss(model, batch, horizon: int):
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


def train_model(multistep: bool, steps: int, seed: int):
    torch.manual_seed(seed)
    model = AdaptiveCognitivePredictor(
        6, 1, 8 * 7, token_count=8, token_dim=8,
        hidden_dim=24, n_prototypes=8,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    for index in range(steps):
        batch = sample_multifactor_batch(32, 8, TRAIN_FACTORS)
        if multistep:
            prediction_loss = rollout_loss(model, batch, horizon=4)
        else:
            output = model(batch["history"], batch["state"], batch["action"])
            prediction_loss = cognitive_prediction_loss(
                output["next_state"], batch["next_state"]
            )

        factor = TRAIN_FACTORS[index % len(TRAIN_FACTORS)]
        pair_a = sample_multifactor_batch(32, 8, (factor,))
        pair_b = sample_multifactor_batch(32, 8, (factor,))
        output_a = model(pair_a["history"], pair_a["state"], pair_a["action"])
        output_b = model(pair_b["history"], pair_b["state"], pair_b["action"])
        consistency = token_consistency_loss(
            output_a["pooled"], output_b["pooled"],
            output_a["gates"], output_b["gates"],
        )
        loss = prediction_loss + 0.01 * consistency + 1e-3 * output_a["gates"].mean()
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
    return model


def evaluate(model):
    model.eval()
    with torch.no_grad():
        heldout = sample_multifactor_batch(192, 8, HELDOUT_FACTORS)
        one_step = rollout_loss(model, heldout, horizon=1).item()
        four_step = rollout_loss(model, heldout, horizon=4).item()

        same_a = sample_multifactor_batch(128, 8, (HELDOUT_FACTORS[0],))
        same_b = sample_multifactor_batch(128, 8, (HELDOUT_FACTORS[0],))
        out_a = model(same_a["history"], same_a["state"], same_a["action"])
        out_b = model(same_b["history"], same_b["state"], same_b["action"])
        different = sample_multifactor_batch(128, 8, (HELDOUT_FACTORS[1],))
        out_d = model(different["history"], different["state"], different["action"])
        consistency = token_consistency_loss(
            out_a["pooled"], out_b["pooled"], out_a["gates"], out_b["gates"]
        ).item()
        separation = (
            out_a["pooled"].mean(0) - out_d["pooled"].mean(0)
        ).norm().item()
        return {
            "one_step_loss": one_step,
            "four_step_loss": four_step,
            "same_factor_consistency": consistency,
            "different_factor_distance": separation,
            "mean_effective_slots": out_a["gates"].gt(0.5).float().sum(-1).mean().item(),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    results = []
    for multistep in (False, True):
        model = train_model(multistep, args.steps, args.seed)
        metrics = evaluate(model)
        metrics["training"] = "multi_step" if multistep else "one_step"
        results.append(metrics)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
