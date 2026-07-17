"""Force the split cognitive model to use physics tokens for factor changes."""

from __future__ import annotations

import argparse
import json

import torch

from physics_transfer.losses import cognitive_prediction_loss, token_consistency_loss
from physics_transfer.multifactor_data import sample_multifactor_batch
from physics_transfer.split_cognitive_v2 import SplitCognitivePredictorV2
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
    return torch.cat([history[:, 7:], torch.cat([state, action], dim=-1)], dim=-1)


def _true_step(state, action, factors):
    return variant_step(state, action, factors[:, 0], factors[:, 1],
                        factors[:, 2], factors[:, 3])


def _counterfactual_pair(factor_a, factor_b, batch_size=32):
    context_a = sample_multifactor_batch(batch_size, 8, (factor_a,))
    context_b = sample_multifactor_batch(batch_size, 8, (factor_b,))
    state = context_a["state"]
    action = torch.rand(batch_size, 1) * 2.0 - 1.0
    factors_a = torch.tensor(factor_a).float().repeat(batch_size, 1)
    factors_b = torch.tensor(factor_b).float().repeat(batch_size, 1)
    return {
        "history_a": context_a["history"],
        "history_b": context_b["history"],
        "state": state,
        "action": action,
        "target_a": _true_step(state, action, factors_a),
        "target_b": _true_step(state, action, factors_b),
    }


def train(steps: int, seed: int):
    torch.manual_seed(seed)
    model = SplitCognitivePredictorV2(6, 1, 8, token_count=8, token_dim=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    for index in range(steps):
        batch = sample_multifactor_batch(32, 8, TRAIN_FACTORS)
        output = model(batch["history"], batch["state"], batch["action"])
        standard = cognitive_prediction_loss(output["next_state"], batch["next_state"])

        factor_a = TRAIN_FACTORS[index % len(TRAIN_FACTORS)]
        factor_b = TRAIN_FACTORS[(index + 1) % len(TRAIN_FACTORS)]
        pair = _counterfactual_pair(factor_a, factor_b)
        output_a = model(pair["history_a"], pair["state"], pair["action"])
        output_b = model(pair["history_b"], pair["state"], pair["action"])
        counterfactual = (
            cognitive_prediction_loss(output_a["next_state"], pair["target_a"])
            + cognitive_prediction_loss(output_b["next_state"], pair["target_b"])
        )

        same_factor = sample_multifactor_batch(32, 8, (factor_a,))
        same_other = sample_multifactor_batch(32, 8, (factor_a,))
        same_a = model(same_factor["history"], same_factor["state"], same_factor["action"])
        same_b = model(same_other["history"], same_other["state"], same_other["action"])
        consistency = token_consistency_loss(
            same_a["physics_pooled"], same_b["physics_pooled"],
            same_a["physics_gates"], same_b["physics_gates"],
        )
        loss = standard + counterfactual + 0.01 * consistency + 1e-3 * same_a[
            "physics_gates"
        ].mean()
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
    return model


def _rollout(model, batch, horizon):
    history, predicted, true_state = batch["history"], batch["state"], batch["state"]
    losses = []
    for _ in range(horizon):
        action = torch.rand(predicted.shape[0], 1) * 2.0 - 1.0
        output = model(history, predicted, action)
        true_next = _true_step(true_state, action, batch["factors"])
        losses.append(cognitive_prediction_loss(output["next_state"], true_next))
        history = _shift_history(history, predicted, action)
        predicted, true_state = output["next_state"], true_next
    return torch.stack(losses).mean().item()


def evaluate(model):
    model.eval()
    with torch.no_grad():
        heldout = sample_multifactor_batch(192, 8, HELDOUT_FACTORS)
        same_a = sample_multifactor_batch(128, 8, (HELDOUT_FACTORS[0],))
        same_b = sample_multifactor_batch(128, 8, (HELDOUT_FACTORS[0],))
        out_a = model(same_a["history"], same_a["state"], same_a["action"])
        out_b = model(same_b["history"], same_b["state"], same_b["action"])
        different = sample_multifactor_batch(128, 8, (HELDOUT_FACTORS[1],))
        out_d = model(different["history"], different["state"], different["action"])

        pair = _counterfactual_pair(HELDOUT_FACTORS[0], HELDOUT_FACTORS[1], 128)
        cf_a = model(pair["history_a"], pair["state"], pair["action"])
        cf_b = model(pair["history_b"], pair["state"], pair["action"])
        predicted_gap = (cf_a["next_state"] - cf_b["next_state"]).norm(dim=-1).mean().item()
        true_gap = (pair["target_a"] - pair["target_b"]).norm(dim=-1).mean().item()
        return {
            "one_step_loss": _rollout(model, heldout, 1),
            "four_step_loss": _rollout(model, heldout, 4),
            "physics_consistency": token_consistency_loss(
                out_a["physics_pooled"], out_b["physics_pooled"],
                out_a["physics_gates"], out_b["physics_gates"],
            ).item(),
            "physics_separation": (
                out_a["physics_pooled"].mean(0) - out_d["physics_pooled"].mean(0)
            ).norm().item(),
            "counterfactual_predicted_gap": predicted_gap,
            "counterfactual_true_gap": true_gap,
            "effective_physics_slots": out_a["physics_gates"].gt(0.5).float().sum(-1).mean().item(),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    model = train(args.steps, args.seed)
    print(json.dumps(evaluate(model), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
