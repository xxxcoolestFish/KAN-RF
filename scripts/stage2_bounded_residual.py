"""Tune the bounded physical residual scale with counterfactual supervision."""

from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F

from physics_transfer.bounded_split import BoundedSplitCognitivePredictor
from physics_transfer.losses import cognitive_prediction_loss, token_consistency_loss
from physics_transfer.multifactor_data import sample_multifactor_batch
from physics_transfer.variants import step as variant_step


FACTORS = (
    (7.35, 0.00, 0.80, 0.80),
    (7.35, 0.08, 1.20, 1.20),
    (12.25, 0.00, 1.20, 1.20),
    (12.25, 0.08, 0.80, 0.80),
    (14.70, 0.04, 1.00, 1.00),
)
HELDOUT = (
    (9.80, 0.04, 1.10, 0.90),
    (13.475, 0.06, 0.90, 1.10),
)


def _true_step(state, action, factors):
    return variant_step(state, action, factors[:, 0], factors[:, 1],
                        factors[:, 2], factors[:, 3])


def _pair(factor_a, factor_b, batch_size=32):
    a = sample_multifactor_batch(batch_size, 8, (factor_a,))
    b = sample_multifactor_batch(batch_size, 8, (factor_b,))
    state = a["state"]
    action = torch.rand(batch_size, 1) * 2.0 - 1.0
    fa = torch.tensor(factor_a).float().repeat(batch_size, 1)
    fb = torch.tensor(factor_b).float().repeat(batch_size, 1)
    return {
        "history_a": a["history"], "history_b": b["history"],
        "state": state, "action": action,
        "target_a": _true_step(state, action, fa),
        "target_b": _true_step(state, action, fb),
    }


def train(scale: float, steps: int, seed: int):
    torch.manual_seed(seed)
    model = BoundedSplitCognitivePredictor(
        residual_scale=scale, state_dim=6, action_dim=1, history_steps=8,
        token_count=8, token_dim=8,
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
        difference = F.smooth_l1_loss(
            out_a["next_state"] - out_b["next_state"],
            pair["target_a"] - pair["target_b"],
        )
        same_a = sample_multifactor_batch(32, 8, (factor_a,))
        same_b = sample_multifactor_batch(32, 8, (factor_a,))
        same_out_a = model(same_a["history"], same_a["state"], same_a["action"])
        same_out_b = model(same_b["history"], same_b["state"], same_b["action"])
        consistency = token_consistency_loss(
            same_out_a["physics_pooled"], same_out_b["physics_pooled"],
            same_out_a["physics_gates"], same_out_b["physics_gates"],
        )
        energy = out_a["physics_residual"].square().mean() + out_b[
            "physics_residual"
        ].square().mean()
        loss = standard + absolute + difference + 0.01 * consistency + 0.1 * energy
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
        same_a = sample_multifactor_batch(128, 8, (HELDOUT[0],))
        same_b = sample_multifactor_batch(128, 8, (HELDOUT[0],))
        same_out_a = model(same_a["history"], same_a["state"], same_a["action"])
        same_out_b = model(same_b["history"], same_b["state"], same_b["action"])
        return {
            "counterfactual_absolute_loss": (
                cognitive_prediction_loss(out_a["next_state"], pair["target_a"])
                + cognitive_prediction_loss(out_b["next_state"], pair["target_b"])
            ).item(),
            "counterfactual_predicted_gap": torch.linalg.vector_norm(
                out_a["next_state"] - out_b["next_state"], dim=-1
            ).mean().item(),
            "counterfactual_true_gap": torch.linalg.vector_norm(
                pair["target_a"] - pair["target_b"], dim=-1
            ).mean().item(),
            "physics_consistency": token_consistency_loss(
                same_out_a["physics_pooled"], same_out_b["physics_pooled"],
                same_out_a["physics_gates"], same_out_b["physics_gates"],
            ).item(),
            "mean_gate": same_out_a["physics_gates"].mean().item(),
            "mean_residual_norm": torch.linalg.vector_norm(
                out_a["physics_residual"], dim=-1
            ).mean().item(),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    results = []
    for scale in (0.05, 0.1, 0.2):
        model = train(scale, args.steps, args.seed)
        metrics = evaluate(model)
        metrics["residual_scale"] = scale
        results.append(metrics)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
