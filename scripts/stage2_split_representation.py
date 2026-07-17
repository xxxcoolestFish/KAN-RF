"""Compare a single mixed representation with separated physics/memory branches."""

from __future__ import annotations

import argparse
import json

import torch

from physics_transfer.adaptive import AdaptiveCognitivePredictor
from physics_transfer.losses import cognitive_prediction_loss, token_consistency_loss
from physics_transfer.multifactor_data import sample_multifactor_batch
from physics_transfer.split_cognitive import SplitCognitivePredictor
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


def _train(model, split: bool, steps: int, seed: int):
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    for index in range(steps):
        batch = sample_multifactor_batch(32, 8, TRAIN_FACTORS)
        output = model(batch["history"], batch["state"], batch["action"])
        next_key = "next_state"
        prediction = cognitive_prediction_loss(output[next_key], batch["next_state"])
        factor = TRAIN_FACTORS[index % len(TRAIN_FACTORS)]
        pair_a = sample_multifactor_batch(32, 8, (factor,))
        pair_b = sample_multifactor_batch(32, 8, (factor,))
        out_a = model(pair_a["history"], pair_a["state"], pair_a["action"])
        out_b = model(pair_b["history"], pair_b["state"], pair_b["action"])
        if split:
            consistency = token_consistency_loss(
                out_a["physics_pooled"], out_b["physics_pooled"],
                out_a["physics_gates"], out_b["physics_gates"],
            )
            sparse = out_a["physics_gates"].mean()
        else:
            consistency = token_consistency_loss(
                out_a["pooled"], out_b["pooled"],
                out_a["gates"], out_b["gates"],
            )
            sparse = out_a["gates"].mean()
        loss = prediction + 0.01 * consistency + 1e-3 * sparse
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()


def _rollout(model, batch, horizon: int):
    history = batch["history"]
    predicted = batch["state"]
    true_state = batch["state"]
    losses = []
    for _ in range(horizon):
        action = torch.rand(predicted.shape[0], 1) * 2.0 - 1.0
        output = model(history, predicted, action)
        true_next = _true_step(true_state, action, batch["factors"])
        losses.append(cognitive_prediction_loss(output["next_state"], true_next))
        history = _shift_history(history, predicted, action)
        predicted, true_state = output["next_state"], true_next
    return torch.stack(losses).mean().item()


def _evaluate(model, split: bool):
    model.eval()
    with torch.no_grad():
        heldout = sample_multifactor_batch(192, 8, HELDOUT_FACTORS)
        one_step = _rollout(model, heldout, 1)
        four_step = _rollout(model, heldout, 4)
        same_a = sample_multifactor_batch(128, 8, (HELDOUT_FACTORS[0],))
        same_b = sample_multifactor_batch(128, 8, (HELDOUT_FACTORS[0],))
        out_a = model(same_a["history"], same_a["state"], same_a["action"])
        out_b = model(same_b["history"], same_b["state"], same_b["action"])
        other = sample_multifactor_batch(128, 8, (HELDOUT_FACTORS[1],))
        out_d = model(other["history"], other["state"], other["action"])
        if split:
            pooled_a, pooled_b, gates_a, gates_b = (
                out_a["physics_pooled"], out_b["physics_pooled"],
                out_a["physics_gates"], out_b["physics_gates"],
            )
            active = gates_a.gt(0.5).float().sum(-1).mean().item()
            memory_variation = (out_a["state_memory"].mean(0) -
                                out_b["state_memory"].mean(0)).norm().item()
        else:
            pooled_a, pooled_b, gates_a, gates_b = (
                out_a["pooled"], out_b["pooled"], out_a["gates"], out_b["gates"]
            )
            active = gates_a.gt(0.5).float().sum(-1).mean().item()
            memory_variation = 0.0
        return {
            "one_step_loss": one_step,
            "four_step_loss": four_step,
            "physics_consistency": token_consistency_loss(
                pooled_a, pooled_b, gates_a, gates_b
            ).item(),
            "physics_separation": (pooled_a.mean(0) - out_d[
                "physics_pooled" if split else "pooled"
            ].mean(0)).norm().item(),
            "effective_physics_slots": active,
            "state_memory_variation": memory_variation,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    results = []
    for split in (False, True):
        model = (SplitCognitivePredictor(6, 1, 8) if split else
                 AdaptiveCognitivePredictor(6, 1, 56, 8, 8, 24, 8))
        _train(model, split, args.steps, args.seed)
        metrics = _evaluate(model, split)
        metrics["architecture"] = "split" if split else "single"
        results.append(metrics)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
