"""Evaluate closed-loop multi-step rollout for calibrated low-rank transport."""

from __future__ import annotations

import argparse
import json

import torch

from physics_transfer.multifactor_data import sample_multifactor_batch
from physics_transfer.variants import step as variant_step
from scripts.stage2_lowrank_gap_calibration import HELDOUT, train


def _append_transition(history, state, action, state_dim=6, action_dim=1):
    sequence = history.view(history.shape[0], -1, state_dim + action_dim)
    transition = torch.cat([state, action], dim=-1).unsqueeze(1)
    return torch.cat([sequence[:, 1:], transition], dim=1).flatten(start_dim=1)


def evaluate(model, factor, horizon=4, batch_size=128):
    model.eval()
    with torch.no_grad():
        batch = sample_multifactor_batch(batch_size, 8, (factor,))
        factor_tensor = torch.tensor(factor).float().repeat(batch_size, 1)
        actions = [batch["action"]]
        actions.extend(torch.rand(batch_size, 1) * 2.0 - 1.0 for _ in range(horizon - 1))
        pred_history = batch["history"].clone()
        true_history = batch["history"].clone()
        pred_state = batch["state"].clone()
        true_state = batch["state"].clone()
        errors = []
        for action in actions:
            pred_out = model(pred_history, pred_state, action)
            pred_next = pred_out["next_state"]
            true_next = variant_step(
                true_state, action, factor_tensor[:, 0], factor_tensor[:, 1],
                factor_tensor[:, 2], factor_tensor[:, 3]
            )
            errors.append((pred_next - true_next).square().mean().item())
            pred_history = _append_transition(pred_history, pred_state, action)
            true_history = _append_transition(true_history, true_state, action)
            pred_state, true_state = pred_next, true_next
        return {
            "factor": factor,
            "rollout_step_mse": errors,
            "rollout_mean_mse": sum(errors) / len(errors),
            "rollout_final_mse": errors[-1],
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    model = train(50.0, args.steps, args.seed)
    print(json.dumps([
        evaluate(model, factor) for factor in HELDOUT
    ], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
