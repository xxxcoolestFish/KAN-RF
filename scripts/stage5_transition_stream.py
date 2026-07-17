"""Validate the corrected atomic-transition cognitive interface."""

from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F

from physics_transfer.streaming_cognitive import StreamingCognitiveWorldModel
from physics_transfer.transition_data import sample_transition_sequence_batch
from scripts.stage2_lowrank_gap_calibration import FACTORS, HELDOUT


def train(steps: int, sequence_steps: int, seed: int):
    torch.manual_seed(seed)
    model = StreamingCognitiveWorldModel(latent_dim=16, hidden_dim=32, n_prototypes=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    for _ in range(steps):
        batch = sample_transition_sequence_batch(32, sequence_steps, FACTORS)
        output = model.forward_sequence(
            batch["state"], batch["action"], batch["next_state"]
        )
        loss = F.smooth_l1_loss(output["predictions"], batch["next_state"])
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
    return model


def evaluate(model, sequence_steps: int):
    model.eval()
    with torch.no_grad():
        batch = sample_transition_sequence_batch(256, sequence_steps, HELDOUT)
        output = model.forward_sequence(
            batch["state"], batch["action"], batch["next_state"]
        )
        errors = (output["predictions"] - batch["next_state"]).square().mean(dim=(0, 2))
        same_a = sample_transition_sequence_batch(256, sequence_steps, (HELDOUT[0],))
        same_b = sample_transition_sequence_batch(256, sequence_steps, (HELDOUT[0],))
        different = sample_transition_sequence_batch(256, sequence_steps, (HELDOUT[1],))
        code_a = model.physics_code(model.forward_sequence(same_a["state"], same_a["action"], same_a["next_state"])["latent"])
        code_b = model.physics_code(model.forward_sequence(same_b["state"], same_b["action"], same_b["next_state"])["latent"])
        code_d = model.physics_code(model.forward_sequence(different["state"], different["action"], different["next_state"])["latent"])
        same_distance = torch.linalg.vector_norm(code_a - code_b, dim=-1).mean().item()
        different_distance = torch.linalg.vector_norm(code_a - code_d, dim=-1).mean().item()
        return {
            "first_step_loss": errors[0].item(),
            "last_step_loss": errors[-1].item(),
            "same_context_code_distance": same_distance,
            "different_context_code_distance": different_distance,
            "context_separation_margin": different_distance - same_distance,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--sequence-steps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    model = train(args.steps, args.sequence_steps, args.seed)
    print(json.dumps(evaluate(model, args.sequence_steps), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
