"""Train and evaluate operator-level equivalence for streaming cognition."""

from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F

from physics_transfer.streaming_cognitive import StreamingCognitiveWorldModel
from physics_transfer.transition_data import sample_transition_batch, sample_transition_sequence_batch
from scripts.stage2_lowrank_gap_calibration import FACTORS, HELDOUT


def probe_batch(batch_size: int, probe_count: int):
    probe = sample_transition_batch(batch_size * probe_count, (FACTORS[0],))
    state = probe["state"].view(batch_size, probe_count, -1)
    action = probe["action"].view(batch_size, probe_count, -1)
    return state, action


def operator_signature(model, latent, probe_state, probe_action):
    """Return the latent-induced response, subtracting the zero-context base."""
    batch, probes = probe_state.shape[:2]
    state = probe_state.reshape(batch * probes, -1)
    action = probe_action.reshape(batch * probes, -1)
    zero = model.initial_latent(batch * probes, state.device)
    latent = latent[:, None, :].expand(batch, probes, -1).reshape(batch * probes, -1)
    response = model.predict_next(state, action, latent)
    base = model.predict_next(state, action, zero)
    return (response - base).view(batch, -1)


def final_latent(model, sequence):
    return model.forward_sequence(
        sequence["state"], sequence["action"], sequence["next_state"]
    )["latent"]


def train(operator_weight: float, steps: int, sequence_steps: int, seed: int,
          margin: float = 0.5):
    torch.manual_seed(seed)
    model = StreamingCognitiveWorldModel(latent_dim=16, hidden_dim=32, n_prototypes=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    for index in range(steps):
        sequence = sample_transition_sequence_batch(24, sequence_steps, FACTORS)
        output = model.forward_sequence(
            sequence["state"], sequence["action"], sequence["next_state"]
        )
        dynamics = F.smooth_l1_loss(output["predictions"], sequence["next_state"])
        factor_a = FACTORS[index % len(FACTORS)]
        factor_b = FACTORS[(index + 1) % len(FACTORS)]
        same_a = sample_transition_sequence_batch(12, sequence_steps, (factor_a,))
        same_b = sample_transition_sequence_batch(12, sequence_steps, (factor_a,))
        diff_a = sample_transition_sequence_batch(12, sequence_steps, (factor_a,))
        diff_b = sample_transition_sequence_batch(12, sequence_steps, (factor_b,))
        latent_same_a = final_latent(model, same_a)
        latent_same_b = final_latent(model, same_b)
        latent_diff_a = final_latent(model, diff_a)
        latent_diff_b = final_latent(model, diff_b)
        probes_state, probes_action = probe_batch(12, 8)
        signature_same_a = operator_signature(model, latent_same_a, probes_state, probes_action)
        signature_same_b = operator_signature(model, latent_same_b, probes_state, probes_action)
        signature_diff_a = operator_signature(model, latent_diff_a, probes_state, probes_action)
        signature_diff_b = operator_signature(model, latent_diff_b, probes_state, probes_action)
        same_loss = F.mse_loss(signature_same_a, signature_same_b)
        diff_distance = torch.linalg.vector_norm(
            F.normalize(signature_diff_a, dim=-1)
            - F.normalize(signature_diff_b, dim=-1), dim=-1
        )
        diff_loss = F.relu(margin - diff_distance).square().mean()
        loss = dynamics + operator_weight * (same_loss + diff_loss)
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
    return model


def evaluate(model, sequence_steps: int):
    model.eval()
    with torch.no_grad():
        sequence = sample_transition_sequence_batch(256, sequence_steps, HELDOUT)
        output = model.forward_sequence(
            sequence["state"], sequence["action"], sequence["next_state"]
        )
        errors = (output["predictions"] - sequence["next_state"]).square().mean(dim=(0, 2))
        same_a = sample_transition_sequence_batch(256, sequence_steps, (HELDOUT[0],))
        same_b = sample_transition_sequence_batch(256, sequence_steps, (HELDOUT[0],))
        diff = sample_transition_sequence_batch(256, sequence_steps, (HELDOUT[1],))
        latent_a = final_latent(model, same_a)
        latent_b = final_latent(model, same_b)
        latent_d = final_latent(model, diff)
        probes_state, probes_action = probe_batch(256, 8)
        signature_a = operator_signature(model, latent_a, probes_state, probes_action)
        signature_b = operator_signature(model, latent_b, probes_state, probes_action)
        signature_d = operator_signature(model, latent_d, probes_state, probes_action)
        raw_same = torch.linalg.vector_norm(latent_a - latent_b, dim=-1).mean().item()
        raw_diff = torch.linalg.vector_norm(latent_a - latent_d, dim=-1).mean().item()
        op_same = torch.linalg.vector_norm(signature_a - signature_b, dim=-1).mean().item()
        op_diff = torch.linalg.vector_norm(signature_a - signature_d, dim=-1).mean().item()
        return {
            "first_step_loss": errors[0].item(),
            "last_step_loss": errors[-1].item(),
            "raw_same_distance": raw_same,
            "raw_different_distance": raw_diff,
            "raw_separation_margin": raw_diff - raw_same,
            "operator_same_distance": op_same,
            "operator_different_distance": op_diff,
            "operator_separation_margin": op_diff - op_same,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--sequence-steps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    results = []
    for weight in (0.0, 0.1, 1.0):
        metrics = evaluate(train(weight, args.steps, args.sequence_steps, args.seed), args.sequence_steps)
        metrics["operator_weight"] = weight
        results.append(metrics)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
