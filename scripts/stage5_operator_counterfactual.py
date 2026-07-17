"""Add counterfactual operator-difference supervision to streaming cognition."""

from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F

from physics_transfer.multifactor_data import _random_states
from physics_transfer.streaming_cognitive import StreamingCognitiveWorldModel
from physics_transfer.transition_data import sample_transition_sequence_batch
from physics_transfer.variants import step
from scripts.stage2_lowrank_gap_calibration import FACTORS, HELDOUT
from scripts.stage5_operator_interface import operator_signature, probe_batch, final_latent


def shared_counterfactual(factor_a, factor_b, batch_size):
    state = _random_states(batch_size)
    action = torch.rand(batch_size, 1) * 2.0 - 1.0
    fa = torch.tensor(factor_a).float().repeat(batch_size, 1)
    fb = torch.tensor(factor_b).float().repeat(batch_size, 1)
    target_a = step(state, action, fa[:, 0], fa[:, 1], fa[:, 2], fa[:, 3])
    target_b = step(state, action, fb[:, 0], fb[:, 1], fb[:, 2], fb[:, 3])
    return state, action, target_a, target_b


def train(operator_weight: float, counterfactual_weight: float, steps: int,
          sequence_steps: int, seed: int, margin: float = 0.5):
    torch.manual_seed(seed)
    model = StreamingCognitiveWorldModel(latent_dim=16, hidden_dim=32, n_prototypes=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    for index in range(steps):
        sequence = sample_transition_sequence_batch(24, sequence_steps, FACTORS)
        output = model.forward_sequence(sequence["state"], sequence["action"], sequence["next_state"])
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
            F.normalize(signature_diff_a, dim=-1) - F.normalize(signature_diff_b, dim=-1), dim=-1
        )
        diff_loss = F.relu(margin - diff_distance).square().mean()
        probe_state, probe_action, target_a, target_b = shared_counterfactual(factor_a, factor_b, 12)
        pred_a = model.predict_next(probe_state, probe_action, latent_diff_a)
        pred_b = model.predict_next(probe_state, probe_action, latent_diff_b)
        counterfactual = F.smooth_l1_loss(pred_a - pred_b, target_a - target_b)
        loss = dynamics + operator_weight * (same_loss + diff_loss) + counterfactual_weight * counterfactual
        optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
    return model


def evaluate(model, sequence_steps: int):
    model.eval()
    with torch.no_grad():
        sequence = sample_transition_sequence_batch(256, sequence_steps, HELDOUT)
        output = model.forward_sequence(sequence["state"], sequence["action"], sequence["next_state"])
        errors = (output["predictions"] - sequence["next_state"]).square().mean(dim=(0, 2))
        same_a = sample_transition_sequence_batch(256, sequence_steps, (HELDOUT[0],))
        same_b = sample_transition_sequence_batch(256, sequence_steps, (HELDOUT[0],))
        diff = sample_transition_sequence_batch(256, sequence_steps, (HELDOUT[1],))
        latent_a, latent_b = final_latent(model, same_a), final_latent(model, same_b)
        latent_d = final_latent(model, diff)
        probes_state, probes_action = probe_batch(256, 8)
        signature_a = operator_signature(model, latent_a, probes_state, probes_action)
        signature_b = operator_signature(model, latent_b, probes_state, probes_action)
        signature_d = operator_signature(model, latent_d, probes_state, probes_action)
        state, action, target_a, target_b = shared_counterfactual(HELDOUT[0], HELDOUT[1], 256)
        pred_a = model.predict_next(state, action, latent_a)
        pred_b = model.predict_next(state, action, latent_d)
        return {
            "first_step_loss": errors[0].item(), "last_step_loss": errors[-1].item(),
            "raw_separation_margin": (torch.linalg.vector_norm(latent_a - latent_d, dim=-1).mean() - torch.linalg.vector_norm(latent_a - latent_b, dim=-1).mean()).item(),
            "operator_same_distance": torch.linalg.vector_norm(signature_a - signature_b, dim=-1).mean().item(),
            "operator_different_distance": torch.linalg.vector_norm(signature_a - signature_d, dim=-1).mean().item(),
            "operator_separation_margin": (torch.linalg.vector_norm(signature_a - signature_d, dim=-1).mean() - torch.linalg.vector_norm(signature_a - signature_b, dim=-1).mean()).item(),
            "counterfactual_difference_loss": F.mse_loss(pred_a - pred_b, target_a - target_b).item(),
        }


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--steps", type=int, default=150); parser.add_argument("--sequence-steps", type=int, default=16); parser.add_argument("--counterfactual-weight", type=float, default=20.0); parser.add_argument("--seed", type=int, default=42); args = parser.parse_args()
    results = []
    for weight in (0.0, 0.1, 1.0):
        metrics = evaluate(train(weight, args.counterfactual_weight, args.steps, args.sequence_steps, args.seed), args.sequence_steps)
        metrics["operator_weight"] = weight; results.append(metrics)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
