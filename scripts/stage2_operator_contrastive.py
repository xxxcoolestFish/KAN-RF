"""Contrast functional physics operators instead of assigning latent semantics.

The cognitive history encoder produces a coefficient vector, but its coordinate
system is not identifiable.  This experiment therefore compares the operator
induced by that vector on a shared probe set of states/actions.  Histories from
the same physical context should induce the same operator; histories from
different contexts should be separated by a margin.
"""

from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F

from physics_transfer.losses import cognitive_prediction_loss, token_consistency_loss
from physics_transfer.low_rank_split import LowRankSplitCognitivePredictor
from physics_transfer.multifactor_data import sample_multifactor_batch
from scripts.stage2_lowrank_gap_calibration import FACTORS, HELDOUT, _pair
from scripts.stage2_lowrank_semantic import rollout_loss
from scripts.shared_dictionary_utils import dynamics_terms


def _operator_signature(model, history, probe_state, probe_action):
    """Evaluate the history-conditioned residual operator on shared probes."""
    batch, probes = probe_state.shape[0], probe_state.shape[1]
    history_rep = history[:, None].expand(batch, probes, history.shape[-1]).reshape(
        batch * probes, history.shape[-1]
    )
    state_rep = probe_state.reshape(batch * probes, probe_state.shape[-1])
    action_rep = probe_action.reshape(batch * probes, probe_action.shape[-1])
    residual = model(history_rep, state_rep, action_rep)["physics_residual"]
    return residual.view(batch, probes * residual.shape[-1])


def _probe_batch(batch_size: int, probes: int):
    state = sample_multifactor_batch(batch_size * probes, 8, (FACTORS[0],))["state"]
    action = torch.rand(batch_size * probes, 1) * 2.0 - 1.0
    return state.view(batch_size, probes, -1), action.view(batch_size, probes, -1)


def train(operator_weight: float, steps: int, seed: int, margin: float = 0.5):
    torch.manual_seed(seed)
    model = LowRankSplitCognitivePredictor(
        physics_rank=4, residual_scale=0.1, state_dim=6,
        action_dim=1, history_steps=8, token_count=8, token_dim=8,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    for index in range(steps):
        batch = sample_multifactor_batch(16, 8, FACTORS)
        pair, output, out_a, out_b, standard, absolute, gap, token, _ = dynamics_terms(
            model, batch, index
        )
        factor_a = FACTORS[index % len(FACTORS)]
        same_a = sample_multifactor_batch(16, 8, (factor_a,))
        same_b = sample_multifactor_batch(16, 8, (factor_a,))
        diff_a = sample_multifactor_batch(16, 8, (factor_a,))
        diff_b = sample_multifactor_batch(16, 8, (FACTORS[(index + 1) % len(FACTORS)],))
        probe_state, probe_action = _probe_batch(16, 8)
        sig_same_a = F.normalize(
            _operator_signature(model, same_a["history"], probe_state, probe_action), dim=-1
        )
        sig_same_b = F.normalize(
            _operator_signature(model, same_b["history"], probe_state, probe_action), dim=-1
        )
        sig_diff_a = F.normalize(
            _operator_signature(model, diff_a["history"], probe_state, probe_action), dim=-1
        )
        sig_diff_b = F.normalize(
            _operator_signature(model, diff_b["history"], probe_state, probe_action), dim=-1
        )
        same_distance = torch.linalg.vector_norm(sig_same_a - sig_same_b, dim=-1)
        diff_distance = torch.linalg.vector_norm(sig_diff_a - sig_diff_b, dim=-1)
        same_operator_loss = same_distance.square().mean()
        diff_operator_loss = F.relu(margin - diff_distance).square().mean()
        horizon = rollout_loss(model, batch, horizon=4)
        loss = standard + absolute + 50.0 * gap + 0.01 * token
        loss = loss + operator_weight * (same_operator_loss + diff_operator_loss)
        loss = loss + 0.05 * output["basis_gram_error"].square().mean()
        loss = loss + 0.01 * output["physics_residual"].square().mean() + 0.25 * horizon
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
    return model


def evaluate(model):
    model.eval()
    with torch.no_grad():
        probe_state, probe_action = _probe_batch(256, 8)
        same_a = sample_multifactor_batch(256, 8, (HELDOUT[0],))
        same_b = sample_multifactor_batch(256, 8, (HELDOUT[0],))
        different = sample_multifactor_batch(256, 8, (HELDOUT[1],))
        sig_a = F.normalize(_operator_signature(model, same_a["history"], probe_state, probe_action), dim=-1)
        sig_b = F.normalize(_operator_signature(model, same_b["history"], probe_state, probe_action), dim=-1)
        sig_d = F.normalize(_operator_signature(model, different["history"], probe_state, probe_action), dim=-1)
        same_distance = torch.linalg.vector_norm(sig_a - sig_b, dim=-1).mean().item()
        different_distance = torch.linalg.vector_norm(sig_a - sig_d, dim=-1).mean().item()
        heldout = sample_multifactor_batch(128, 8, HELDOUT)
        return {
            "same_operator_distance": same_distance,
            "different_operator_distance": different_distance,
            "operator_separation_margin": different_distance - same_distance,
            "one_step_loss": rollout_loss(model, heldout, 1).item(),
            "four_step_loss": rollout_loss(model, heldout, 4).item(),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--margin", type=float, default=0.5)
    args = parser.parse_args()
    results = []
    for weight in (0.1, 1.0, 5.0):
        metrics = evaluate(train(weight, args.steps, args.seed, args.margin))
        metrics["operator_weight"] = weight
        results.append(metrics)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
