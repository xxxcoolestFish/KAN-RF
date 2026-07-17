"""Joint training with a counterfactual decision-difference objective."""

from __future__ import annotations

import argparse
import copy
import json

import torch
import torch.nn.functional as F

from physics_transfer.decision import PhysicsAwareDecision
from physics_transfer.interface import PhysicsTransport
from physics_transfer.losses import cognitive_prediction_loss, token_consistency_loss
from physics_transfer.multifactor_data import sample_multifactor_batch
from scripts.stage2_lowrank_gap_calibration import FACTORS, HELDOUT, _pair
from scripts.stage2_lowrank_semantic import rollout_loss, train as train_semantic
from scripts.stage3_oracle_receiver_probe import conditional_teacher, factor_code


def train_joint(cognitive_model, cf_weight: float, steps: int, seed: int):
    torch.manual_seed(seed + 4000)
    decision = PhysicsAwareDecision(6, 1, physics_slots=4, hidden_dim=24, n_prototypes=8)
    transport = PhysicsTransport(4, 4)
    trainable = list(cognitive_model.parameters()) + list(decision.parameters()) + list(transport.parameters())
    optimizer = torch.optim.Adam(trainable, lr=2e-3)
    for index in range(steps):
        batch = sample_multifactor_batch(16, 8, FACTORS)
        output = cognitive_model(batch["history"], batch["state"], batch["action"])
        standard = cognitive_prediction_loss(output["next_state"], batch["next_state"])
        factor_a = FACTORS[index % len(FACTORS)]
        factor_b = FACTORS[(index + 1) % len(FACTORS)]
        pair = _pair(factor_a, factor_b, batch_size=16)
        out_a = cognitive_model(pair["history_a"], pair["state"], pair["action"])
        out_b = cognitive_model(pair["history_b"], pair["state"], pair["action"])
        absolute = (
            cognitive_prediction_loss(out_a["next_state"], pair["target_a"])
            + cognitive_prediction_loss(out_b["next_state"], pair["target_b"])
        )
        dynamics_gap = F.mse_loss(
            torch.linalg.vector_norm(out_a["next_state"] - out_b["next_state"], dim=-1),
            torch.linalg.vector_norm(pair["target_a"] - pair["target_b"], dim=-1),
        )
        same_a = sample_multifactor_batch(16, 8, (factor_a,))
        same_b = sample_multifactor_batch(16, 8, (factor_a,))
        same_out_a = cognitive_model(same_a["history"], same_a["state"], same_a["action"])
        same_out_b = cognitive_model(same_b["history"], same_b["state"], same_b["action"])
        token_consistency = token_consistency_loss(
            same_out_a["physics_pooled"], same_out_b["physics_pooled"],
            same_out_a["physics_gates"], same_out_b["physics_gates"],
        )
        coefficient_consistency = F.mse_loss(
            same_out_a["physics_coefficients"], same_out_b["physics_coefficients"]
        )
        receiver_a = transport(out_a["physics_coefficients"])
        receiver_b = transport(out_b["physics_coefficients"])
        target_action_a = conditional_teacher(
            pair["state"], torch.tensor(factor_a).float().repeat(16, 1)
        )
        target_action_b = conditional_teacher(
            pair["state"], torch.tensor(factor_b).float().repeat(16, 1)
        )
        prediction_a = decision(pair["state"], receiver_a)
        prediction_b = decision(pair["state"], receiver_b)
        policy_absolute = F.mse_loss(prediction_a, target_action_a) + F.mse_loss(
            prediction_b, target_action_b
        )
        policy_gap = F.mse_loss(
            prediction_a - prediction_b, target_action_a - target_action_b
        )
        horizon = rollout_loss(cognitive_model, batch, horizon=4)
        loss = (
            standard + absolute + 50.0 * dynamics_gap + 0.01 * token_consistency
            + 0.1 * coefficient_consistency + 0.05 * output["basis_gram_error"].square().mean()
            + 0.01 * output["physics_residual"].square().mean() + 0.25 * horizon
            + cf_weight * (policy_absolute + policy_gap)
            + 1e-3 * transport.alignment_loss()
        )
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 5.0)
        optimizer.step()
    return cognitive_model, decision, transport


def evaluate(cognitive_model, decision, transport):
    cognitive_model.eval(); decision.eval(); transport.eval()
    with torch.no_grad():
        pair = _pair(HELDOUT[0], HELDOUT[1], batch_size=256)
        out_a = cognitive_model(pair["history_a"], pair["state"], pair["action"])
        out_b = cognitive_model(pair["history_b"], pair["state"], pair["action"])
        factor_a = torch.tensor(HELDOUT[0]).float().repeat(256, 1)
        factor_b = torch.tensor(HELDOUT[1]).float().repeat(256, 1)
        target_a = conditional_teacher(pair["state"], factor_a)
        target_b = conditional_teacher(pair["state"], factor_b)
        prediction_a = decision(pair["state"], transport(out_a["physics_coefficients"]))
        prediction_b = decision(pair["state"], transport(out_b["physics_coefficients"]))
        predicted_gap = (prediction_a - prediction_b).abs().mean().item()
        true_gap = (target_a - target_b).abs().mean().item()
        return {
            "policy_absolute_mse": (
                F.mse_loss(prediction_a, target_a) + F.mse_loss(prediction_b, target_b)
            ).item(),
            "predicted_action_gap": predicted_gap,
            "true_action_gap": true_gap,
            "action_gap_ratio": predicted_gap / max(true_gap, 1e-8),
            "one_step_loss": rollout_loss(cognitive_model, sample_multifactor_batch(128, 8, HELDOUT), 1).item(),
            "four_step_loss": rollout_loss(cognitive_model, sample_multifactor_batch(128, 8, HELDOUT), 4).item(),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain-steps", type=int, default=100)
    parser.add_argument("--joint-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    base = train_semantic(0.1, args.pretrain_steps, args.seed)
    results = []
    for weight in (1.0, 10.0):
        model, decision, transport = train_joint(
            copy.deepcopy(base), weight, args.joint_steps, args.seed
        )
        results.append({"counterfactual_policy_weight": weight, "metrics": evaluate(model, decision, transport)})
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
