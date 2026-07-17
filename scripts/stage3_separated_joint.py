"""Jointly align cognitive coefficients with a separated physics decision branch."""

from __future__ import annotations

import argparse
import copy
import json

import torch
import torch.nn.functional as F

from physics_transfer.multifactor_data import sample_multifactor_batch
from physics_transfer.separated_decision import SeparatedPhysicsDecision
from scripts.shared_dictionary_utils import dynamics_terms
from scripts.stage2_lowrank_gap_calibration import FACTORS, HELDOUT, _pair
from scripts.stage2_lowrank_semantic import rollout_loss, train as train_cognitive
from scripts.stage3_oracle_receiver_probe import conditional_teacher


def train_task_baseline(steps: int, seed: int):
    torch.manual_seed(seed + 10000)
    model = SeparatedPhysicsDecision(6, 1, 4, hidden_dim=24, n_prototypes=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    for _ in range(steps):
        batch = sample_multifactor_batch(64, 8, FACTORS)
        target = conditional_teacher(batch["state"], batch["factors"])
        output = model(batch["state"], torch.zeros(batch["state"].shape[0], 4))
        loss = F.mse_loss(output["action"], target)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    return model


def joint_train(cognitive, task_model, steps: int, seed: int):
    torch.manual_seed(seed + 11000)
    decision = copy.deepcopy(task_model)
    for parameter in decision.task_trunk.parameters(): parameter.requires_grad = False
    for parameter in decision.task_head.parameters(): parameter.requires_grad = False
    trainable = list(cognitive.parameters()) + [decision.physics_basis]
    optimizer = torch.optim.Adam(trainable, lr=2e-3)
    for index in range(steps):
        batch = sample_multifactor_batch(16, 8, FACTORS)
        pair, output, out_a, out_b, standard, absolute, gap, token, coefficient = dynamics_terms(cognitive, batch, index)
        horizon = rollout_loss(cognitive, batch, horizon=4)
        factor_a, factor_b = FACTORS[index % len(FACTORS)], FACTORS[(index + 1) % len(FACTORS)]
        fa = torch.tensor(factor_a).float().repeat(16, 1)
        fb = torch.tensor(factor_b).float().repeat(16, 1)
        pred_a = decision(pair["state"], out_a["physics_coefficients"])["action"]
        pred_b = decision(pair["state"], out_b["physics_coefficients"])["action"]
        target_a = conditional_teacher(pair["state"], fa)
        target_b = conditional_teacher(pair["state"], fb)
        policy = F.mse_loss(pred_a, target_a) + F.mse_loss(pred_b, target_b)
        policy = policy + F.mse_loss(pred_a - pred_b, target_a - target_b)
        loss = standard + absolute + 50.0 * gap + 0.01 * token + 0.1 * coefficient
        loss = loss + 0.05 * output["basis_gram_error"].square().mean() + 0.01 * output["physics_residual"].square().mean()
        loss = loss + 0.25 * horizon + 10.0 * policy + 0.05 * decision.separation_loss() + 0.05 * decision.physics_basis_orthogonality_loss()
        optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(trainable, 5.0); optimizer.step()
    return cognitive, decision


def evaluate(cognitive, decision):
    cognitive.eval(); decision.eval()
    with torch.no_grad():
        pair = _pair(HELDOUT[0], HELDOUT[1], batch_size=256)
        out_a = cognitive(pair["history_a"], pair["state"], pair["action"])
        out_b = cognitive(pair["history_b"], pair["state"], pair["action"])
        fa = torch.tensor(HELDOUT[0]).float().repeat(256, 1)
        fb = torch.tensor(HELDOUT[1]).float().repeat(256, 1)
        ta, tb = conditional_teacher(pair["state"], fa), conditional_teacher(pair["state"], fb)
        pa = decision(pair["state"], out_a["physics_coefficients"])["action"]
        pb = decision(pair["state"], out_b["physics_coefficients"])["action"]
        predicted_gap, true_gap = (pa - pb).abs().mean().item(), (ta - tb).abs().mean().item()
        return {"policy_absolute_mse": (F.mse_loss(pa, ta) + F.mse_loss(pb, tb)).item(), "predicted_action_gap": predicted_gap, "true_action_gap": true_gap, "action_gap_ratio": predicted_gap / max(true_gap, 1e-8), "one_step_loss": rollout_loss(cognitive, sample_multifactor_batch(128, 8, HELDOUT), 1).item(), "four_step_loss": rollout_loss(cognitive, sample_multifactor_batch(128, 8, HELDOUT), 4).item()}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--cognitive-steps", type=int, default=100); parser.add_argument("--task-steps", type=int, default=100); parser.add_argument("--joint-steps", type=int, default=100); parser.add_argument("--seed", type=int, default=42); args = parser.parse_args()
    cognitive = train_cognitive(0.1, args.cognitive_steps, args.seed)
    task = train_task_baseline(args.task_steps, args.seed)
    cognitive, decision = joint_train(cognitive, task, args.joint_steps, args.seed)
    print(json.dumps(evaluate(cognitive, decision), ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
