"""Prototype shared physical dictionary between cognition and decision."""

from __future__ import annotations

import argparse
import copy
import json

import torch
import torch.nn.functional as F

from physics_transfer.decision import PhysicsAwareDecision
from physics_transfer.shared_dictionary import SharedDictionaryCognitivePredictor, SharedPhysicsDictionary
from physics_transfer.multifactor_data import sample_multifactor_batch
from scripts.shared_dictionary_utils import dynamics_terms
from scripts.stage2_lowrank_gap_calibration import FACTORS, HELDOUT, _pair
from scripts.stage2_lowrank_semantic import rollout_loss
from scripts.stage3_frozen_task_counterfactual import train_task_baseline
from scripts.stage3_oracle_receiver_probe import conditional_teacher


def pretrain(model, steps, seed):
    torch.manual_seed(seed + 7000)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    for index in range(steps):
        batch = sample_multifactor_batch(16, 8, FACTORS)
        _, output, _, _, standard, absolute, gap, token, coefficient = dynamics_terms(model, batch, index)
        horizon = rollout_loss(model, batch, horizon=4)
        loss = standard + absolute + 50.0 * gap + 0.01 * token + 0.1 * coefficient
        loss = loss + 0.05 * output["basis_gram_error"].square().mean() + 0.01 * output["physics_residual"].square().mean() + 0.25 * horizon + 0.1 * output["dictionary_orthogonality"]
        optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
    return model


def joint_train(cognitive, task_model, steps, seed):
    torch.manual_seed(seed + 8000)
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
        factors_a = torch.tensor(factor_a).float().repeat(16, 1)
        factors_b = torch.tensor(factor_b).float().repeat(16, 1)
        prediction_a = decision(pair["state"], out_a["physics_latent"])
        prediction_b = decision(pair["state"], out_b["physics_latent"])
        target_a, target_b = conditional_teacher(pair["state"], factors_a), conditional_teacher(pair["state"], factors_b)
        policy = F.mse_loss(prediction_a, target_a) + F.mse_loss(prediction_b, target_b) + F.mse_loss(prediction_a - prediction_b, target_a - target_b)
        dictionary = F.normalize(cognitive.shared_dictionary.dictionary, dim=0)
        dictionary_orth = (dictionary.T @ dictionary - torch.eye(4)).square().mean()
        loss = standard + absolute + 50.0 * gap + 0.01 * token + 0.1 * coefficient
        loss = loss + 0.05 * output["basis_gram_error"].square().mean() + 0.01 * output["physics_residual"].square().mean() + 0.25 * horizon + 10.0 * policy + 0.1 * dictionary_orth
        optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(trainable, 5.0); optimizer.step()
    return cognitive, decision


def evaluate(cognitive, decision):
    cognitive.eval(); decision.eval()
    with torch.no_grad():
        pair = _pair(HELDOUT[0], HELDOUT[1], batch_size=256)
        out_a = cognitive(pair["history_a"], pair["state"], pair["action"])
        out_b = cognitive(pair["history_b"], pair["state"], pair["action"])
        factors_a = torch.tensor(HELDOUT[0]).float().repeat(256, 1)
        factors_b = torch.tensor(HELDOUT[1]).float().repeat(256, 1)
        target_a, target_b = conditional_teacher(pair["state"], factors_a), conditional_teacher(pair["state"], factors_b)
        prediction_a, prediction_b = decision(pair["state"], out_a["physics_latent"]), decision(pair["state"], out_b["physics_latent"])
        predicted_gap, true_gap = (prediction_a - prediction_b).abs().mean().item(), (target_a - target_b).abs().mean().item()
        return {"policy_absolute_mse": (F.mse_loss(prediction_a, target_a) + F.mse_loss(prediction_b, target_b)).item(), "predicted_action_gap": predicted_gap, "true_action_gap": true_gap, "action_gap_ratio": predicted_gap / max(true_gap, 1e-8), "one_step_loss": rollout_loss(cognitive, sample_multifactor_batch(128, 8, HELDOUT), 1).item(), "four_step_loss": rollout_loss(cognitive, sample_multifactor_batch(128, 8, HELDOUT), 4).item()}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--pretrain-steps", type=int, default=100); parser.add_argument("--task-steps", type=int, default=100); parser.add_argument("--joint-steps", type=int, default=100); parser.add_argument("--seed", type=int, default=42); args = parser.parse_args()
    shared = SharedPhysicsDictionary(4, 4)
    cognitive = SharedDictionaryCognitivePredictor(shared, state_dim=6, action_dim=1, history_steps=8, latent_dim=4, token_count=8, token_dim=8)
    cognitive = pretrain(cognitive, args.pretrain_steps, args.seed)
    task = train_task_baseline(args.task_steps, args.seed)
    cognitive, decision = joint_train(cognitive, task, args.joint_steps, args.seed)
    print(json.dumps(evaluate(cognitive, decision), ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
