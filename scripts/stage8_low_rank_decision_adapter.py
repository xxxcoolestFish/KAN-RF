"""Controlled comparison of frozen, full, and low-rank decision adaptation."""

from __future__ import annotations

import argparse
import copy
import json

import torch
import torch.nn.functional as F

from physics_transfer.low_rank_decision import LowRankPhysicsAdapterDecision
from physics_transfer.transition_data import sample_transition_sequence_batch
from scripts.stage2_lowrank_gap_calibration import HELDOUT
from scripts.stage6_single_env_adaptation import PRETRAIN_FACTOR, pretrain
from scripts.stage7_mpc_decision_adaptation import cognitive_mpc_teacher, true_mpc_teacher, pretrain_decision, operator_query


def deploy(cognitive, pretrained_decision, factor, mode, rank, sequence_steps, episodes, seed, online_lr):
    torch.manual_seed(seed)
    if mode == "low_rank":
        decision = LowRankPhysicsAdapterDecision(copy.deepcopy(pretrained_decision), rank=rank)
        trainable = [decision.adapter_left, decision.adapter_right]
        optimizer = torch.optim.Adam(trainable, lr=online_lr)
    else:
        decision = copy.deepcopy(pretrained_decision)
        for parameter in decision.task_trunk.parameters(): parameter.requires_grad = False
        for parameter in decision.task_head.parameters(): parameter.requires_grad = False
        optimizer = torch.optim.Adam([decision.physics_basis], lr=online_lr) if mode == "full" else None
        trainable = [decision.physics_basis] if mode == "full" else []
    true_errors, cognitive_errors, norms = [], [], []
    for _ in range(episodes):
        batch = sample_transition_sequence_batch(1, sequence_steps, (factor,))
        latent = cognitive.initial_latent(1)
        episode_true, episode_cognitive, episode_norm = [], [], []
        for index in range(sequence_steps):
            state = batch["state"][:, index]
            random_action = batch["action"][:, index]
            target_state = batch["next_state"][:, index]
            code = operator_query(cognitive, state, latent)
            prediction = decision(state, code)["action"]
            target_cognitive = cognitive_mpc_teacher(cognitive, state, latent)
            target_true = true_mpc_teacher(state, factor)
            episode_true.append(F.mse_loss(prediction, target_true).item())
            episode_cognitive.append(F.mse_loss(prediction, target_cognitive).item())
            if optimizer is not None:
                loss = F.mse_loss(prediction, target_cognitive)
                if mode == "low_rank": loss = loss + 1e-3 * decision.stabilization_loss()
                optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(trainable, 5.0); optimizer.step()
            episode_norm.append((decision.adaptation_norm().item() if mode == "low_rank" else 0.0))
            with torch.no_grad(): latent = cognitive.observe_transition(state, random_action, target_state, latent).detach()
        true_errors.append(torch.tensor(episode_true)); cognitive_errors.append(torch.tensor(episode_cognitive)); norms.append(torch.tensor(episode_norm))
    true_errors, cognitive_errors, norms = torch.stack(true_errors), torch.stack(cognitive_errors), torch.stack(norms)
    return {
        "mode": mode, "rank": rank, "target_factor": factor,
        "true_first8_mse": true_errors[:, :8].mean().item(), "true_last8_mse": true_errors[:, -8:].mean().item(),
        "cognitive_first8_mse": cognitive_errors[:, :8].mean().item(), "cognitive_last8_mse": cognitive_errors[:, -8:].mean().item(),
        "adapter_norm_final": norms[:, -1].mean().item(),
    }


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--pretrain-cognitive-steps", type=int, default=100); parser.add_argument("--pretrain-decision-steps", type=int, default=100); parser.add_argument("--pretrain-sequence-steps", type=int, default=16); parser.add_argument("--deploy-sequence-steps", type=int, default=64); parser.add_argument("--episodes", type=int, default=4); parser.add_argument("--online-lr", type=float, default=1e-4); parser.add_argument("--rank", type=int, default=4); parser.add_argument("--seed", type=int, default=42); args = parser.parse_args()
    cognitive = pretrain(args.pretrain_cognitive_steps, args.pretrain_sequence_steps, args.seed)
    decision = pretrain_decision(cognitive, args.pretrain_decision_steps, args.pretrain_sequence_steps, args.seed)
    results = []
    for factor in HELDOUT:
        for mode in ("frozen", "full", "low_rank"):
            results.append(deploy(cognitive, decision, factor, mode, args.rank, args.deploy_sequence_steps, args.episodes, args.seed + 100, args.online_lr))
    print(json.dumps({"pretrain_factor": PRETRAIN_FACTOR[0], "results": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
