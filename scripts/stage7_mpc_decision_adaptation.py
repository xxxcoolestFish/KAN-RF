"""Single-environment decision transfer with a multi-step MPC teacher."""

from __future__ import annotations

import argparse
import copy
import json

import torch
import torch.nn.functional as F

from physics_transfer.separated_decision import SeparatedPhysicsDecision
from physics_transfer.transition_data import sample_transition_sequence_batch
from physics_transfer.variants import step
from scripts.stage2_lowrank_gap_calibration import HELDOUT
from scripts.stage6_single_env_adaptation import PRETRAIN_FACTOR, pretrain
from scripts.stage7_single_env_decision_adaptation import operator_query


ACTIONS = torch.linspace(-1.0, 1.0, 9)
HORIZON = 5
TEMPERATURE = 0.10


def state_cost(state):
    return 0.5 * state[:, 4].square() + 0.5 * state[:, 5].square() + 0.5 * (1.0 - state[:, 0]) + 0.25 * (1.0 - state[:, 2])


def cognitive_mpc_teacher(cognitive, state, latent):
    batch = state.shape[0]
    actions = ACTIONS.to(state.device).view(1, -1, 1).expand(batch, -1, -1)
    current = state[:, None, :].expand(-1, actions.shape[1], -1).reshape(-1, state.shape[-1])
    action = actions.reshape(-1, 1)
    latent_rep = latent[:, None, :].expand(-1, actions.shape[1], -1).reshape(-1, latent.shape[-1])
    with torch.no_grad():
        for _ in range(HORIZON):
            current = cognitive.predict_next(current, action, latent_rep)
        costs = state_cost(current).view(batch, -1)
        weights = torch.softmax(-costs / TEMPERATURE, dim=-1)
    return (weights * ACTIONS.to(state.device).view(1, -1)).sum(dim=-1, keepdim=True)


def true_mpc_teacher(state, factor):
    batch = state.shape[0]
    actions = ACTIONS.to(state.device).view(1, -1, 1).expand(batch, -1, -1)
    current0 = state[:, None, :].expand(-1, actions.shape[1], -1).reshape(-1, state.shape[-1])
    action = actions.reshape(-1, 1)
    factors = torch.tensor(factor, dtype=state.dtype).repeat(batch * actions.shape[1], 1)
    with torch.no_grad():
        current = current0
        for _ in range(HORIZON):
            current = step(current, action, factors[:, 0], factors[:, 1], factors[:, 2], factors[:, 3])
        costs = state_cost(current).view(batch, -1)
        weights = torch.softmax(-costs / TEMPERATURE, dim=-1)
    return (weights * ACTIONS.to(state.device).view(1, -1)).sum(dim=-1, keepdim=True)


def pretrain_decision(cognitive, steps, sequence_steps, seed):
    torch.manual_seed(seed + 3000)
    decision = SeparatedPhysicsDecision(6, 1, 54, hidden_dim=24, n_prototypes=8)
    optimizer = torch.optim.Adam(decision.parameters(), lr=2e-3)
    cognitive.eval()
    for _ in range(steps):
        batch = sample_transition_sequence_batch(32, sequence_steps, PRETRAIN_FACTOR)
        with torch.no_grad():
            output = cognitive.forward_sequence(batch["state"], batch["action"], batch["next_state"])
            state = batch["state"].reshape(-1, 6)
            latent = output["pre_latents"].reshape(-1, cognitive.latent_dim)
            code = operator_query(cognitive, state, latent)
            target = cognitive_mpc_teacher(cognitive, state, latent)
        prediction = decision(state, code)["action"]
        loss = F.mse_loss(prediction, target)
        optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(decision.parameters(), 5.0); optimizer.step()
    return decision


def deploy(cognitive, pretrained_decision, factor, mode, sequence_steps, episodes, seed, online_lr):
    torch.manual_seed(seed)
    decision = copy.deepcopy(pretrained_decision)
    for parameter in decision.task_trunk.parameters(): parameter.requires_grad = False
    for parameter in decision.task_head.parameters(): parameter.requires_grad = False
    optimizer = torch.optim.Adam([decision.physics_basis], lr=online_lr) if mode == "online_physics" else None
    true_errors, cognitive_errors = [], []
    for _ in range(episodes):
        batch = sample_transition_sequence_batch(1, sequence_steps, (factor,))
        latent = cognitive.initial_latent(1)
        episode_true, episode_cognitive = [], []
        for index in range(sequence_steps):
            state, random_action, target_state = batch["state"][:, index], batch["action"][:, index], batch["next_state"][:, index]
            code = operator_query(cognitive, state, latent)
            prediction = decision(state, code)["action"]
            cognitive_target = cognitive_mpc_teacher(cognitive, state, latent)
            true_target = true_mpc_teacher(state, factor)
            episode_true.append(F.mse_loss(prediction, true_target).item())
            episode_cognitive.append(F.mse_loss(prediction, cognitive_target).item())
            if optimizer is not None:
                loss = F.mse_loss(prediction, cognitive_target)
                optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_([decision.physics_basis], 5.0); optimizer.step()
            with torch.no_grad(): latent = cognitive.observe_transition(state, random_action, target_state, latent).detach()
        true_errors.append(torch.tensor(episode_true)); cognitive_errors.append(torch.tensor(episode_cognitive))
    true_errors, cognitive_errors = torch.stack(true_errors), torch.stack(cognitive_errors)
    return {"mode": mode, "target_factor": factor, "true_teacher_first8_mse": true_errors[:, :8].mean().item(), "true_teacher_last8_mse": true_errors[:, -8:].mean().item(), "cognitive_teacher_first8_mse": cognitive_errors[:, :8].mean().item(), "cognitive_teacher_last8_mse": cognitive_errors[:, -8:].mean().item()}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--pretrain-cognitive-steps", type=int, default=100); parser.add_argument("--pretrain-decision-steps", type=int, default=100); parser.add_argument("--pretrain-sequence-steps", type=int, default=16); parser.add_argument("--deploy-sequence-steps", type=int, default=64); parser.add_argument("--episodes", type=int, default=4); parser.add_argument("--online-lr", type=float, default=5e-4); parser.add_argument("--seed", type=int, default=42); args = parser.parse_args()
    cognitive = pretrain(args.pretrain_cognitive_steps, args.pretrain_sequence_steps, args.seed)
    decision = pretrain_decision(cognitive, args.pretrain_decision_steps, args.pretrain_sequence_steps, args.seed)
    results = []
    for factor in HELDOUT:
        for mode in ("zero_shot", "online_physics"):
            results.append(deploy(cognitive, decision, factor, mode, args.deploy_sequence_steps, args.episodes, args.seed + 100, args.online_lr))
    print(json.dumps({"pretrain_factor": PRETRAIN_FACTOR[0], "horizon": HORIZON, "results": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
