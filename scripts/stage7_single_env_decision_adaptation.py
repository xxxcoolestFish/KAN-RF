"""Single-environment pretraining and decision adaptation via operator queries."""

from __future__ import annotations

import argparse
import copy
import json

import torch
import torch.nn.functional as F

from physics_transfer.separated_decision import SeparatedPhysicsDecision
from physics_transfer.streaming_cognitive import StreamingCognitiveWorldModel
from physics_transfer.transition_data import sample_transition_sequence_batch
from physics_transfer.variants import step
from scripts.stage2_lowrank_gap_calibration import FACTORS, HELDOUT
from scripts.stage6_single_env_adaptation import PRETRAIN_FACTOR, pretrain


ACTIONS = torch.linspace(-1.0, 1.0, 9)
STATE_COST_TEMPERATURE = 0.10


def state_cost(state):
    return (
        0.5 * state[:, 4].square() + 0.5 * state[:, 5].square()
        + 0.5 * (1.0 - state[:, 0]) + 0.25 * (1.0 - state[:, 2])
    )


def operator_query(cognitive, state, latent):
    """Query the cognitive operator on a fixed action probe set."""
    batch = state.shape[0]
    actions = ACTIONS.to(state.device).view(1, -1, 1).expand(batch, -1, -1)
    state_rep = state[:, None, :].expand(-1, actions.shape[1], -1).reshape(-1, state.shape[-1])
    action_rep = actions.reshape(-1, 1)
    latent_rep = latent[:, None, :].expand(-1, actions.shape[1], -1).reshape(-1, latent.shape[-1])
    zero = cognitive.initial_latent(state_rep.shape[0], state.device)
    response = cognitive.predict_next(state_rep, action_rep, latent_rep)
    base = cognitive.predict_next(state_rep, action_rep, zero)
    return (response - base).view(batch, -1)


def cognitive_teacher(cognitive, state, latent):
    batch = state.shape[0]
    actions = ACTIONS.to(state.device).view(1, -1, 1).expand(batch, -1, -1)
    state_rep = state[:, None, :].expand(-1, actions.shape[1], -1).reshape(-1, state.shape[-1])
    action_rep = actions.reshape(-1, 1)
    latent_rep = latent[:, None, :].expand(-1, actions.shape[1], -1).reshape(-1, latent.shape[-1])
    with torch.no_grad():
        predicted = cognitive.predict_next(state_rep, action_rep, latent_rep)
        costs = state_cost(predicted).view(batch, -1)
        weights = torch.softmax(-costs / STATE_COST_TEMPERATURE, dim=-1)
    return (weights * ACTIONS.to(state.device).view(1, -1)).sum(dim=-1, keepdim=True)


def true_teacher(state, factor):
    batch = state.shape[0]
    actions = ACTIONS.to(state.device).view(1, -1, 1).expand(batch, -1, -1)
    state_rep = state[:, None, :].expand(-1, actions.shape[1], -1).reshape(-1, state.shape[-1])
    action_rep = actions.reshape(-1, 1)
    factors = torch.tensor(factor, dtype=state.dtype).repeat(batch * actions.shape[1], 1)
    with torch.no_grad():
        predicted = step(state_rep, action_rep, factors[:, 0], factors[:, 1], factors[:, 2], factors[:, 3])
        costs = state_cost(predicted).view(batch, -1)
        weights = torch.softmax(-costs / STATE_COST_TEMPERATURE, dim=-1)
    return (weights * ACTIONS.to(state.device).view(1, -1)).sum(dim=-1, keepdim=True)


def pretrain_decision(cognitive, steps: int, sequence_steps: int, seed: int):
    torch.manual_seed(seed + 3000)
    decision = SeparatedPhysicsDecision(6, 1, 9 * 6, hidden_dim=24, n_prototypes=8)
    optimizer = torch.optim.Adam(decision.parameters(), lr=2e-3)
    cognitive.eval()
    for _ in range(steps):
        batch = sample_transition_sequence_batch(32, sequence_steps, PRETRAIN_FACTOR)
        with torch.no_grad():
            output = cognitive.forward_sequence(batch["state"], batch["action"], batch["next_state"])
            states = batch["state"].reshape(-1, 6)
            latents = output["pre_latents"].reshape(-1, cognitive.latent_dim)
            code = operator_query(cognitive, states, latents)
            target = cognitive_teacher(cognitive, states, latents)
        prediction = decision(states, code)["action"]
        loss = F.mse_loss(prediction, target)
        optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(decision.parameters(), 5.0); optimizer.step()
    return decision


def deploy(cognitive, pretrained_decision, factor, mode: str, sequence_steps: int,
           episodes: int, seed: int, online_lr: float):
    torch.manual_seed(seed)
    decision = copy.deepcopy(pretrained_decision)
    for parameter in decision.task_trunk.parameters():
        parameter.requires_grad = False
    for parameter in decision.task_head.parameters():
        parameter.requires_grad = False
    optimizer = None
    if mode == "online_physics":
        optimizer = torch.optim.Adam([decision.physics_basis], lr=online_lr)
    errors_true, errors_cognitive = [], []
    for _ in range(episodes):
        batch = sample_transition_sequence_batch(1, sequence_steps, (factor,))
        latent = cognitive.initial_latent(1)
        episode_true, episode_cognitive = [], []
        for index in range(sequence_steps):
            state = batch["state"][:, index]
            random_action = batch["action"][:, index]
            target_state = batch["next_state"][:, index]
            code = operator_query(cognitive, state, latent)
            prediction = decision(state, code)["action"]
            target_cognitive = cognitive_teacher(cognitive, state, latent)
            target_true = true_teacher(state, factor)
            episode_true.append(F.mse_loss(prediction, target_true).item())
            episode_cognitive.append(F.mse_loss(prediction, target_cognitive).item())
            if optimizer is not None:
                loss = F.mse_loss(prediction, target_cognitive)
                optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_([decision.physics_basis], 5.0); optimizer.step()
            with torch.no_grad():
                latent = cognitive.observe_transition(state, random_action, target_state, latent).detach()
        errors_true.append(torch.tensor(episode_true)); errors_cognitive.append(torch.tensor(episode_cognitive))
    true_errors = torch.stack(errors_true); cognitive_errors = torch.stack(errors_cognitive)
    return {
        "mode": mode, "target_factor": factor,
        "true_teacher_first8_mse": true_errors[:, :8].mean().item(),
        "true_teacher_last8_mse": true_errors[:, -8:].mean().item(),
        "cognitive_teacher_first8_mse": cognitive_errors[:, :8].mean().item(),
        "cognitive_teacher_last8_mse": cognitive_errors[:, -8:].mean().item(),
        "true_teacher_gain": (true_errors[:, :8].mean() - true_errors[:, -8:].mean()).item(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain-cognitive-steps", type=int, default=200)
    parser.add_argument("--pretrain-decision-steps", type=int, default=200)
    parser.add_argument("--pretrain-sequence-steps", type=int, default=16)
    parser.add_argument("--deploy-sequence-steps", type=int, default=64)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--online-lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    cognitive = pretrain(args.pretrain_cognitive_steps, args.pretrain_sequence_steps, args.seed)
    decision = pretrain_decision(cognitive, args.pretrain_decision_steps, args.pretrain_sequence_steps, args.seed)
    results = []
    for factor in HELDOUT:
        for mode in ("zero_shot", "online_physics"):
            results.append(deploy(cognitive, decision, factor, mode, args.deploy_sequence_steps, args.episodes, args.seed + 100, args.online_lr))
    print(json.dumps({"pretrain_factor": PRETRAIN_FACTOR[0], "results": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
