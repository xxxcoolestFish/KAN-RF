"""Corrected experiment: single-environment pretraining and online transfer."""

from __future__ import annotations

import argparse
import copy
import json

import torch
import torch.nn.functional as F

from physics_transfer.streaming_cognitive import StreamingCognitiveWorldModel
from physics_transfer.transition_data import sample_transition_sequence_batch
from scripts.stage2_lowrank_gap_calibration import FACTORS, HELDOUT


PRETRAIN_FACTOR = (FACTORS[0],)


def pretrain(steps: int, sequence_steps: int, seed: int):
    torch.manual_seed(seed)
    model = StreamingCognitiveWorldModel(latent_dim=16, hidden_dim=32, n_prototypes=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    for _ in range(steps):
        batch = sample_transition_sequence_batch(32, sequence_steps, PRETRAIN_FACTOR)
        output = model.forward_sequence(batch["state"], batch["action"], batch["next_state"])
        loss = F.smooth_l1_loss(output["predictions"], batch["next_state"])
        optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
    return model


def deploy(pretrained, factor, mode: str, sequence_steps: int, episodes: int,
           seed: int, online_lr: float):
    torch.manual_seed(seed)
    model = copy.deepcopy(pretrained)
    if mode == "online_dynamics":
        optimizer = torch.optim.Adam(model.dynamics.parameters(), lr=online_lr)
    elif mode == "online_full":
        optimizer = torch.optim.Adam(model.parameters(), lr=online_lr)
    else:
        optimizer = None
    all_errors = []
    for _ in range(episodes):
        batch = sample_transition_sequence_batch(1, sequence_steps, (factor,))
        latent = model.initial_latent(1)
        episode_errors = []
        for index in range(sequence_steps):
            state = batch["state"][:, index]
            action = batch["action"][:, index]
            target = batch["next_state"][:, index]
            prediction = model.predict_next(state, action, latent)
            loss = F.mse_loss(prediction, target)
            episode_errors.append(loss.item())
            if optimizer is not None:
                optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
            if mode != "frozen":
                with torch.no_grad():
                    latent = model.observe_transition(state, action, target, latent).detach()
            else:
                latent = latent.detach()
        all_errors.append(torch.tensor(episode_errors))
    errors = torch.stack(all_errors)
    return {
        "mode": mode,
        "target_factor": factor,
        "first_step_loss": errors[:, 0].mean().item(),
        "last_step_loss": errors[:, -1].mean().item(),
        "first_8_loss": errors[:, :8].mean().item(),
        "last_8_loss": errors[:, -8:].mean().item(),
        "adaptation_gain": (errors[:, :8].mean() - errors[:, -8:].mean()).item(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain-steps", type=int, default=300)
    parser.add_argument("--pretrain-sequence-steps", type=int, default=16)
    parser.add_argument("--deploy-sequence-steps", type=int, default=64)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--online-lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    model = pretrain(args.pretrain_steps, args.pretrain_sequence_steps, args.seed)
    results = []
    for factor in HELDOUT:
        for mode in ("frozen", "latent_only", "online_dynamics", "online_full"):
            results.append(deploy(model, factor, mode, args.deploy_sequence_steps, args.episodes, args.seed + 100, args.online_lr))
    print(json.dumps({"pretrain_factor": PRETRAIN_FACTOR[0], "results": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
