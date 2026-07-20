"""Compare fixed-clock and state-feedback phase for a trained corridor Actor."""

from __future__ import annotations

import argparse
import json
from argparse import Namespace

import torch

from cpbn import OracleAcrobotDynamics, tip_height
from cpbn.corridor_policy import DirectCorridorActor, future_corridor
from cpbn.feedback_phase import (
    belief_phase,
    bounded_nearest_phase,
    initialize_phase_belief,
    update_phase_belief,
)
from scripts.validate_direct_corridor_actor import perturbed_reference


@torch.no_grad()
def evaluate_mode(actor, dynamics, reference, config, cli, mode, seed):
    generator = torch.Generator().manual_seed(seed)
    phase = torch.zeros(cli.count, dtype=torch.long)
    state = perturbed_reference(
        reference, phase, generator, config.full_initial_noise,
    )
    belief = initialize_phase_belief(phase, reference.shape[0])
    success = torch.zeros(cli.count, dtype=torch.bool)
    success_step = torch.full((cli.count,), -1, dtype=torch.long)
    maximum = torch.full((cli.count,), -2.0)
    previous_phase = phase.clone()
    stayed = backward = jumped = 0
    active_transitions = 0
    mean_lag = []
    for step in range(cli.steps):
        active = ~success
        corridor = future_corridor(reference, phase, config.corridor_horizon)
        action, _ = actor.sample(state, corridor, deterministic=True)
        candidate_state = dynamics(state, action)
        state = torch.where(active.unsqueeze(-1), candidate_state, state)
        height = tip_height(state)
        maximum = torch.maximum(maximum, height)
        newly_successful = active & (height >= 1.0)
        success_step = torch.where(
            newly_successful, torch.full_like(success_step, step + 1), success_step,
        )
        success |= newly_successful
        if mode == "fixed":
            candidate_phase = torch.full_like(
                phase, min(step + 1, reference.shape[0] - 1),
            )
        elif mode == "nearest":
            candidate_phase = bounded_nearest_phase(
                state, reference, phase,
                backtrack=cli.nearest_backtrack,
                advance=cli.nearest_advance,
            )
        elif mode == "bayes":
            candidate_belief = update_phase_belief(
                belief, state, reference,
                observation_scale=cli.observation_scale,
                minimum_phase=phase,
            )
            candidate_phase = belief_phase(candidate_belief)
            belief = torch.where(
                success.unsqueeze(-1), belief, candidate_belief,
            )
        else:
            raise ValueError(f"unknown phase mode: {mode}")
        phase = torch.where(success, phase, candidate_phase)
        delta = phase - previous_phase
        stayed += int(((delta == 0) & active).sum())
        backward += int(((delta < 0) & active).sum())
        jumped += int(((delta > 1) & active).sum())
        active_transitions += int(active.sum())
        previous_phase = phase.clone()
        clock = min(step + 1, reference.shape[0] - 1)
        if bool((~success).any()):
            mean_lag.append(float((clock - phase[~success]).float().mean()))
    successful_steps = success_step[success]
    return {
        "seed": seed,
        "success_count": int(success.sum()),
        "success_rate": float(success.float().mean()),
        "mean_maximum_height": float(maximum.mean()),
        "minimum_maximum_height": float(maximum.min()),
        "mean_final_phase": float(phase.float().mean()),
        "minimum_final_phase": int(phase.min()),
        "mean_success_step": (
            float(successful_steps.float().mean())
            if successful_steps.numel() else None
        ),
        "mean_clock_lag_of_active_routes": (
            sum(mean_lag) / len(mean_lag) if mean_lag else 0.0
        ),
        "stay_rate": stayed / active_transitions,
        "backward_rate": backward / active_transitions,
        "jump_rate": jumped / active_transitions,
    }


def aggregate(records):
    total = sum(item["success_count"] for item in records)
    count = len(records) * records[0]["evaluation_count"]
    return {
        "success_count": total,
        "evaluation_count": count,
        "success_rate": total / count,
        "mean_maximum_height": sum(
            item["mean_maximum_height"] for item in records
        ) / len(records),
        "minimum_maximum_height": min(
            item["minimum_maximum_height"] for item in records
        ),
        "mean_final_phase": sum(
            item["mean_final_phase"] for item in records
        ) / len(records),
        "mean_success_step": sum(
            item["mean_success_step"] * item["success_count"] for item in records
            if item["mean_success_step"] is not None
        ) / max(total, 1),
        "per_seed": records,
    }


def run(cli):
    checkpoint = torch.load(cli.checkpoint, map_location="cpu", weights_only=False)
    config = Namespace(**checkpoint["config"])
    reference = checkpoint["reference"]
    actor = DirectCorridorActor(config.hidden_dim, config.log_std_init)
    actor.load_state_dict(checkpoint["actor"])
    actor.eval()
    dynamics = OracleAcrobotDynamics()
    seeds = [cli.seed + index * 1009 for index in range(cli.num_seeds)]
    modes = {}
    for mode in ("fixed", "nearest", "bayes"):
        records = []
        for seed in seeds:
            record = evaluate_mode(
                actor, dynamics, reference, config, cli, mode, seed,
            )
            record["evaluation_count"] = cli.count
            records.append(record)
        modes[mode] = aggregate(records)
    return {
        "experiment": "FeedbackPhaseActorValidation",
        "checkpoint": cli.checkpoint,
        "config": vars(cli),
        "modes": modes,
        "best_mode": max(modes, key=lambda name: modes[name]["success_rate"]),
        "action_teacher_used": False,
        "actor_retrained": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", default="results/direct_corridor_actor_strong_seed0.pt",
    )
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--observation-scale", type=float, default=0.10)
    parser.add_argument("--nearest-backtrack", type=int, default=4)
    parser.add_argument("--nearest-advance", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument(
        "--json-out", default="results/feedback_phase_actor_seed0.json",
    )
    cli = parser.parse_args()
    output = run(cli)
    text = json.dumps(output, ensure_ascii=False, indent=2)
    with open(cli.json_out, "w", encoding="utf-8") as handle:
        handle.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
