"""Robust, task-aware evaluation for a trained direct corridor Actor."""

from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

import torch

from cpbn import OracleAcrobotDynamics, tip_height
from cpbn.corridor_policy import DirectCorridorActor, future_corridor
from cpbn.receding_tube import local_state_distance
from scripts.validate_direct_corridor_actor import evaluate_full, perturbed_reference


@torch.no_grad()
def evaluate_segments_task_aware(actor, dynamics, reference, args, trials, seed):
    generator = torch.Generator().manual_seed(seed)
    starts = torch.arange(0, reference.shape[0] - 1, args.segment_steps)
    edge = torch.arange(starts.shape[0]).repeat_interleave(trials)
    start_phase = starts[edge]
    state = perturbed_reference(
        reference, start_phase, generator, args.initial_noise,
    )
    task_success = torch.zeros(state.shape[0], dtype=torch.bool)
    for step in range(args.segment_steps):
        phase = (start_phase + step).clamp_max(reference.shape[0] - 1)
        corridor = future_corridor(reference, phase, args.corridor_horizon)
        action, _ = actor.sample(state, corridor, deterministic=True)
        state = dynamics(state, action)
        task_success |= tip_height(state) >= 1.0
    final_phase = (start_phase + args.segment_steps).clamp_max(
        reference.shape[0] - 1,
    )
    endpoint = local_state_distance(
        state, reference[final_phase],
    ) <= args.segment_completion_radius
    task_or_endpoint = endpoint | task_success

    def per_segment(value):
        return value.view(starts.shape[0], trials).float().mean(dim=1)

    endpoint_per_segment = per_segment(endpoint)
    task_per_segment = per_segment(task_success)
    combined_per_segment = per_segment(task_or_endpoint)
    return {
        "endpoint_completion_rate": float(endpoint.float().mean()),
        "minimum_endpoint_completion": float(endpoint_per_segment.min()),
        "per_segment_endpoint_completion": endpoint_per_segment.tolist(),
        "per_segment_task_success": task_per_segment.tolist(),
        "task_or_endpoint_completion_rate": float(task_or_endpoint.float().mean()),
        "minimum_task_or_endpoint_completion": float(combined_per_segment.min()),
        "per_segment_task_or_endpoint_completion": combined_per_segment.tolist(),
    }


def aggregate_full(records):
    total = sum(item["success_count"] for item in records)
    count = sum(item["evaluation_count"] for item in records)
    return {
        "success_count": total,
        "evaluation_count": count,
        "success_rate": total / count,
        "mean_of_seed_mean_maximum_height": sum(
            item["mean_maximum_height"] for item in records
        ) / len(records),
        "minimum_maximum_height": min(
            item["minimum_maximum_height"] for item in records
        ),
        "per_seed": records,
    }


def run(cli):
    payload = torch.load(cli.checkpoint, map_location="cpu", weights_only=False)
    config = dict(payload["config"])
    args = SimpleNamespace(**config)
    reference = payload["reference"]
    actor = DirectCorridorActor(config["hidden_dim"], config["log_std_init"])
    actor.load_state_dict(payload["actor"])
    actor.eval()
    dynamics = OracleAcrobotDynamics()
    seeds = [cli.seed + index * 1009 for index in range(cli.num_seeds)]

    full_records, shuffled_records, segment_records = [], [], []
    for seed in seeds:
        normal = evaluate_full(
            actor, dynamics, reference, args, cli.full_count, seed,
        )
        normal["seed"] = seed
        normal["evaluation_count"] = cli.full_count
        full_records.append(normal)
        shuffled = evaluate_full(
            actor, dynamics, reference, args, cli.full_count, seed,
            shuffled=True,
        )
        shuffled["seed"] = seed
        shuffled["evaluation_count"] = cli.full_count
        shuffled_records.append(shuffled)
        segment_records.append(evaluate_segments_task_aware(
            actor, dynamics, reference, args, cli.segment_trials,
            seed + 500000,
        ))

    segment_endpoint = torch.tensor([
        item["per_segment_endpoint_completion"] for item in segment_records
    ]).mean(dim=0)
    segment_combined = torch.tensor([
        item["per_segment_task_or_endpoint_completion"]
        for item in segment_records
    ]).mean(dim=0)
    segment_task = torch.tensor([
        item["per_segment_task_success"] for item in segment_records
    ]).mean(dim=0)
    output = {
        "experiment": "RobustDirectCorridorActorEvaluation",
        "checkpoint": cli.checkpoint,
        "num_seeds": cli.num_seeds,
        "full_count_per_seed": cli.full_count,
        "segment_trials_per_seed": cli.segment_trials,
        "full_route": aggregate_full(full_records),
        "shuffled_corridor_full_route": aggregate_full(shuffled_records),
        "segments": {
            "endpoint_completion_rate": float(segment_endpoint.mean()),
            "minimum_endpoint_completion": float(segment_endpoint.min()),
            "per_segment_endpoint_completion": segment_endpoint.tolist(),
            "task_or_endpoint_completion_rate": float(segment_combined.mean()),
            "minimum_task_or_endpoint_completion": float(segment_combined.min()),
            "per_segment_task_or_endpoint_completion": segment_combined.tolist(),
            "per_segment_task_success": segment_task.tolist(),
        },
    }
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", default="results/direct_corridor_actor_strong_seed0.pt",
    )
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--full-count", type=int, default=64)
    parser.add_argument("--segment-trials", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument(
        "--json-out", default="results/direct_corridor_actor_robust_seed0.json",
    )
    cli = parser.parse_args()
    output = run(cli)
    text = json.dumps(output, ensure_ascii=False, indent=2)
    with open(cli.json_out, "w", encoding="utf-8") as handle:
        handle.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
