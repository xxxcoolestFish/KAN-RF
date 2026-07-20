"""Extract and validate a temporal causal graph from a trained ProtoKAN WM."""

from __future__ import annotations

import argparse
import json

import torch

from physics_transfer.causal_graph import (
    cosine_by_sample,
    finite_difference_input_effect,
    summarize_effect,
    temporal_action_effect,
)
from physics_transfer.multifactor_data import _random_states
from physics_transfer.variants import step
from scripts.stage15_real_outcome_replay import PRETRAIN_FACTOR
from scripts.stage23_multistep_terminal_value import SimpleCognitiveKAN
from scripts.stage27_parameter_transport import pretrain_cognitive


STATE_NAMES = ["cos_theta1", "sin_theta1", "cos_theta2", "sin_theta2",
               "velocity1", "velocity2"]
INPUT_NAMES = STATE_NAMES + ["action"]


def exact_transition(state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    factor = torch.tensor(
        PRETRAIN_FACTOR[0], dtype=state.dtype, device=state.device,
    ).view(1, 4).expand(state.shape[0], -1)
    return step(state, action, factor[:, 0], factor[:, 1], factor[:, 2], factor[:, 3])


def top_edges(summary: dict, source_names: list[str], target_names: list[str], top_k: int):
    strength = summary["median_abs"]
    signed = summary["signed_mean"]
    stability = summary["sign_consistency"]
    edges = []
    for source_index, source_name in enumerate(source_names):
        for target_index, target_name in enumerate(target_names):
            edges.append({
                "source": source_name,
                "target": target_name,
                "median_abs": float(strength[source_index][target_index]),
                "signed_mean": float(signed[source_index][target_index]),
                "sign_consistency": float(stability[source_index][target_index]),
            })
    return sorted(edges, key=lambda item: item["median_abs"], reverse=True)[:top_k]


def temporal_summary(summary: dict, horizon: int):
    signed = summary["signed_mean"]
    strength = summary["median_abs"]
    stability = summary["sign_consistency"]
    return [
        {
            "horizon_step": step_index + 1,
            "target": STATE_NAMES[target_index],
            "median_abs": float(strength[step_index][target_index]),
            "signed_mean": float(signed[step_index][target_index]),
            "sign_consistency": float(stability[step_index][target_index]),
        }
        for step_index in range(horizon)
        for target_index in range(len(STATE_NAMES))
    ]


def run_seed(args: argparse.Namespace, seed: int) -> dict:
    torch.manual_seed(seed)
    cognitive = SimpleCognitiveKAN()
    cognitive_fit = pretrain_cognitive(
        cognitive, args.cognitive_steps, args.batch_size, seed,
    )
    states = _random_states(
        args.probe_count,
        generator=torch.Generator().manual_seed(args.probe_seed + seed),
    )
    actions = torch.rand(args.probe_count, 1) * 2.0 - 1.0
    with torch.no_grad():
        learned_direct = finite_difference_input_effect(
            cognitive, states, actions, args.epsilon,
        )
        exact_direct = finite_difference_input_effect(
            exact_transition, states, actions, args.epsilon,
        )
    direct_cosine = cosine_by_sample(learned_direct, exact_direct)
    result = {
        "seed": seed,
        "cognitive_fit": cognitive_fit,
        "direct_effect": {
            "learned_summary": summarize_effect(learned_direct),
            "exact_summary": summarize_effect(exact_direct),
            "mean_cosine_similarity": float(direct_cosine.mean().item()),
            "median_cosine_similarity": float(direct_cosine.median().item()),
        },
        "top_direct_edges": {
            "learned": top_edges(
                summarize_effect(learned_direct), INPUT_NAMES, STATE_NAMES, args.top_k,
            ),
            "exact": top_edges(
                summarize_effect(exact_direct), INPUT_NAMES, STATE_NAMES, args.top_k,
            ),
        },
        "temporal_action_effect": {},
    }
    for horizon in args.horizons:
        sequence = torch.rand(args.probe_count, horizon, 1) * 2.0 - 1.0
        with torch.no_grad():
            learned_temporal = temporal_action_effect(
                cognitive, states, sequence, args.epsilon,
            )
            exact_temporal = temporal_action_effect(
                exact_transition, states, sequence, args.epsilon,
            )
        cosine = cosine_by_sample(learned_temporal, exact_temporal)
        learned_summary = summarize_effect(learned_temporal)
        exact_summary = summarize_effect(exact_temporal)
        result["temporal_action_effect"][f"horizon_{horizon}"] = {
            "learned": temporal_summary(learned_summary, horizon),
            "exact": temporal_summary(exact_summary, horizon),
            "mean_cosine_similarity": float(cosine.mean().item()),
            "median_cosine_similarity": float(cosine.median().item()),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cognitive-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--probe-count", type=int, default=128)
    parser.add_argument("--probe-seed", type=int, default=20260719)
    parser.add_argument("--epsilon", type=float, default=1e-3)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--json-out", type=str, default="")
    args = parser.parse_args()
    output = {
        "architecture": "ProtoKANTemporalCausalGraph",
        "experiment": "local_intervention_and_temporal_action_effect",
        "source_factor": PRETRAIN_FACTOR[0],
        "state_names": STATE_NAMES,
        "input_names": INPUT_NAMES,
        "config": vars(args),
        "seeds": [run_seed(args, seed) for seed in args.seeds],
    }
    text = json.dumps(output, ensure_ascii=False, indent=2)
    print(text)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")


if __name__ == "__main__":
    main()
