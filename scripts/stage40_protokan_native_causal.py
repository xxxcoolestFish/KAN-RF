"""Compare native ProtoKAN path derivatives with finite differences and truth."""

from __future__ import annotations

import argparse
import json

import torch

from physics_transfer.causal_graph import (
    cosine_by_sample,
    finite_difference_input_effect,
    temporal_action_effect,
)
from physics_transfer.multifactor_data import _random_states
from physics_transfer.protokan_causal import (
    cognitive_forward_jacobian,
    native_temporal_action_effect,
)
from physics_transfer.variants import step
from scripts.stage15_real_outcome_replay import PRETRAIN_FACTOR
from scripts.stage23_multistep_terminal_value import SimpleCognitiveKAN
from scripts.stage27_parameter_transport import pretrain_cognitive


def exact_transition(state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    factor = torch.tensor(
        PRETRAIN_FACTOR[0], dtype=state.dtype, device=state.device,
    ).view(1, 4).expand(state.shape[0], -1)
    return step(state, action, factor[:, 0], factor[:, 1], factor[:, 2], factor[:, 3])


def mean_cosine(first: torch.Tensor, second: torch.Tensor) -> float:
    return float(cosine_by_sample(first, second).mean().item())


def rmse(first: torch.Tensor, second: torch.Tensor) -> float:
    return float((first - second).square().mean().sqrt().item())


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
        native_prediction, native_jacobian = cognitive_forward_jacobian(
            cognitive, states, actions,
        )
        finite_prediction = cognitive(states, actions)
        finite_jacobian = finite_difference_input_effect(
            cognitive, states, actions, args.epsilon,
        )
        exact_jacobian = finite_difference_input_effect(
            exact_transition, states, actions, args.epsilon,
        )
    result = {
        "seed": seed,
        "cognitive_fit": cognitive_fit,
        "one_step": {
            "native_vs_forward_rmse": rmse(native_prediction, finite_prediction),
            "native_vs_finite_difference_cosine": mean_cosine(
                native_jacobian, finite_jacobian,
            ),
            "native_vs_finite_difference_rmse": rmse(
                native_jacobian, finite_jacobian,
            ),
            "native_action_vs_exact_action_cosine": mean_cosine(
                native_jacobian[:, :, -1:], exact_jacobian[:, -1:, :].transpose(1, 2),
            ) if False else mean_cosine(
                native_jacobian[:, :, -1:], exact_jacobian[:, -1:, :],
            ),
        },
        "temporal": {},
    }
    # The native and finite-difference model effects should agree exactly if
    # the analytical chain rule is implemented correctly.  The truth effect
    # is computed with the exact source dynamics and is not differentiated.
    for horizon in args.horizons:
        sequence = torch.rand(args.probe_count, horizon, 1) * 2.0 - 1.0
        with torch.no_grad():
            native_effect = native_temporal_action_effect(
                cognitive, states, sequence,
            )
            finite_model_effect = temporal_action_effect(
                cognitive, states, sequence, args.epsilon,
            )
            exact_effect = temporal_action_effect(
                exact_transition, states, sequence, args.epsilon,
            )
        result["temporal"][f"horizon_{horizon}"] = {
            "native_vs_finite_model_cosine": mean_cosine(
                native_effect, finite_model_effect,
            ),
            "native_vs_finite_model_rmse": rmse(
                native_effect, finite_model_effect,
            ),
            "native_vs_exact_cosine": mean_cosine(native_effect, exact_effect),
            "finite_model_vs_exact_cosine": mean_cosine(
                finite_model_effect, exact_effect,
            ),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cognitive-steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--probe-count", type=int, default=128)
    parser.add_argument("--probe-seed", type=int, default=20260719)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--json-out", type=str, default="")
    args = parser.parse_args()
    output = {
        "architecture": "ProtoKANNativeCausalJacobian",
        "experiment": "native_path_derivative_vs_finite_difference_vs_exact",
        "source_factor": PRETRAIN_FACTOR[0],
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
