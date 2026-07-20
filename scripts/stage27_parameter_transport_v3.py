"""Parameter-transport validation with a configurable CLI."""

from __future__ import annotations

import argparse
import copy
import json

import torch

from physics_transfer.parameter_transport_v2 import ParameterTransport, TransportedMLPPolicy
from scripts import stage27_parameter_transport as base
from scripts.stage15_real_outcome_replay import PRETRAIN_FACTOR
from scripts.stage2_lowrank_loss_ablation import FACTORS, HELDOUT
from scripts.stage23_multistep_terminal_value import GOAL, SimpleCognitiveKAN


def adapt_cognitive(cognitive, factor, steps, batch_size, seed):
    for parameter in cognitive.parameters():
        parameter.requires_grad = True
    return base.adapt_cognitive(cognitive, factor, steps, batch_size, seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cognitive-steps", type=int, default=250)
    parser.add_argument("--transport-steps", type=int, default=200)
    parser.add_argument("--policy-steps", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-horizon", type=int, default=4)
    parser.add_argument("--rollout-steps", type=int, default=64)
    parser.add_argument("--test-count", type=int, default=20)
    parser.add_argument("--test-seed", type=int, default=20260718)
    parser.add_argument("--adapt-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json-out", type=str, default="")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    cognitive = SimpleCognitiveKAN()
    cognitive_fit = base.pretrain_cognitive(
        cognitive, args.cognitive_steps, args.batch_size, args.seed,
    )
    transport = ParameterTransport(cognitive, ((32, 12), (1, 32)), rank=4)
    _, transport_fit = base.fit_transport(
        cognitive, transport, args.transport_steps, args.batch_size, args.seed,
    )
    transported = TransportedMLPPolicy(cognitive, transport)
    plain = copy.deepcopy(transported)
    goal = GOAL.view(1, -1)
    transported_fit = base.train_policy(
        transported, cognitive, goal, args.policy_steps, args.batch_size,
        args.max_horizon, args.seed, True,
    )
    plain_fit = base.train_policy(
        plain, plain.cognitive, goal, args.policy_steps, args.batch_size,
        args.max_horizon, args.seed, False,
    )
    generator = torch.Generator().manual_seed(args.test_seed)
    states = base._random_states(args.test_count, generator=generator)
    source = PRETRAIN_FACTOR[0]
    factors = [("source", source)]
    factors += [(f"factor_{i + 1}", f) for i, f in enumerate(FACTORS[1:])]
    factors += [(f"heldout_{i + 1}", f) for i, f in enumerate(HELDOUT)]
    results = []
    for label, factor in factors:
        results.append({
            "label": label, "factor": factor,
            "cognitive_prediction": base.prediction_error(cognitive, factor),
            "transport_policy": base.evaluate(
                transported, states, factor, goal, args.rollout_steps, True,
            ),
            "plain_policy": base.evaluate(
                plain, states, factor, goal, args.rollout_steps, False,
            ),
        })
    changed = HELDOUT[0]
    adapted_cognitive = copy.deepcopy(cognitive)
    before_code = transported.transport_code().detach()
    adaptation_fit = adapt_cognitive(
        adapted_cognitive, changed, args.adapt_steps, args.batch_size, args.seed,
    )
    adapted_policy = copy.deepcopy(transported)
    adapted_policy.cognitive = adapted_cognitive
    after_code = adapted_policy.transport_code().detach()
    output = {
        "architecture": "FullParameterTransportLowRankPolicyV2",
        "training_factor": source,
        "cognitive_parameter_count": sum(p.numel() for p in cognitive.parameters()),
        "runtime_action_probes": "none",
        "cognitive_loss_and_decision_loss_separate": True,
        "cognitive_fit": cognitive_fit,
        "transport_fit": transport_fit,
        "transported_policy_fit": transported_fit,
        "plain_policy_fit": plain_fit,
        "results": results,
        "online_parameter_refresh": {
            "factor": changed,
            "prediction_before": base.prediction_error(cognitive, changed),
            "prediction_after": base.prediction_error(adapted_cognitive, changed),
            "adaptation_fit": adaptation_fit,
            "transport_code_shift_l2": float(torch.linalg.vector_norm(after_code - before_code).item()),
            "decision_before": base.evaluate(
                transported, states, changed, goal, args.rollout_steps, True,
            ),
            "decision_after": base.evaluate(
                adapted_policy, states, changed, goal, args.rollout_steps, True,
            ),
        },
        "test_seed": args.test_seed,
    }
    text = json.dumps(output, ensure_ascii=False, indent=2)
    print(text)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")


if __name__ == "__main__":
    main()
