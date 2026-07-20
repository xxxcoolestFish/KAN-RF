"""Separate learned long-route error from learned local-tube error."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

import torch

from cpbn import OracleAcrobotDynamics
from cpbn.cognition import ProtoKANDynamics
from cpbn.time_varying_tube import TimeVaryingTubeSet, plan_continuous_cem_route
from scripts.validate_learned_cognitive_tubes import execute_hidden_controller
from scripts.validate_learned_cognitive_tubes_v2 import train_cognition


def run(args):
    torch.manual_seed(args.seed)
    source = OracleAcrobotDynamics()
    cognition = ProtoKANDynamics(args.hidden_dim, args.n_prototypes)
    training = train_cognition(cognition, source, args)
    oracle_route = plan_continuous_cem_route(
        source,
        segment_count=args.route_segments,
        segment_steps=args.segment_steps,
        population=args.cem_population,
        elite_count=args.cem_elite,
        iterations=args.cem_iterations,
        seed=args.seed + 200,
    )
    oracle_tubes = TimeVaryingTubeSet(
        source, oracle_route,
        construction_samples=args.construction_samples,
        seed=args.seed + 300,
    )
    learned_local_tubes = TimeVaryingTubeSet(
        cognition, oracle_route,
        construction_samples=args.construction_samples,
        seed=args.seed + 400,
    )
    oracle_baseline = execute_hidden_controller(
        oracle_tubes, source, args.trials_per_edge, args.test_seed,
    )
    learned_predicted = learned_local_tubes.evaluate_hidden_lqr(
        args.trials_per_edge, args.test_seed + 100,
    )
    learned_executed_real = execute_hidden_controller(
        learned_local_tubes, source,
        args.trials_per_edge, args.test_seed + 200,
    )
    return {
        "training_history": training,
        "oracle_route": asdict(oracle_route.diagnostics),
        "oracle_tube_construction": asdict(oracle_tubes.diagnostics),
        "learned_local_tube_construction": asdict(
            learned_local_tubes.diagnostics
        ),
        "oracle_route_oracle_local_dynamics": oracle_baseline,
        "oracle_route_learned_local_dynamics_predicted": learned_predicted,
        "oracle_route_learned_local_dynamics_executed_real": (
            learned_executed_real
        ),
        "learned_local_dynamics_passes_90_percent": (
            learned_executed_real["minimum_edge_completion"] >= 0.90
        ),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--n-prototypes", type=int, default=8)
    parser.add_argument("--cognitive-steps", type=int, default=1600)
    parser.add_argument("--cognitive-batch", type=int, default=128)
    parser.add_argument("--maximum-training-horizon", type=int, default=16)
    parser.add_argument("--secant-batch", type=int, default=128)
    parser.add_argument("--secant-noise", type=float, default=0.025)
    parser.add_argument("--secant-weight", type=float, default=0.5)
    parser.add_argument("--cognitive-lr", type=float, default=2e-3)
    parser.add_argument("--repulsion-weight", type=float, default=1e-5)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--route-segments", type=int, default=20)
    parser.add_argument("--segment-steps", type=int, default=24)
    parser.add_argument("--cem-population", type=int, default=1024)
    parser.add_argument("--cem-elite", type=int, default=64)
    parser.add_argument("--cem-iterations", type=int, default=10)
    parser.add_argument("--construction-samples", type=int, default=512)
    parser.add_argument("--trials-per-edge", type=int, default=64)
    parser.add_argument("--test-seed", type=int, default=20260806)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-out", type=str, default="")
    return parser.parse_args()


def main():
    args = parse_args()
    output = {
        "experiment": "LearnedRouteVsLocalDynamicsDiagnostic",
        "config": vars(args),
        "result": run(args),
    }
    text = json.dumps(output, ensure_ascii=False, indent=2)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
