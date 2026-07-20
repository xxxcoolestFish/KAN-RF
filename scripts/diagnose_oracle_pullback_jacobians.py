"""Measure whether Oracle cognitive Jacobians identify physics changes."""

from __future__ import annotations

import argparse
import json

import torch

from cpbn import OracleAcrobotDynamics, SOURCE_FACTOR
from cpbn.cognitive_pullback import local_jacobians_batch
from cpbn.time_varying_tube import plan_continuous_cem_route


FACTORS = {
    "source": SOURCE_FACTOR,
    "strong_gravity": (9.81, 0.0, 0.8, 0.8),
    "damped": (7.35, 0.15, 0.8, 0.8),
    "weak_actuator": (7.35, 0.0, 0.55, 0.8),
    "heavy_inertia": (7.35, 0.0, 0.8, 1.10),
    "combined_shift": (9.81, 0.10, 0.60, 1.05),
}


def relative_difference(target, source):
    numerator = (target - source).flatten(1).norm(dim=1)
    denominator = source.flatten(1).norm(dim=1).clamp_min(1e-8)
    return numerator / denominator


def run(args):
    source_dynamics = OracleAcrobotDynamics()
    route = plan_continuous_cem_route(
        source_dynamics,
        population=args.population,
        elite_count=args.elite,
        iterations=args.iterations,
        seed=args.seed,
    )
    centers = route.states[::args.stride]
    generator = torch.Generator().manual_seed(args.seed + 1000)
    costate = torch.randn(centers.shape[0], 4, generator=generator)
    costate = costate / costate.norm(dim=1, keepdim=True).clamp_min(1e-8)
    jacobians = {}
    for name, factor in FACTORS.items():
        jacobians[name] = local_jacobians_batch(
            OracleAcrobotDynamics(factor), centers,
        )
    source_a, source_b = jacobians["source"]

    def direct_action(action_jacobian):
        return torch.tanh(
            -args.pullback_scale * torch.bmm(
                action_jacobian.transpose(1, 2), costate.unsqueeze(-1),
            ).squeeze(-1)
        )

    def adjoint_action(state_jacobian, action_jacobian):
        propagated = costate
        for _ in range(args.adjoint_horizon):
            propagated = torch.bmm(
                state_jacobian.transpose(1, 2), propagated.unsqueeze(-1),
            ).squeeze(-1)
        return torch.tanh(
            -args.pullback_scale * torch.bmm(
                action_jacobian.transpose(1, 2), propagated.unsqueeze(-1),
            ).squeeze(-1)
        )

    source_action = direct_action(source_b)
    source_adjoint_action = adjoint_action(source_a, source_b)
    comparisons = {}
    identity = torch.eye(4).view(1, 4, 4)
    for name, factor in FACTORS.items():
        state_jacobian, action_jacobian = jacobians[name]
        action = direct_action(action_jacobian)
        propagated_action = adjoint_action(state_jacobian, action_jacobian)
        cosine = torch.nn.functional.cosine_similarity(
            source_b.squeeze(-1), action_jacobian.squeeze(-1), dim=1,
        )
        comparisons[name] = {
            "factor": list(factor),
            "mean_relative_A_change": float(
                relative_difference(state_jacobian, source_a).mean()
            ),
            "mean_relative_centered_A_change": float(relative_difference(
                state_jacobian - identity, source_a - identity,
            ).mean()),
            "mean_relative_B_change": float(
                relative_difference(action_jacobian, source_b).mean()
            ),
            "mean_B_norm": float(action_jacobian.flatten(1).norm(dim=1).mean()),
            "minimum_B_norm": float(action_jacobian.flatten(1).norm(dim=1).min()),
            "mean_B_direction_cosine_to_source": float(cosine.mean()),
            "mean_direct_pullback_action_change": float(
                (action - source_action).abs().mean()
            ),
            "mean_adjoint_pullback_action_change": float(
                (propagated_action - source_adjoint_action).abs().mean()
            ),
        }
    return {
        "experiment": "OraclePullbackJacobianDiagnostic",
        "sampled_route_states": int(centers.shape[0]),
        "route_stride": args.stride,
        "action_anchor": 0.0,
        "pullback_scale": args.pullback_scale,
        "adjoint_horizon": args.adjoint_horizon,
        "comparisons": comparisons,
        "passed": (
            comparisons["weak_actuator"]["mean_relative_B_change"] > 0.10
            and comparisons["strong_gravity"][
                "mean_adjoint_pullback_action_change"
            ] > 0.01
            and comparisons["combined_shift"][
                "mean_direct_pullback_action_change"
            ] > 0.01
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--population", type=int, default=2048)
    parser.add_argument("--elite", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--pullback-scale", type=float, default=80.0)
    parser.add_argument("--adjoint-horizon", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--json-out", default="results/oracle_pullback_jacobian_seed0.json",
    )
    args = parser.parse_args()
    output = run(args)
    text = json.dumps(output, ensure_ascii=False, indent=2)
    with open(args.json_out, "w", encoding="utf-8") as handle:
        handle.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
