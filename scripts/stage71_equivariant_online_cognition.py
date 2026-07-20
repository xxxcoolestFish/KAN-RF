"""Online cognition-only transfer for the fixed-semantic causal actor."""

from __future__ import annotations

import argparse
import copy
import json

import torch

from kanrf.protokan_causal_router_stable import (
    StableProtoKANNonlinearEdgeRouter,
)
from scripts import stage43_standard_ppo_baseline as ppo
from scripts.stage23_multistep_terminal_value import GOAL, SimpleCognitiveKAN
from scripts.stage68_online_cognition_drift_audit import (
    free_rollout_metrics,
    one_step_metrics,
    proposal_rollout_metrics,
)
from scripts.stage69_trajectory_complete_online_cognition import (
    collect_complete_trajectories,
    update_multistep,
)
from scripts.stage70_equivariant_causal_update_actor import (
    EquivariantCausalUpdateActor,
)


def build_actor(config):
    cognitive = SimpleCognitiveKAN(hidden_dim=32, n_prototypes=8)
    router = StableProtoKANNonlinearEdgeRouter(delta=config["edge_delta"])
    return EquivariantCausalUpdateActor(
        cognitive, router, config["route_horizon"], config["hidden_dim"],
        config["temperature"], config["route_scale"],
        config["causal_gain"],
    )


@torch.no_grad()
def fixed_interface_drift(source_actor, adapted_actor, count, seed):
    torch.manual_seed(seed)
    state = ppo.reset_down_states(count)
    goal = GOAL.view(1, -1).expand(count, -1)
    source_mean, _, source_routes, _ = source_actor.plan_and_route(state, goal)
    adapted_mean, _, adapted_routes, _ = adapted_actor.plan_and_route(
        state, goal,
    )
    source_action = torch.tanh(source_mean)
    adapted_action = torch.tanh(adapted_mean)
    return {
        "source_first_route_mean": float(source_routes[:, 0].mean()),
        "adapted_first_route_mean": float(adapted_routes[:, 0].mean()),
        "source_first_route_rms": float(
            source_routes[:, 0].square().mean().sqrt()
        ),
        "adapted_first_route_rms": float(
            adapted_routes[:, 0].square().mean().sqrt()
        ),
        "mean_absolute_bounded_action_change": float(
            (adapted_action - source_action).abs().mean()
        ),
        "maximum_absolute_bounded_action_change": float(
            (adapted_action - source_action).abs().max()
        ),
    }


def run(args):
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=True,
    )
    config = checkpoint["config"]
    target_factor = tuple(args.target_factor)
    source_actor = build_actor(config)
    source_actor.load_state_dict(checkpoint["actor"])
    source_actor.eval()
    actor = copy.deepcopy(source_actor)
    for name, parameter in actor.named_parameters():
        parameter.requires_grad = name.startswith("cognitive.")
    optimizer = torch.optim.Adam(
        actor.cognitive.parameters(), lr=args.cognitive_lr,
    )
    goal = GOAL.view(1, -1)
    before = ppo.evaluate(
        actor, target_factor, goal, args.test_count, args.eval_steps,
        args.eval_seed,
    )
    diagnostics_before = {
        "one_step": one_step_metrics(
            actor.cognitive, target_factor, args.audit_batch,
            args.audit_seed,
        ),
        "free_rollout": free_rollout_metrics(
            actor.cognitive, target_factor, args.audit_batch,
            args.multistep_horizon, args.audit_seed + 1,
        ),
        "proposal_rollout": proposal_rollout_metrics(
            actor, target_factor, args.audit_batch, args.audit_seed + 2,
        ),
    }
    replay, history = [], []
    for round_index in range(args.adaptation_rounds):
        trajectories, coverage = collect_complete_trajectories(
            actor, target_factor, args.collection_envs,
            args.collection_steps, args.collection_seed + round_index,
        )
        replay.append(trajectories)
        fit = update_multistep(
            actor.cognitive, optimizer, replay, args.update_steps,
            args.update_batch, args.multistep_horizon,
            args.update_seed + round_index,
        )
        history.append({
            "round": round_index + 1,
            "cumulative_real_transitions": (
                (round_index + 1) * args.collection_envs
                * args.collection_steps
            ),
            "coverage": coverage,
            "cognitive_multistep_fit": fit,
            **ppo.evaluate(
                actor, target_factor, goal, args.test_count,
                args.eval_steps, args.eval_seed,
            ),
        })
    after = ppo.evaluate(
        actor, target_factor, goal, args.test_count, args.eval_steps,
        args.eval_seed,
    )
    diagnostics_after = {
        "one_step": one_step_metrics(
            actor.cognitive, target_factor, args.audit_batch,
            args.audit_seed,
        ),
        "free_rollout": free_rollout_metrics(
            actor.cognitive, target_factor, args.audit_batch,
            args.multistep_horizon, args.audit_seed + 1,
        ),
        "proposal_rollout": proposal_rollout_metrics(
            actor, target_factor, args.audit_batch, args.audit_seed + 2,
        ),
    }
    if args.checkpoint_out:
        torch.save({
            "actor": actor.state_dict(),
            "source_actor": source_actor.state_dict(),
            "target_factor": target_factor,
            "config": vars(args),
            "source_config": config,
        }, args.checkpoint_out)
    return {
        "target_before": before,
        "adaptation_history": history,
        "target_after": after,
        "diagnostics_before": diagnostics_before,
        "diagnostics_after": diagnostics_after,
        "fixed_interface_drift": fixed_interface_drift(
            source_actor, actor, args.audit_batch, args.audit_seed + 3,
        ),
        "decision_updated_on_target": False,
        "router_updated_on_target": False,
        "cognitive_updated_on_target": True,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default="results/stage70_source_seed0_checkpoint.pt")
    parser.add_argument("--target-factor", type=float, nargs=4,
                        default=[9.8, 0.04, 1.1, 0.9])
    parser.add_argument("--adaptation-rounds", type=int, default=3)
    parser.add_argument("--collection-envs", type=int, default=32)
    parser.add_argument("--collection-steps", type=int, default=500)
    parser.add_argument("--multistep-horizon", type=int, default=8)
    parser.add_argument("--update-steps", type=int, default=100)
    parser.add_argument("--update-batch", type=int, default=64)
    parser.add_argument("--cognitive-lr", type=float, default=1e-4)
    parser.add_argument("--test-count", type=int, default=64)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--audit-batch", type=int, default=512)
    parser.add_argument("--eval-seed", type=int, default=20260721)
    parser.add_argument("--audit-seed", type=int, default=20260722)
    parser.add_argument("--collection-seed", type=int, default=20260723)
    parser.add_argument("--update-seed", type=int, default=20260724)
    parser.add_argument("--checkpoint-out", type=str, default="")
    parser.add_argument("--json-out", type=str, default="")
    args = parser.parse_args()
    output = {
        "experiment": "EquivariantCausalActor_OnlineCognitionOnly",
        "config": vars(args),
        "result": run(args),
    }
    text = json.dumps(output, ensure_ascii=False, indent=2)
    print(text)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")


if __name__ == "__main__":
    main()
