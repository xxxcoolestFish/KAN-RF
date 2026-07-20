"""Second source composition attempt using a time-indexed state corridor."""

from __future__ import annotations

import argparse
import copy
import json

import torch

from cpbn import OracleAcrobotDynamics, tip_height
from cpbn.cognition import ProtoKANDynamics
from cpbn.feedback_corridor import plan_feedback_corridor
from cpbn.receding_tube import local_state_distance, variable_riccati_gains
from cpbn.time_varying_tube import plan_continuous_cem_route, tangent_error
from scripts.validate_learned_cognitive_tubes_v2 import train_cognition
from scripts.validate_receding_source_route import make_rehearsal, update_cognition


def reference_window(reference, phase, horizon):
    start = min(reference.shape[0] - 1, phase + 1)
    stop = min(reference.shape[0], start + horizon)
    window = reference[start:stop]
    if window.shape[0] < horizon:
        padding = reference[-1:].expand(horizon - window.shape[0], -1)
        window = torch.cat([window, padding])
    return window


def execute_corridor(
    cognition,
    source,
    reference,
    rehearsal,
    args,
    online_update,
    seed_offset,
):
    state = reference[0].view(1, 6).clone()
    maximum_height = float(tip_height(state))
    success_step = -1
    replans = 0
    trigger_count = 0
    innovations, tracking_errors = [], []
    plan_distances, update_losses = [], []
    checkpoints = []
    recent = []
    optimizer = torch.optim.Adam(
        cognition.parameters(), lr=args.online_cognitive_lr,
    ) if online_update else None

    step_count = 0
    while step_count < args.maximum_execution_steps and success_step < 0:
        phase = min(step_count, reference.shape[0] - 1)
        target_path = reference_window(reference, phase, args.local_horizon)
        plan = plan_feedback_corridor(
            cognition, state.squeeze(0), target_path,
            action_segments=args.local_action_segments,
            population=args.local_population,
            elite_count=args.local_elite,
            iterations=args.local_iterations,
            seed=args.seed + seed_offset + replans,
        )
        gains = variable_riccati_gains(
            cognition, plan.states, plan.actions,
        )
        plan_distances.append(plan.terminal_distance)
        replans += 1
        triggered = False
        for local_step in range(args.execution_chunk):
            error = tangent_error(state, plan.states[local_step].view(1, 6))
            action = (
                plan.actions[local_step].view(1, 1)
                + error @ gains[local_step].T
            ).clamp(-1.0, 1.0)
            with torch.no_grad():
                predicted_next = cognition(state, action)
                real_next = source(state, action)
                innovation = float(
                    tangent_error(predicted_next, real_next).norm(dim=-1)
                )
                tracking = float(local_state_distance(
                    real_next, plan.states[local_step + 1].view(1, 6),
                ))
            recent.append((state.detach(), action.detach(), real_next.detach()))
            recent = recent[-args.recent_buffer:]
            innovations.append(innovation)
            tracking_errors.append(tracking)
            state = real_next
            step_count += 1
            height = float(tip_height(state))
            maximum_height = max(maximum_height, height)
            if height >= 1.0:
                success_step = step_count
                break
            if (
                innovation > args.innovation_threshold
                or tracking > args.tracking_threshold
            ):
                trigger_count += 1
                triggered = True
                break
        if online_update:
            update_loss = update_cognition(
                cognition, optimizer, recent, rehearsal,
                args.online_update_steps, args.online_batch,
                args.seed + 10000 + seed_offset + replans,
            )
            if update_loss is not None:
                update_losses.append(update_loss)
        if replans == 1 or replans % args.checkpoint_every == 0:
            checkpoints.append({
                "replan": replans,
                "step": step_count,
                "phase": min(step_count, reference.shape[0] - 1),
                "maximum_height": maximum_height,
                "plan_terminal_distance": plan.terminal_distance,
                "last_innovation": innovations[-1],
                "last_tracking_error": tracking_errors[-1],
            })
        if triggered:
            continue
    return {
        "online_cognitive_update": online_update,
        "success": success_step >= 0,
        "success_step": success_step,
        "maximum_height": maximum_height,
        "final_phase": min(step_count, reference.shape[0] - 1),
        "replan_count": replans,
        "innovation_trigger_count": trigger_count,
        "mean_prediction_innovation": sum(innovations) / len(innovations),
        "maximum_prediction_innovation": max(innovations),
        "mean_tracking_error": sum(tracking_errors) / len(tracking_errors),
        "mean_local_plan_terminal_distance": (
            sum(plan_distances) / len(plan_distances)
        ),
        "mean_online_update_loss": (
            sum(update_losses) / len(update_losses) if update_losses else None
        ),
        "checkpoints": checkpoints,
        "runtime_reference_actions_used": False,
    }


def run(args):
    torch.manual_seed(args.seed)
    source = OracleAcrobotDynamics()
    cognition = ProtoKANDynamics(args.hidden_dim, args.n_prototypes)
    cognition_training = train_cognition(cognition, source, args)
    construction_route = plan_continuous_cem_route(
        source,
        segment_count=args.reference_segments,
        segment_steps=args.reference_segment_steps,
        population=args.reference_population,
        elite_count=args.reference_elite,
        iterations=args.reference_iterations,
        seed=args.reference_seed,
    )
    reference = construction_route.states.detach().clone()
    rehearsal = make_rehearsal(
        source, args.rehearsal_count, args.seed + 300,
    )
    frozen = execute_corridor(
        copy.deepcopy(cognition), source, reference, rehearsal,
        args, False, 1000,
    )
    online = execute_corridor(
        copy.deepcopy(cognition), source, reference, rehearsal,
        args, True, 2000,
    )
    return {
        "cognition_training": cognition_training,
        "reference_route": {
            "state_count": int(reference.shape[0]),
            "maximum_height": construction_route.diagnostics.maximum_height,
            "success_step": construction_route.diagnostics.success_step,
            "runtime_actions_discarded": True,
        },
        "frozen_cognition": frozen,
        "online_cognition": online,
        "source_feedback_corridor_passed": frozen["success"] or online["success"],
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
    parser.add_argument("--reference-segments", type=int, default=20)
    parser.add_argument("--reference-segment-steps", type=int, default=24)
    parser.add_argument("--reference-population", type=int, default=2048)
    parser.add_argument("--reference-elite", type=int, default=128)
    parser.add_argument("--reference-iterations", type=int, default=12)
    parser.add_argument("--reference-seed", type=int, default=0)
    parser.add_argument("--local-horizon", type=int, default=24)
    parser.add_argument("--local-action-segments", type=int, default=4)
    parser.add_argument("--local-population", type=int, default=256)
    parser.add_argument("--local-elite", type=int, default=32)
    parser.add_argument("--local-iterations", type=int, default=5)
    parser.add_argument("--execution-chunk", type=int, default=4)
    parser.add_argument("--innovation-threshold", type=float, default=0.035)
    parser.add_argument("--tracking-threshold", type=float, default=0.12)
    parser.add_argument("--maximum-execution-steps", type=int, default=600)
    parser.add_argument("--rehearsal-count", type=int, default=4096)
    parser.add_argument("--recent-buffer", type=int, default=512)
    parser.add_argument("--online-update-steps", type=int, default=4)
    parser.add_argument("--online-batch", type=int, default=128)
    parser.add_argument("--online-cognitive-lr", type=float, default=3e-4)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-out", type=str, default="")
    return parser.parse_args()


def main():
    args = parse_args()
    output = {
        "experiment": "TimeIndexedFeedbackCorridorSourceValidation",
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
