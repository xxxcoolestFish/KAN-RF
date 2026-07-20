"""Validate feedback composition of short cognitive tubes on the source task."""

from __future__ import annotations

import argparse
import copy
import json

import torch

from cpbn import OracleAcrobotDynamics, random_states, tip_height
from cpbn.cognition import ProtoKANDynamics
from cpbn.receding_tube import (
    local_state_distance,
    nearest_reference_progress,
    plan_local_cem,
    variable_riccati_gains,
)
from cpbn.time_varying_tube import plan_continuous_cem_route, tangent_error
from scripts.validate_learned_cognitive_tubes_v2 import train_cognition


def make_rehearsal(source, count, seed):
    generator = torch.Generator().manual_seed(seed)
    state = random_states(count, generator)
    action = torch.rand(count, 1, generator=generator) * 2.0 - 1.0
    with torch.no_grad():
        next_state = source(state, action)
    return state, action, next_state


def update_cognition(
    cognition,
    optimizer,
    recent,
    rehearsal,
    update_steps,
    batch_size,
    seed,
):
    if not recent or update_steps <= 0:
        return None
    generator = torch.Generator().manual_seed(seed)
    recent_state = torch.cat([item[0] for item in recent])
    recent_action = torch.cat([item[1] for item in recent])
    recent_next = torch.cat([item[2] for item in recent])
    anchor_state, anchor_action, anchor_next = rehearsal
    losses = []
    recent_count = max(1, int(batch_size * 0.75))
    anchor_count = batch_size - recent_count
    for _ in range(update_steps):
        recent_index = torch.randint(
            recent_state.shape[0], (recent_count,), generator=generator,
        )
        anchor_index = torch.randint(
            anchor_state.shape[0], (anchor_count,), generator=generator,
        )
        state = torch.cat([
            recent_state[recent_index], anchor_state[anchor_index],
        ])
        action = torch.cat([
            recent_action[recent_index], anchor_action[anchor_index],
        ])
        next_state = torch.cat([
            recent_next[recent_index], anchor_next[anchor_index],
        ])
        loss = cognition.prediction_loss(state, action, next_state)
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(cognition.parameters(), 5.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    return sum(losses) / len(losses)


def execute_feedback_route(
    cognition,
    source,
    reference,
    rehearsal,
    args,
    online_update,
    seed_offset,
):
    state = reference[0].view(1, 6).clone()
    progress = 0
    maximum_height = float(tip_height(state))
    success_step = -1
    replans = 0
    innovation_triggers = 0
    innovations = []
    terminal_plan_distances = []
    tracking_errors = []
    update_losses = []
    recent = []
    optimizer = torch.optim.Adam(
        cognition.parameters(), lr=args.online_cognitive_lr,
    ) if online_update else None

    step_count = 0
    while step_count < args.maximum_execution_steps and success_step < 0:
        target_index = min(
            reference.shape[0] - 1, progress + args.lookahead_steps,
        )
        target = reference[target_index]
        plan = plan_local_cem(
            cognition, state.squeeze(0), target,
            horizon=args.local_horizon,
            action_segments=args.local_action_segments,
            population=args.local_population,
            elite_count=args.local_elite,
            iterations=args.local_iterations,
            seed=args.seed + seed_offset + replans,
        )
        gains = variable_riccati_gains(
            cognition, plan.states, plan.actions,
        )
        terminal_plan_distances.append(plan.terminal_distance)
        replans += 1
        trigger_replan = False
        for local_step in range(args.execution_chunk):
            error = tangent_error(state, plan.states[local_step].view(1, 6))
            correction = error @ gains[local_step].T
            action = (
                plan.actions[local_step].view(1, 1) + correction
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
            progress = nearest_reference_progress(
                state.squeeze(0), reference, progress,
                args.reference_search_window,
            )
            height = float(tip_height(state))
            maximum_height = max(maximum_height, height)
            if height >= 1.0:
                success_step = step_count
                break
            if (
                innovation > args.innovation_threshold
                or tracking > args.tracking_threshold
            ):
                innovation_triggers += 1
                trigger_replan = True
                break
        if online_update:
            update_loss = update_cognition(
                cognition, optimizer, recent, rehearsal,
                args.online_update_steps, args.online_batch,
                args.seed + 10000 + seed_offset + replans,
            )
            if update_loss is not None:
                update_losses.append(update_loss)
        if trigger_replan:
            continue
    return {
        "online_cognitive_update": online_update,
        "success": success_step >= 0,
        "success_step": success_step,
        "maximum_height": maximum_height,
        "final_reference_progress": progress,
        "reference_length": int(reference.shape[0]),
        "replan_count": replans,
        "innovation_trigger_count": innovation_triggers,
        "mean_prediction_innovation": (
            sum(innovations) / len(innovations) if innovations else None
        ),
        "maximum_prediction_innovation": max(innovations) if innovations else None,
        "mean_tracking_error": (
            sum(tracking_errors) / len(tracking_errors)
            if tracking_errors else None
        ),
        "mean_local_plan_terminal_distance": (
            sum(terminal_plan_distances) / len(terminal_plan_distances)
        ),
        "mean_online_update_loss": (
            sum(update_losses) / len(update_losses) if update_losses else None
        ),
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
        seed=args.seed + 200,
    )
    # Runtime receives only the state path. Construction actions are discarded.
    reference = construction_route.states.detach().clone()
    rehearsal = make_rehearsal(
        source, args.rehearsal_count, args.seed + 300,
    )
    frozen = execute_feedback_route(
        copy.deepcopy(cognition), source, reference, rehearsal,
        args, online_update=False, seed_offset=1000,
    )
    online = execute_feedback_route(
        copy.deepcopy(cognition), source, reference, rehearsal,
        args, online_update=True, seed_offset=2000,
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
        "source_feedback_route_passed": frozen["success"] or online["success"],
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
    parser.add_argument("--local-horizon", type=int, default=24)
    parser.add_argument("--local-action-segments", type=int, default=4)
    parser.add_argument("--local-population", type=int, default=256)
    parser.add_argument("--local-elite", type=int, default=32)
    parser.add_argument("--local-iterations", type=int, default=5)
    parser.add_argument("--execution-chunk", type=int, default=4)
    parser.add_argument("--lookahead-steps", type=int, default=24)
    parser.add_argument("--reference-search-window", type=int, default=48)
    parser.add_argument("--innovation-threshold", type=float, default=0.035)
    parser.add_argument("--tracking-threshold", type=float, default=0.12)
    parser.add_argument("--maximum-execution-steps", type=int, default=600)
    parser.add_argument("--rehearsal-count", type=int, default=4096)
    parser.add_argument("--recent-buffer", type=int, default=512)
    parser.add_argument("--online-update-steps", type=int, default=4)
    parser.add_argument("--online-batch", type=int, default=128)
    parser.add_argument("--online-cognitive-lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-out", type=str, default="")
    return parser.parse_args()


def main():
    args = parse_args()
    output = {
        "experiment": "RecedingCognitiveTubeSourceValidation",
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
