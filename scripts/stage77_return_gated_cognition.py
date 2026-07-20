"""Diagnostic upper bound for safely deploying online cognition updates.

The cognition model is still optimized only for multistep transition
prediction.  Before a fitted update is deployed into the actor's mandatory
cognitive forward path, paired deterministic target-environment rollouts
check whether its long-horizon behaviour improved.  This is deliberately an
oracle diagnostic, not the final deployable algorithm: its purpose is to test
whether useful cognition updates exist before learning a replay-only gate.
"""

from __future__ import annotations

import argparse
import copy
import json
import math

import torch

from scripts import stage43_standard_ppo_baseline as ppo
from scripts.stage23_multistep_terminal_value import GOAL
from scripts.stage72_joint_online_equivariant_transfer import (
    collect_online_chunk,
)
from scripts.stage75_severe_transfer_attribution import build_actor
from scripts.stage76_delayed_trust_cognition import (
    update_cognition_with_policy_trust,
)


@torch.no_grad()
def deterministic_outcomes(actor, factor, goal, count, steps, seed):
    """Return per-start success and maximum height for a paired policy test."""
    torch.manual_seed(seed)
    state = ppo.reset_down_states(count)
    factor_tensor = torch.tensor(
        factor, dtype=state.dtype,
    ).view(1, 4).expand(count, -1)
    goal_batch = goal.expand(count, -1)
    maximum = torch.full((count,), -float("inf"))
    success = torch.zeros(count, dtype=torch.bool)
    for _ in range(steps):
        action, _, _ = actor.sample(
            state, goal_batch, deterministic=True,
        )
        state = ppo.step(
            state, action, factor_tensor[:, 0], factor_tensor[:, 1],
            factor_tensor[:, 2], factor_tensor[:, 3],
        )
        height = ppo.tip_height(state)
        maximum = torch.maximum(maximum, height)
        success |= height >= 1.0
    return success, maximum


def update_cognition_with_return_gate(
    actor, optimizer, replay, factor, goal, args, iteration,
):
    """Fit by prediction, then accept deployment only after a paired gate."""
    old_parameters = [
        parameter.detach().clone()
        for parameter in actor.cognitive.parameters()
    ]
    old_optimizer = copy.deepcopy(optimizer.state_dict())
    gate_seed = args.gate_seed + iteration
    old_success, old_height = deterministic_outcomes(
        actor, factor, goal, args.gate_count, args.gate_steps, gate_seed,
    )
    fit = update_cognition_with_policy_trust(
        actor, optimizer, replay, args.cognitive_update_steps,
        args.cognitive_batch, args.cognitive_horizon,
        args.cognitive_seed + iteration, args.anchor_count,
        args.max_action_rmse,
    )
    new_success, new_height = deterministic_outcomes(
        actor, factor, goal, args.gate_count, args.gate_steps, gate_seed,
    )

    paired_delta = new_height - old_height
    delta_mean = float(paired_delta.mean())
    delta_sem = float(
        paired_delta.std(unbiased=False) / math.sqrt(args.gate_count)
    )
    lower_confidence = delta_mean - args.gate_beta * delta_sem
    old_count = int(old_success.sum())
    new_count = int(new_success.sum())
    if new_count > old_count:
        accepted = (
            float(new_height.mean())
            >= float(old_height.mean()) - args.success_height_tolerance
        )
        reason = "more_successes" if accepted else "height_safety_reject"
    elif new_count == old_count:
        accepted = lower_confidence >= args.min_height_improvement
        reason = "paired_height_lcb" if accepted else "no_lcb_improvement"
    else:
        accepted = False
        reason = "fewer_successes"

    if not accepted:
        with torch.no_grad():
            for parameter, old in zip(
                actor.cognitive.parameters(), old_parameters,
            ):
                parameter.copy_(old)
        optimizer.load_state_dict(old_optimizer)
    return {
        **fit,
        "gate_old_successes": old_count,
        "gate_new_successes": new_count,
        "gate_old_mean_height": float(old_height.mean()),
        "gate_new_mean_height": float(new_height.mean()),
        "paired_height_delta_mean": delta_mean,
        "paired_height_delta_sem": delta_sem,
        "paired_height_delta_lcb": lower_confidence,
        "accepted": accepted,
        "acceptance_reason": reason,
        "deployed_policy_action_rmse": (
            fit["final_policy_action_rmse"] if accepted else 0.0
        ),
        "gate_real_transitions": 2 * args.gate_count * args.gate_steps,
    }


def run_mode(name, checkpoint, source_config, args, gated_cognition):
    actor = build_actor(source_config)
    actor.load_state_dict(checkpoint["actor"])
    torch.manual_seed(args.critic_seed)
    critic = ppo.ValueCritic(source_config["hidden_dim"])
    for parameter in actor.cognitive.parameters():
        parameter.requires_grad = False
    for parameter in actor.router.parameters():
        parameter.requires_grad = False
    decision_parameters = [
        parameter for parameter_name, parameter in actor.named_parameters()
        if not parameter_name.startswith("cognitive.")
        and not parameter_name.startswith("router.")
    ]
    decision_optimizer = torch.optim.Adam(
        decision_parameters, lr=args.actor_lr,
    )
    critic_optimizer = torch.optim.Adam(
        critic.parameters(), lr=args.critic_lr,
    )
    cognitive_optimizer = None
    if gated_cognition:
        cognitive_optimizer = torch.optim.Adam(
            actor.cognitive.parameters(), lr=args.cognitive_lr,
        )

    factor = tuple(args.target_factor)
    goal = GOAL.view(1, -1)
    before = ppo.evaluate(
        actor, factor, goal, args.test_count, args.eval_steps,
        args.eval_seed,
    )
    torch.manual_seed(args.collection_seed)
    state = ppo.reset_down_states(args.num_envs)
    replay, history = [], []
    cumulative_successes = 0
    cumulative_gate_transitions = 0
    for iteration in range(args.adaptation_iterations):
        rollout, trajectories, state, successes = collect_online_chunk(
            actor, critic, factor, goal, state, args.rollout_horizon,
            args.gamma, args.gae_lambda, args.collection_seed + iteration,
        )
        replay.append(trajectories)
        cumulative_successes += successes
        decision_update = ppo.ppo_update(
            actor, critic, rollout, decision_optimizer, critic_optimizer,
            args.clip_ratio, args.value_coef, args.entropy_coef,
            args.ppo_epochs, args.minibatch, args.update_seed + iteration,
        )
        cognitive_update = None
        if gated_cognition and iteration + 1 >= args.cognition_start:
            cognitive_update = update_cognition_with_return_gate(
                actor, cognitive_optimizer, replay, factor, goal,
                args, iteration,
            )
            cumulative_gate_transitions += cognitive_update[
                "gate_real_transitions"
            ]
        if iteration == 0 or (iteration + 1) % args.eval_every == 0:
            history.append({
                "iteration": iteration + 1,
                "cumulative_policy_transitions": (
                    (iteration + 1) * args.num_envs * args.rollout_horizon
                ),
                "cumulative_gate_transitions": cumulative_gate_transitions,
                "chunk_successes": successes,
                "cumulative_collected_successes": cumulative_successes,
                "policy_std": float(actor.log_std.detach().exp()),
                "decision_update": decision_update,
                "cognitive_update": cognitive_update,
                **ppo.evaluate(
                    actor, factor, goal, args.test_count, args.eval_steps,
                    args.eval_seed,
                ),
            })
    return {
        "name": name,
        "target_before": before,
        "history": history,
        "target_after": ppo.evaluate(
            actor, factor, goal, args.test_count, args.eval_steps,
            args.eval_seed,
        ),
        "total_policy_transitions": (
            args.adaptation_iterations * args.num_envs
            * args.rollout_horizon
        ),
        "total_gate_transitions": cumulative_gate_transitions,
    }


def run(args):
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=True,
    )
    source_config = checkpoint["config"]
    return {
        "decision_only_reset_critic": run_mode(
            "decision_only_reset_critic", checkpoint, source_config,
            args, False,
        ),
        "return_gated_cognition": run_mode(
            "return_gated_cognition", checkpoint, source_config,
            args, True,
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default="results/stage73_source_seed0_checkpoint.pt")
    parser.add_argument("--target-factor", type=float, nargs=4,
                        default=[13.475, 0.06, 0.90, 1.10])
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--rollout-horizon", type=int, default=128)
    parser.add_argument("--adaptation-iterations", type=int, default=15)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--minibatch", type=int, default=256)
    parser.add_argument("--cognition-start", type=int, default=7)
    parser.add_argument("--cognitive-horizon", type=int, default=8)
    parser.add_argument("--cognitive-update-steps", type=int, default=25)
    parser.add_argument("--cognitive-batch", type=int, default=64)
    parser.add_argument("--anchor-count", type=int, default=128)
    parser.add_argument("--max-action-rmse", type=float, default=0.05)
    parser.add_argument("--gate-count", type=int, default=16)
    parser.add_argument("--gate-steps", type=int, default=500)
    parser.add_argument("--gate-beta", type=float, default=1.0)
    parser.add_argument("--min-height-improvement", type=float, default=0.0)
    parser.add_argument("--success-height-tolerance", type=float,
                        default=0.05)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--cognitive-lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--eval-every", type=int, default=3)
    parser.add_argument("--test-count", type=int, default=32)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--eval-seed", type=int, default=20260721)
    parser.add_argument("--collection-seed", type=int, default=20260723)
    parser.add_argument("--update-seed", type=int, default=20260724)
    parser.add_argument("--cognitive-seed", type=int, default=20260725)
    parser.add_argument("--critic-seed", type=int, default=20260727)
    parser.add_argument("--gate-seed", type=int, default=20260729)
    parser.add_argument("--json-out", type=str, default="")
    args = parser.parse_args()
    output = {
        "experiment": "OracleReturnGatedCognitionDiagnostic",
        "diagnostic_only": True,
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
