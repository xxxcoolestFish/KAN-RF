"""Delayed, policy-trust-region cognition adaptation on the severe target."""

from __future__ import annotations

import argparse
import json

import torch

from scripts import stage43_standard_ppo_baseline as ppo
from scripts.stage23_multistep_terminal_value import GOAL
from scripts.stage72_joint_online_equivariant_transfer import (
    collect_online_chunk,
    update_cognitive_sequences,
)
from scripts.stage75_severe_transfer_attribution import build_actor


@torch.no_grad()
def stratified_anchor_states(replay, count):
    states = torch.cat([item[0].reshape(-1, 6) for item in replay])
    heights = ppo.tip_height(states)
    order = torch.argsort(heights)
    if order.numel() <= count:
        return states[order]
    positions = torch.linspace(
        0, order.numel() - 1, count,
    ).round().long()
    return states[order[positions]]


def update_cognition_with_policy_trust(
    actor, optimizer, replay, updates, batch_size, horizon, seed,
    anchor_count, max_action_rmse,
):
    """Project one cognition update block into a composed-policy trust region."""
    anchor_state = stratified_anchor_states(replay, anchor_count)
    anchor_goal = GOAL.view(1, -1).expand(anchor_state.shape[0], -1)
    with torch.no_grad():
        reference_action, _, _ = actor.sample(
            anchor_state, anchor_goal, deterministic=True,
        )
        old_parameters = [
            parameter.detach().clone()
            for parameter in actor.cognitive.parameters()
        ]
    fit = update_cognitive_sequences(
        actor.cognitive, optimizer, replay, updates, batch_size,
        horizon, seed,
    )
    with torch.no_grad():
        raw_action, _, _ = actor.sample(
            anchor_state, anchor_goal, deterministic=True,
        )
        raw_rmse = float(
            (raw_action - reference_action).square().mean().sqrt()
        )
        new_parameters = [
            parameter.detach().clone()
            for parameter in actor.cognitive.parameters()
        ]
        projection_scale = 1.0
        if raw_rmse > max_action_rmse:
            projection_scale = max_action_rmse / max(raw_rmse, 1e-8)
            for _ in range(8):
                for parameter, old, new in zip(
                    actor.cognitive.parameters(), old_parameters,
                    new_parameters,
                ):
                    parameter.copy_(old + projection_scale * (new - old))
                projected_action, _, _ = actor.sample(
                    anchor_state, anchor_goal, deterministic=True,
                )
                projected_rmse = float(
                    (projected_action - reference_action)
                    .square().mean().sqrt()
                )
                if projected_rmse <= max_action_rmse * 1.02:
                    break
                projection_scale *= 0.5
            # Adam moments correspond to the unprojected step and would push
            # straight back outside the trust region on the next iteration.
            optimizer.state.clear()
        final_action, _, _ = actor.sample(
            anchor_state, anchor_goal, deterministic=True,
        )
        final_rmse = float(
            (final_action - reference_action).square().mean().sqrt()
        )
    return {
        **fit,
        "anchor_count": int(anchor_state.shape[0]),
        "raw_policy_action_rmse": raw_rmse,
        "projection_scale": projection_scale,
        "final_policy_action_rmse": final_rmse,
    }


def run_mode(name, checkpoint, source_config, args, trusted_cognition):
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
    if trusted_cognition:
        cognitive_optimizer = torch.optim.Adam(
            actor.cognitive.parameters(), lr=args.cognitive_lr,
        )

    target_factor = tuple(args.target_factor)
    goal = GOAL.view(1, -1)
    before = ppo.evaluate(
        actor, target_factor, goal, args.test_count, args.eval_steps,
        args.eval_seed,
    )
    torch.manual_seed(args.collection_seed)
    state = ppo.reset_down_states(args.num_envs)
    replay, history = [], []
    cumulative_successes = 0
    for iteration in range(args.adaptation_iterations):
        rollout, trajectories, state, successes = collect_online_chunk(
            actor, critic, target_factor, goal, state,
            args.rollout_horizon, args.gamma, args.gae_lambda,
            args.collection_seed + iteration,
        )
        replay.append(trajectories)
        cumulative_successes += successes
        decision_update = ppo.ppo_update(
            actor, critic, rollout, decision_optimizer, critic_optimizer,
            args.clip_ratio, args.value_coef, args.entropy_coef,
            args.ppo_epochs, args.minibatch,
            args.update_seed + iteration,
        )
        cognitive_update = None
        if trusted_cognition and iteration + 1 >= args.cognition_start:
            cognitive_update = update_cognition_with_policy_trust(
                actor, cognitive_optimizer, replay,
                args.cognitive_update_steps, args.cognitive_batch,
                args.cognitive_horizon, args.cognitive_seed + iteration,
                args.anchor_count, args.max_action_rmse,
            )
        if iteration == 0 or (iteration + 1) % args.eval_every == 0:
            history.append({
                "iteration": iteration + 1,
                "cumulative_real_transitions": (
                    (iteration + 1) * args.num_envs
                    * args.rollout_horizon
                ),
                "chunk_successes": successes,
                "cumulative_collected_successes": cumulative_successes,
                "policy_std": float(actor.log_std.detach().exp()),
                "decision_update": decision_update,
                "cognitive_update": cognitive_update,
                **ppo.evaluate(
                    actor, target_factor, goal, args.test_count,
                    args.eval_steps, args.eval_seed,
                ),
            })
    return {
        "name": name,
        "target_before": before,
        "history": history,
        "target_after": ppo.evaluate(
            actor, target_factor, goal, args.test_count, args.eval_steps,
            args.eval_seed,
        ),
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
        "delayed_trusted_joint": run_mode(
            "delayed_trusted_joint", checkpoint, source_config,
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
    parser.add_argument("--json-out", type=str, default="")
    args = parser.parse_args()
    output = {
        "experiment": "DelayedPolicyTrustRegionCognitionOnSevereTarget",
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
