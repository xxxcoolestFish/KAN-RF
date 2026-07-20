"""Trajectory-complete online cognition adaptation for the causal actor.

This starts from the Stage 67 source decision policy reconstructed from its
checkpoint, then adapts cognition only.  Unlike Stage 67, each update round is
fed full 500-step continuous target trajectories and is trained on contiguous
eight-step free rollouts instead of shuffled one-step records.
"""

from __future__ import annotations

import argparse
import copy
import json

import torch
import torch.nn.functional as F

from scripts import stage43_standard_ppo_baseline as ppo
from scripts.stage23_multistep_terminal_value import GOAL, SimpleCognitiveKAN
from scripts.stage27_parameter_transport import pretrain_cognitive
from scripts.stage68_online_cognition_drift_audit import (
    build_actor,
    free_rollout_metrics,
    interface_drift,
    one_step_metrics,
    proposal_rollout_metrics,
)


@torch.no_grad()
def collect_complete_trajectories(actor, factor, count, steps, seed):
    """Collect continuous real trajectories without success resets."""
    torch.manual_seed(seed)
    state = ppo.reset_down_states(count)
    factor_tensor = torch.tensor(factor).view(1, 4).expand(count, -1)
    goal = GOAL.view(1, -1).expand(count, -1)
    states, actions, next_states = [], [], []
    for _ in range(steps):
        action, _, _ = actor.sample(state, goal, deterministic=False)
        next_state = ppo.step(
            state, action, factor_tensor[:, 0], factor_tensor[:, 1],
            factor_tensor[:, 2], factor_tensor[:, 3],
        )
        states.append(state)
        actions.append(action)
        next_states.append(next_state)
        state = next_state
    states = torch.stack(states)
    actions = torch.stack(actions)
    next_states = torch.stack(next_states)
    heights = ppo.tip_height(states.reshape(-1, 6))
    return (states, actions, next_states), {
        "transition_count": int(steps * count),
        "height_mean": float(heights.mean()),
        "height_max": float(heights.max()),
        "fraction_height_above_zero": float((heights > 0.0).float().mean()),
        "action_mean": float(actions.mean()),
        "action_std": float(actions.std()),
        "action_min": float(actions.min()),
        "action_max": float(actions.max()),
    }


def update_multistep(cognitive, optimizer, replay, updates, batch_size,
                     horizon, seed):
    """Fit free-running predictions on contiguous windows from all rounds."""
    for parameter in cognitive.parameters():
        parameter.requires_grad = True
    states = torch.stack([item[0] for item in replay])
    actions = torch.stack([item[1] for item in replay])
    next_states = torch.stack([item[2] for item in replay])
    rounds, steps, envs = states.shape[:3]
    generator = torch.Generator().manual_seed(seed)
    losses = []
    for _ in range(updates):
        round_index = torch.randint(
            rounds, (batch_size,), generator=generator,
        )
        start = torch.randint(
            steps - horizon + 1, (batch_size,), generator=generator,
        )
        env_index = torch.randint(
            envs, (batch_size,), generator=generator,
        )
        prediction = states[round_index, start, env_index]
        horizon_losses = []
        for offset in range(horizon):
            index = start + offset
            action = actions[round_index, index, env_index]
            target = next_states[round_index, index, env_index]
            prediction = cognitive(prediction, action)
            horizon_losses.append(F.smooth_l1_loss(prediction, target))
        loss = torch.stack(horizon_losses).mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(cognitive.parameters(), 5.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    for parameter in cognitive.parameters():
        parameter.requires_grad = False
    return {
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "mean_last_20_loss": sum(losses[-20:]) / min(20, len(losses)),
    }


def run(args):
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=True,
    )
    config = checkpoint["config"]
    target_factor = tuple(args.target_factor or checkpoint["target_factor"])

    # Rebuild the exact cognition frozen during source PPO.
    torch.manual_seed(config["seed"])
    source_cognitive = SimpleCognitiveKAN(hidden_dim=32, n_prototypes=8)
    source_fit = pretrain_cognitive(
        source_cognitive, config["cognitive_steps"],
        config["cognitive_batch"], config["seed"],
    )
    source_actor = build_actor(config)
    source_actor.load_state_dict(checkpoint["actor"])
    source_actor.cognitive.load_state_dict(source_cognitive.state_dict())
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
    broad_before = {
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

    replay = []
    history = []
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
        evaluation = ppo.evaluate(
            actor, target_factor, goal, args.test_count, args.eval_steps,
            args.eval_seed,
        )
        history.append({
            "round": round_index + 1,
            "cumulative_real_transitions": (
                (round_index + 1) * args.collection_envs
                * args.collection_steps
            ),
            "coverage": coverage,
            "cognitive_multistep_fit": fit,
            **evaluation,
        })

    after = ppo.evaluate(
        actor, target_factor, goal, args.test_count, args.eval_steps,
        args.eval_seed,
    )
    broad_after = {
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
    drift = interface_drift(
        source_actor, actor, args.audit_batch, args.audit_seed + 3,
    )
    if args.checkpoint_out:
        torch.save({
            "actor": actor.state_dict(),
            "source_actor": source_actor.state_dict(),
            "target_factor": target_factor,
            "config": vars(args),
            "source_config": config,
        }, args.checkpoint_out)
    return {
        "source_cognitive_reconstruction_fit": source_fit,
        "target_before": before,
        "target_adaptation_history": history,
        "target_after": after,
        "broad_diagnostics_before": broad_before,
        "broad_diagnostics_after": broad_after,
        "interface_drift": drift,
        "decision_updated_on_target": False,
        "router_updated_on_target": False,
        "cognitive_updated_on_target": True,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default="results/stage67_seed0_checkpoint.pt")
    parser.add_argument("--target-factor", type=float, nargs=4, default=None)
    parser.add_argument("--adaptation-rounds", type=int, default=5)
    parser.add_argument("--collection-envs", type=int, default=32)
    parser.add_argument("--collection-steps", type=int, default=500)
    parser.add_argument("--multistep-horizon", type=int, default=8)
    parser.add_argument("--update-steps", type=int, default=100)
    parser.add_argument("--update-batch", type=int, default=64)
    parser.add_argument("--cognitive-lr", type=float, default=5e-4)
    parser.add_argument("--test-count", type=int, default=32)
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
        "experiment": "TrajectoryCompleteMultistepOnlineCognition",
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
