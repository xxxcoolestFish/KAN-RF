"""Diagnose target solvability and real-feedback online policy adaptation."""

from __future__ import annotations

import argparse
import copy
import json
from argparse import Namespace

import torch

from cpbn import OracleAcrobotDynamics, tip_height
from cpbn.corridor_policy import (
    CorridorCritic,
    DirectCorridorActor,
    future_corridor,
)
from cpbn.feedback_phase import bounded_nearest_phase
from cpbn.receding_tube import local_state_distance
from cpbn.time_varying_tube import plan_continuous_cem_route
from scripts.validate_direct_corridor_actor import (
    Rollout,
    ppo_update,
    reset_batch,
)
from scripts.validate_feedback_phase_actor import aggregate, evaluate_mode


HEAVY_INERTIA_FACTOR = (7.35, 0.0, 0.80, 1.20)


def collect_feedback_rollout(
    actor, critic, dynamics, reference, args, seed,
):
    """Collect real target transitions with feedback-aligned route phase."""
    generator = torch.Generator().manual_seed(seed)
    state, phase = reset_batch(
        reference, args.num_envs, generator, args.initial_noise,
    )
    states, phases, actions, log_probs = [], [], [], []
    values, rewards, terminals = [], [], []
    successes = 0
    for _ in range(args.rollout_horizon):
        with torch.no_grad():
            corridor = future_corridor(
                reference, phase, args.corridor_horizon,
            )
            action, log_prob = actor.sample(state, corridor)
            value = critic(state, corridor)
            before = local_state_distance(state, reference[phase])
            next_state = dynamics(state, action)
            next_phase = bounded_nearest_phase(
                next_state, reference, phase,
                backtrack=args.phase_backtrack,
                advance=args.phase_advance,
            )
            after = local_state_distance(next_state, reference[next_phase])
            task_success = tip_height(next_state) >= 1.0
            route_end = next_phase >= reference.shape[0] - 1
            terminal = task_success | route_end
            distance_progress = (before - after).clamp(
                -args.progress_clip, args.progress_clip,
            )
            phase_delta = (next_phase - phase).float()
            phase_progress = phase_delta.clamp(-2.0, 2.0)
            inside = after <= args.corridor_radius
            stagnated = phase_delta <= 0
            reward = args.progress_reward * distance_progress
            reward = reward + args.phase_progress_reward * phase_progress
            reward = reward + args.inside_reward * inside.float()
            reward = reward - args.stagnation_penalty * stagnated.float()
            reward = reward + args.success_reward * task_success.float()
            reward = reward - args.action_penalty * action.square().squeeze(-1)
        successes += int(task_success.sum())
        states.append(state); phases.append(phase); actions.append(action)
        log_probs.append(log_prob); values.append(value)
        rewards.append(reward); terminals.append(terminal)
        reset_state, reset_phase = reset_batch(
            reference, args.num_envs, generator, args.initial_noise,
        )
        state = torch.where(terminal.unsqueeze(-1), reset_state, next_state)
        phase = torch.where(terminal, reset_phase, next_phase)

    with torch.no_grad():
        corridor = future_corridor(reference, phase, args.corridor_horizon)
        last_value = critic(state, corridor)
    states = torch.stack(states); phases = torch.stack(phases)
    actions = torch.stack(actions); log_probs = torch.stack(log_probs)
    values = torch.stack(values); rewards = torch.stack(rewards)
    terminals = torch.stack(terminals)
    advantage = torch.zeros_like(rewards)
    gae = torch.zeros(args.num_envs)
    next_value = last_value
    for index in reversed(range(args.rollout_horizon)):
        nonterminal = 1.0 - terminals[index].float()
        delta = (
            rewards[index] + args.gamma * next_value * nonterminal
            - values[index]
        )
        gae = (
            delta + args.gamma * args.gae_lambda * nonterminal * gae
        )
        advantage[index] = gae
        next_value = values[index]
    returns = advantage + values
    return Rollout(
        states.reshape(-1, 6), phases.reshape(-1),
        actions.reshape(-1, 1), log_probs.reshape(-1),
        advantage.reshape(-1), returns.reshape(-1),
    ), {
        "collected_successes": successes,
        "mean_reward": float(rewards.mean()),
        "return_target_std": float(returns.std(unbiased=False)),
        "mean_phase_delta": float(
            (phases[1:] - phases[:-1]).float().mean()
        ),
    }


def evaluation_cli(args, count):
    return Namespace(
        count=count,
        steps=args.evaluation_steps,
        nearest_backtrack=args.phase_backtrack,
        nearest_advance=args.phase_advance,
        observation_scale=0.10,
    )


def evaluate_once(actor, dynamics, reference, config, args, count, seed):
    return evaluate_mode(
        actor, dynamics, reference, config,
        evaluation_cli(args, count), "nearest", seed,
    )


def aggregate_actor(
    actor, dynamics, reference, config, args, count, seed_offset,
):
    records = []
    for index in range(args.num_test_seeds):
        record = evaluate_once(
            actor, dynamics, reference, config, args, count,
            args.test_seed + seed_offset + index * 1009,
        )
        record["evaluation_count"] = count
        records.append(record)
    return aggregate(records)


def train_condition(
    name, initial_actor, dynamics, reference, config, args, actor_lr, seed_offset,
):
    actor = copy.deepcopy(initial_actor)
    critic = CorridorCritic(config.hidden_dim)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=actor_lr)
    critic_optimizer = torch.optim.Adam(
        critic.parameters(), lr=args.critic_lr,
    )
    history = []
    best_score = -float("inf")
    best_iteration = 0
    best_actor = copy.deepcopy(actor.state_dict())
    for iteration in range(args.iterations):
        rollout, collection = collect_feedback_rollout(
            actor, critic, dynamics, reference, args,
            args.seed + seed_offset + iteration,
        )
        update = ppo_update(
            actor, critic, reference, rollout,
            actor_optimizer, critic_optimizer, args,
            args.seed + 10000 + seed_offset + iteration,
        )
        if iteration == 0 or (iteration + 1) % args.eval_every == 0:
            evaluation = evaluate_once(
                actor, dynamics, reference, config, args,
                args.eval_count,
                args.test_seed + seed_offset + iteration,
            )
            record = {
                "iteration": iteration + 1,
                **collection,
                **update,
                "success_rate": evaluation["success_rate"],
                "mean_maximum_height": evaluation["mean_maximum_height"],
                "mean_final_phase": evaluation["mean_final_phase"],
            }
            history.append(record)
            print(json.dumps({"condition": name, **record}), flush=True)
            score = (
                10.0 * evaluation["success_rate"]
                + evaluation["mean_maximum_height"]
            )
            if score > best_score:
                best_score = score
                best_iteration = iteration + 1
                best_actor = copy.deepcopy(actor.state_dict())
    actor.load_state_dict(best_actor)
    final = aggregate_actor(
        actor, dynamics, reference, config, args,
        args.final_count, seed_offset + 50000,
    )
    return actor, {
        "history": history,
        "best_iteration": best_iteration,
        "final": final,
    }


def parameter_displacement(source_actor, adapted_actor):
    source = dict(source_actor.named_parameters())
    adapted = dict(adapted_actor.named_parameters())
    block_norms = {}
    total = torch.zeros(())
    base = torch.zeros(())
    for name in source:
        if name == "log_std":
            continue
        delta = adapted[name].detach() - source[name].detach()
        block_norms[name] = float(delta.norm())
        total = total + delta.square().sum()
        base = base + source[name].detach().square().sum()
    return {
        "total_norm": float(total.sqrt()),
        "relative_norm": float(total.sqrt() / base.sqrt().clamp_min(1e-12)),
        "parameter_block_norms": block_norms,
    }


def run(args):
    torch.manual_seed(args.seed)
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False,
    )
    config = Namespace(**checkpoint["config"])
    source_reference = checkpoint["reference"].detach().clone()
    source_actor = DirectCorridorActor(config.hidden_dim, config.log_std_init)
    source_actor.load_state_dict(checkpoint["actor"])
    target_dynamics = OracleAcrobotDynamics(HEAVY_INERTIA_FACTOR)
    target_construction = plan_continuous_cem_route(
        target_dynamics,
        segment_count=args.reference_segments,
        segment_steps=args.segment_steps,
        population=args.reference_population,
        elite_count=args.reference_elite,
        iterations=args.reference_iterations,
        seed=args.reference_seed,
    )
    target_reference = target_construction.states.detach().clone()
    scratch_actor = DirectCorridorActor(config.hidden_dim, config.log_std_init)

    initial = {
        "source_actor_source_route": aggregate_actor(
            source_actor, target_dynamics, source_reference, config, args,
            args.final_count, 1000,
        ),
        "source_actor_target_route": aggregate_actor(
            source_actor, target_dynamics, target_reference, config, args,
            args.final_count, 2000,
        ),
        "scratch_actor_target_route": aggregate_actor(
            scratch_actor, target_dynamics, target_reference, config, args,
            args.final_count, 3000,
        ),
    }
    source_adapted, source_result = train_condition(
        "source_actor_source_route", source_actor,
        target_dynamics, source_reference, config, args,
        args.adapt_actor_lr, 100000,
    )
    target_adapted, target_result = train_condition(
        "source_actor_target_route", source_actor,
        target_dynamics, target_reference, config, args,
        args.adapt_actor_lr, 200000,
    )
    scratch_adapted, scratch_result = train_condition(
        "scratch_actor_target_route", scratch_actor,
        target_dynamics, target_reference, config, args,
        args.scratch_actor_lr, 300000,
    )
    result = {
        "target_factor": HEAVY_INERTIA_FACTOR,
        "target_reference": {
            "state_count": int(target_reference.shape[0]),
            "maximum_height": target_construction.diagnostics.maximum_height,
            "success_step": target_construction.diagnostics.success_step,
            "actions_exposed_to_actor": False,
        },
        "initial": initial,
        "conditions": {
            "source_actor_source_route": source_result,
            "source_actor_target_route": target_result,
            "scratch_actor_target_route": scratch_result,
        },
        "source_initialized_parameter_displacement": {
            "source_route": parameter_displacement(
                source_actor, source_adapted,
            ),
            "target_route": parameter_displacement(
                source_actor, target_adapted,
            ),
        },
    }
    result["summary"] = {
        "target_route_is_physically_solved": (
            target_construction.diagnostics.success_step >= 0
        ),
        "source_route_online_success_rate": source_result["final"][
            "success_rate"
        ],
        "target_route_online_success_rate": target_result["final"][
            "success_rate"
        ],
        "scratch_target_success_rate": scratch_result["final"][
            "success_rate"
        ],
    }
    if args.checkpoint_out:
        torch.save({
            "source_route_actor": source_adapted.state_dict(),
            "target_route_actor": target_adapted.state_dict(),
            "scratch_target_actor": scratch_adapted.state_dict(),
            "source_reference": source_reference,
            "target_reference": target_reference,
            "config": vars(args),
        }, args.checkpoint_out)
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", default="results/direct_corridor_actor_strong_seed0.pt",
    )
    parser.add_argument("--reference-segments", type=int, default=30)
    parser.add_argument("--segment-steps", type=int, default=24)
    parser.add_argument("--reference-population", type=int, default=4096)
    parser.add_argument("--reference-elite", type=int, default=256)
    parser.add_argument("--reference-iterations", type=int, default=18)
    parser.add_argument("--reference-seed", type=int, default=0)
    parser.add_argument("--corridor-horizon", type=int, default=12)
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--rollout-horizon", type=int, default=96)
    parser.add_argument("--ppo-epochs", type=int, default=3)
    parser.add_argument("--minibatch", type=int, default=768)
    parser.add_argument("--adapt-actor-lr", type=float, default=1e-4)
    parser.add_argument("--scratch-actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.005)
    parser.add_argument("--progress-reward", type=float, default=1.0)
    parser.add_argument("--phase-progress-reward", type=float, default=0.04)
    parser.add_argument("--progress-clip", type=float, default=0.25)
    parser.add_argument("--inside-reward", type=float, default=0.08)
    parser.add_argument("--stagnation-penalty", type=float, default=0.01)
    parser.add_argument("--success-reward", type=float, default=3.0)
    parser.add_argument("--action-penalty", type=float, default=0.002)
    parser.add_argument("--corridor-radius", type=float, default=0.12)
    parser.add_argument("--initial-noise", type=float, default=0.025)
    parser.add_argument("--full-initial-noise", type=float, default=0.02)
    parser.add_argument("--phase-backtrack", type=int, default=4)
    parser.add_argument("--phase-advance", type=int, default=12)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-count", type=int, default=16)
    parser.add_argument("--num-test-seeds", type=int, default=3)
    parser.add_argument("--final-count", type=int, default=32)
    parser.add_argument("--evaluation-steps", type=int, default=750)
    parser.add_argument("--test-seed", type=int, default=20261201)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--checkpoint-out", default="results/target_online_adaptation_seed0.pt",
    )
    parser.add_argument(
        "--json-out", default="results/target_online_adaptation_seed0.json",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output = {
        "experiment": "HeavyInertiaTargetOnlineAdaptationValidation",
        "config": vars(args),
        "result": run(args),
    }
    text = json.dumps(output, ensure_ascii=False, indent=2)
    with open(args.json_out, "w", encoding="utf-8") as handle:
        handle.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
