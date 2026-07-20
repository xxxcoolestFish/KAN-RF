"""Feedback-only refinement focused on weak state-corridor phases."""

from __future__ import annotations

import argparse
import copy
import json
from argparse import Namespace

import torch

from cpbn import OracleAcrobotDynamics, tip_height
from cpbn.corridor_policy import CorridorCritic, DirectCorridorActor, future_corridor
from cpbn.receding_tube import local_state_distance
from scripts import validate_direct_corridor_actor as base


def weighted_reset(reference, count, generator, noise, weights, segment_steps):
    segment = torch.multinomial(
        weights, count, replacement=True, generator=generator,
    )
    offset = torch.randint(segment_steps, (count,), generator=generator)
    phase = (segment * segment_steps + offset).clamp_max(reference.shape[0] - 2)
    state = base.perturbed_reference(reference, phase, generator, noise)
    return state, phase


def collect_weighted(actor, critic, dynamics, reference, args, weights, seed):
    generator = torch.Generator().manual_seed(seed)
    state, phase = weighted_reset(
        reference, args.num_envs, generator, args.initial_noise,
        weights, args.segment_steps,
    )
    states, phases, actions, log_probs = [], [], [], []
    values, rewards, terminals = [], [], []
    successes = 0
    for _ in range(args.rollout_horizon):
        with torch.no_grad():
            corridor = future_corridor(reference, phase, args.corridor_horizon)
            action, log_prob = actor.sample(state, corridor)
            value = critic(state, corridor)
            before = local_state_distance(state, reference[phase])
            next_state = dynamics(state, action)
            next_phase = (phase + 1).clamp_max(reference.shape[0] - 1)
            after = local_state_distance(next_state, reference[next_phase])
            task_success = tip_height(next_state) >= 1.0
            route_end = next_phase >= reference.shape[0] - 1
            terminal = task_success | route_end
            progress = (before - after).clamp(
                -args.progress_clip, args.progress_clip,
            )
            inside = after <= args.corridor_radius
            reward = args.progress_reward * progress
            reward = reward + args.inside_reward * inside.float()
            reward = reward + args.success_reward * task_success.float()
            reward = reward - args.action_penalty * action.square().squeeze(-1)
        successes += int(task_success.sum())
        states.append(state); phases.append(phase); actions.append(action)
        log_probs.append(log_prob); values.append(value)
        rewards.append(reward); terminals.append(terminal)
        reset_state, reset_phase = weighted_reset(
            reference, args.num_envs, generator, args.initial_noise,
            weights, args.segment_steps,
        )
        state = torch.where(terminal.unsqueeze(-1), reset_state, next_state)
        phase = torch.where(terminal, reset_phase, next_phase)
    with torch.no_grad():
        last_corridor = future_corridor(
            reference, phase, args.corridor_horizon,
        )
        last_value = critic(state, last_corridor)
    states = torch.stack(states); phases = torch.stack(phases)
    actions = torch.stack(actions); log_probs = torch.stack(log_probs)
    values = torch.stack(values); rewards = torch.stack(rewards)
    terminals = torch.stack(terminals)
    advantage = torch.zeros_like(rewards)
    gae = torch.zeros(args.num_envs)
    next_value = last_value
    for index in reversed(range(args.rollout_horizon)):
        nonterminal = 1.0 - terminals[index].float()
        delta = rewards[index] + args.gamma * next_value * nonterminal - values[index]
        gae = delta + args.gamma * args.gae_lambda * nonterminal * gae
        advantage[index] = gae
        next_value = values[index]
    return base.Rollout(
        states.reshape(-1, 6), phases.reshape(-1), actions.reshape(-1, 1),
        log_probs.reshape(-1), advantage.reshape(-1),
        (advantage + values).reshape(-1),
    ), {
        "collected_successes": successes,
        "mean_reward": float(rewards.mean()),
        "return_target_std": float((advantage + values).std(unbiased=False)),
    }


def run(cli):
    checkpoint = torch.load(cli.checkpoint_in, map_location="cpu", weights_only=False)
    previous = json.loads(open(cli.result_in, encoding="utf-8").read())
    args = Namespace(**checkpoint["config"])
    args.actor_lr = cli.actor_lr
    args.critic_lr = cli.critic_lr
    args.entropy_coef = cli.entropy_coef
    args.ppo_epochs = cli.ppo_epochs
    args.minibatch = cli.minibatch
    args.num_envs = cli.num_envs
    args.rollout_horizon = cli.rollout_horizon
    reference = checkpoint["reference"]
    dynamics = OracleAcrobotDynamics()
    actor = DirectCorridorActor(args.hidden_dim, args.log_std_init)
    actor.load_state_dict(checkpoint["actor"])
    critic = CorridorCritic(args.hidden_dim)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=0.0)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)

    rates = torch.tensor(
        previous["result"]["final_segments"]["per_segment_completion"],
    )
    weights = 0.20 + cli.difficulty_scale * (1.0 - rates).square()
    weights = weights / weights.sum()
    initial_segments = base.evaluate_segments(
        actor, dynamics, reference, args,
        cli.eval_segment_trials, args.test_seed + 50000,
    )
    initial_full = base.evaluate_full(
        actor, dynamics, reference, args,
        cli.eval_full_count, args.test_seed + 51000,
    )
    history = []
    best_actor = copy.deepcopy(actor.state_dict())
    best_score = (
        2.0 * initial_full["success_rate"]
        + initial_segments["minimum_segment_completion"]
    )
    best_iteration = 0
    for iteration in range(cli.iterations):
        if iteration == cli.critic_warmup_iterations:
            actor_optimizer.param_groups[0]["lr"] = args.actor_lr
        rollout, collection = collect_weighted(
            actor, critic, dynamics, reference, args, weights,
            args.seed + 60000 + iteration,
        )
        update = base.ppo_update(
            actor, critic, reference, rollout,
            actor_optimizer, critic_optimizer, args,
            args.seed + 70000 + iteration,
        )
        if (iteration + 1) % cli.eval_every == 0:
            segments = base.evaluate_segments(
                actor, dynamics, reference, args,
                cli.eval_segment_trials, args.test_seed + 52000 + iteration,
            )
            full = base.evaluate_full(
                actor, dynamics, reference, args,
                cli.eval_full_count, args.test_seed + 53000 + iteration,
            )
            record = {
                "iteration": iteration + 1,
                **collection,
                **update,
                "segment_completion": segments["completion_rate"],
                "minimum_segment_completion": segments[
                    "minimum_segment_completion"
                ],
                "full_success_rate": full["success_rate"],
                "full_mean_maximum_height": full["mean_maximum_height"],
            }
            history.append(record)
            print(json.dumps(record), flush=True)
            score = 2.0 * full["success_rate"] + segments[
                "minimum_segment_completion"
            ]
            if score > best_score:
                best_score = score
                best_iteration = iteration + 1
                best_actor = copy.deepcopy(actor.state_dict())
    actor.load_state_dict(best_actor)
    final_segments = base.evaluate_segments(
        actor, dynamics, reference, args,
        cli.final_segment_trials, args.test_seed + 54000,
    )
    final_full = base.evaluate_full(
        actor, dynamics, reference, args,
        cli.final_full_count, args.test_seed + 55000,
    )
    shuffled = base.evaluate_full(
        actor, dynamics, reference, args,
        cli.final_full_count, args.test_seed + 55000, shuffled=True,
    )
    result = {
        "initial_segments": initial_segments,
        "initial_full_route": initial_full,
        "segment_sampling_weights": weights.tolist(),
        "history": history,
        "best_iteration": best_iteration,
        "final_segments": final_segments,
        "final_full_route": final_full,
        "shuffled_corridor_full_route": shuffled,
        "corridor_action_sensitivity": base.corridor_sensitivity(
            actor, reference, args, 512, args.test_seed + 56000,
        ),
        "full_route_gate_passed": final_full["success_rate"] >= 0.90,
        "strict_local_gate_passed": (
            final_segments["minimum_segment_completion"] >= 0.90
        ),
        "action_teacher_used": False,
    }
    if cli.checkpoint_out:
        torch.save({
            "actor": actor.state_dict(),
            "reference": reference,
            "config": vars(args),
        }, cli.checkpoint_out)
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-in", default="results/direct_corridor_actor_seed0.pt")
    parser.add_argument("--result-in", default="results/direct_corridor_actor_seed0.json")
    parser.add_argument("--checkpoint-out", default="results/direct_corridor_actor_refined_seed0.pt")
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--critic-warmup-iterations", type=int, default=5)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--rollout-horizon", type=int, default=96)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--minibatch", type=int, default=1024)
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--entropy-coef", type=float, default=0.002)
    parser.add_argument("--difficulty-scale", type=float, default=4.0)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-segment-trials", type=int, default=16)
    parser.add_argument("--eval-full-count", type=int, default=16)
    parser.add_argument("--final-segment-trials", type=int, default=64)
    parser.add_argument("--final-full-count", type=int, default=64)
    parser.add_argument("--json-out", default="results/direct_corridor_actor_refined_seed0.json")
    return parser.parse_args()


def main():
    cli = parse_args()
    output = {
        "experiment": "HardPhaseDirectCorridorActorRefinement",
        "config": vars(cli),
        "result": run(cli),
    }
    text = json.dumps(output, ensure_ascii=False, indent=2)
    if cli.json_out:
        with open(cli.json_out, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
