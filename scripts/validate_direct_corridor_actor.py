"""Train a direct Actor from state-corridor rewards without action labels."""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from cpbn import OracleAcrobotDynamics, tip_height
from cpbn.corridor_policy import (
    CorridorCritic,
    DirectCorridorActor,
    future_corridor,
)
from cpbn.receding_tube import local_state_distance
from cpbn.time_varying_tube import apply_tangent_error, plan_continuous_cem_route


@dataclass
class Rollout:
    state: torch.Tensor
    phase: torch.Tensor
    action: torch.Tensor
    old_log_prob: torch.Tensor
    advantage: torch.Tensor
    returns: torch.Tensor


def perturbed_reference(reference, phase, generator, noise):
    center = reference[phase]
    error = torch.randn(phase.shape[0], 4, generator=generator) * noise
    return apply_tangent_error(center, error)


def reset_batch(reference, count, generator, noise):
    phase = torch.randint(reference.shape[0] - 1, (count,), generator=generator)
    state = perturbed_reference(reference, phase, generator, noise)
    return state, phase


def collect_rollout(actor, critic, dynamics, reference, args, seed):
    generator = torch.Generator().manual_seed(seed)
    state, phase = reset_batch(
        reference, args.num_envs, generator, args.initial_noise,
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
        reset_state, reset_phase = reset_batch(
            reference, args.num_envs, generator, args.initial_noise,
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
    returns = advantage + values
    return Rollout(
        states.reshape(-1, 6),
        phases.reshape(-1),
        actions.reshape(-1, 1),
        log_probs.reshape(-1),
        advantage.reshape(-1),
        returns.reshape(-1),
    ), {
        "collected_successes": successes,
        "mean_reward": float(rewards.mean()),
        "return_target_std": float(returns.std(unbiased=False)),
    }


def ppo_update(
    actor, critic, reference, rollout,
    actor_optimizer, critic_optimizer, args, seed,
):
    generator = torch.Generator().manual_seed(seed)
    advantage = rollout.advantage
    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
    count = rollout.state.shape[0]
    actor_losses, critic_losses = [], []
    for _ in range(args.ppo_epochs):
        order = torch.randperm(count, generator=generator)
        for start in range(0, count, args.minibatch):
            index = order[start:start + args.minibatch]
            corridor = future_corridor(
                reference, rollout.phase[index], args.corridor_horizon,
            )
            log_prob, entropy = actor.evaluate(
                rollout.state[index], corridor, rollout.action[index],
            )
            ratio = torch.exp(log_prob - rollout.old_log_prob[index])
            raw = ratio * advantage[index]
            clipped = ratio.clamp(
                1.0 - args.clip_ratio, 1.0 + args.clip_ratio,
            ) * advantage[index]
            actor_loss = -torch.minimum(raw, clipped).mean()
            actor_loss = actor_loss - args.entropy_coef * entropy.mean()
            value_loss = F.smooth_l1_loss(
                critic(rollout.state[index], corridor), rollout.returns[index],
            )
            actor_optimizer.zero_grad(); actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            actor_optimizer.step()
            critic_optimizer.zero_grad(); value_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            critic_optimizer.step()
            actor_losses.append(float(actor_loss.detach()))
            critic_losses.append(float(value_loss.detach()))
    return {
        "actor_loss": sum(actor_losses) / len(actor_losses),
        "critic_loss": sum(critic_losses) / len(critic_losses),
    }


@torch.no_grad()
def evaluate_segments(actor, dynamics, reference, args, trials, seed, shuffled=False):
    generator = torch.Generator().manual_seed(seed)
    starts = torch.arange(0, reference.shape[0] - 1, args.segment_steps)
    edge = torch.arange(starts.shape[0]).repeat_interleave(trials)
    start_phase = starts[edge]
    state = perturbed_reference(
        reference, start_phase, generator, args.initial_noise,
    )
    for step in range(args.segment_steps):
        phase = (start_phase + step).clamp_max(reference.shape[0] - 1)
        descriptor_phase = phase
        if shuffled:
            descriptor_phase = (
                phase + args.shuffle_offset
            ) % (reference.shape[0] - 1)
        corridor = future_corridor(
            reference, descriptor_phase, args.corridor_horizon,
        )
        action, _ = actor.sample(state, corridor, deterministic=True)
        state = dynamics(state, action)
    final_phase = (start_phase + args.segment_steps).clamp_max(
        reference.shape[0] - 1,
    )
    completed = local_state_distance(
        state, reference[final_phase],
    ) <= args.segment_completion_radius
    per_segment = completed.view(starts.shape[0], trials).float().mean(dim=1)
    return {
        "completion_rate": float(completed.float().mean()),
        "minimum_segment_completion": float(per_segment.min()),
        "per_segment_completion": per_segment.tolist(),
    }


@torch.no_grad()
def evaluate_full(actor, dynamics, reference, args, count, seed, shuffled=False):
    generator = torch.Generator().manual_seed(seed)
    phase = torch.zeros(count, dtype=torch.long)
    state = perturbed_reference(
        reference, phase, generator, args.full_initial_noise,
    )
    success = torch.zeros(count, dtype=torch.bool)
    maximum = torch.full((count,), -2.0)
    for step in range(args.full_evaluation_steps):
        phase = torch.full_like(
            phase, min(step, reference.shape[0] - 1),
        )
        descriptor_phase = phase
        if shuffled:
            descriptor_phase = (
                phase + args.shuffle_offset
            ) % (reference.shape[0] - 1)
        corridor = future_corridor(
            reference, descriptor_phase, args.corridor_horizon,
        )
        action, _ = actor.sample(state, corridor, deterministic=True)
        state = dynamics(state, action)
        height = tip_height(state)
        maximum = torch.maximum(maximum, height)
        success |= height >= 1.0
    return {
        "success_count": int(success.sum()),
        "success_rate": float(success.float().mean()),
        "mean_maximum_height": float(maximum.mean()),
        "minimum_maximum_height": float(maximum.min()),
    }


@torch.no_grad()
def corridor_sensitivity(actor, reference, args, count, seed):
    generator = torch.Generator().manual_seed(seed)
    phase = torch.randint(reference.shape[0] - 1, (count,), generator=generator)
    state = perturbed_reference(reference, phase, generator, args.initial_noise)
    correct = future_corridor(reference, phase, args.corridor_horizon)
    wrong_phase = (phase + args.shuffle_offset) % (reference.shape[0] - 1)
    wrong = future_corridor(reference, wrong_phase, args.corridor_horizon)
    correct_action, _ = actor.sample(state, correct, deterministic=True)
    wrong_action, _ = actor.sample(state, wrong, deterministic=True)
    return float((correct_action - wrong_action).abs().mean())


def run(args):
    torch.manual_seed(args.seed)
    dynamics = OracleAcrobotDynamics()
    construction = plan_continuous_cem_route(
        dynamics,
        segment_count=args.reference_segments,
        segment_steps=args.segment_steps,
        population=args.reference_population,
        elite_count=args.reference_elite,
        iterations=args.reference_iterations,
        seed=args.reference_seed,
    )
    reference = construction.states.detach().clone()
    actor = DirectCorridorActor(args.hidden_dim, args.log_std_init)
    critic = CorridorCritic(args.hidden_dim)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)
    history = []
    best_score = -float("inf")
    best_iteration = 0
    best_actor = None
    for iteration in range(args.iterations):
        rollout, collection = collect_rollout(
            actor, critic, dynamics, reference, args, args.seed + iteration,
        )
        update = ppo_update(
            actor, critic, reference, rollout,
            actor_optimizer, critic_optimizer, args,
            args.seed + 10000 + iteration,
        )
        if iteration == 0 or (iteration + 1) % args.eval_every == 0:
            segment = evaluate_segments(
                actor, dynamics, reference, args,
                args.eval_segment_trials, args.test_seed + iteration,
            )
            full = evaluate_full(
                actor, dynamics, reference, args,
                args.eval_full_count, args.test_seed + 1000 + iteration,
            )
            record = {
                "iteration": iteration + 1,
                **collection,
                **update,
                "segment_completion": segment["completion_rate"],
                "minimum_segment_completion": segment[
                    "minimum_segment_completion"
                ],
                "full_success_rate": full["success_rate"],
                "full_mean_maximum_height": full["mean_maximum_height"],
            }
            history.append(record)
            print(json.dumps(record), flush=True)
            score = (
                2.0 * full["success_rate"]
                + segment["minimum_segment_completion"]
            )
            if score > best_score:
                best_score = score
                best_iteration = iteration + 1
                best_actor = copy.deepcopy(actor.state_dict())
    if best_actor is not None:
        actor.load_state_dict(best_actor)
    final_segments = evaluate_segments(
        actor, dynamics, reference, args,
        args.final_segment_trials, args.test_seed + 20000,
    )
    final_full = evaluate_full(
        actor, dynamics, reference, args,
        args.final_full_count, args.test_seed + 30000,
    )
    shuffled_full = evaluate_full(
        actor, dynamics, reference, args,
        args.final_full_count, args.test_seed + 30000, shuffled=True,
    )
    output = {
        "reference_route": {
            "state_count": int(reference.shape[0]),
            "maximum_height": construction.diagnostics.maximum_height,
            "success_step": construction.diagnostics.success_step,
            "actions_exposed_to_actor": False,
        },
        "history": history,
        "best_iteration": best_iteration,
        "final_segments": final_segments,
        "final_full_route": final_full,
        "shuffled_corridor_full_route": shuffled_full,
        "corridor_action_sensitivity": corridor_sensitivity(
            actor, reference, args, 512, args.test_seed + 40000,
        ),
        "actor_parameter_count": sum(p.numel() for p in actor.parameters()),
        "passed": (
            final_segments["minimum_segment_completion"] >= 0.90
            and final_full["success_rate"] >= 0.90
            and shuffled_full["success_rate"] < final_full["success_rate"] - 0.20
        ),
    }
    if args.checkpoint_out:
        torch.save({
            "actor": actor.state_dict(),
            "reference": reference,
            "config": vars(args),
        }, args.checkpoint_out)
    return output


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--corridor-horizon", type=int, default=12)
    parser.add_argument("--reference-segments", type=int, default=20)
    parser.add_argument("--segment-steps", type=int, default=24)
    parser.add_argument("--reference-population", type=int, default=2048)
    parser.add_argument("--reference-elite", type=int, default=128)
    parser.add_argument("--reference-iterations", type=int, default=12)
    parser.add_argument("--reference-seed", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=180)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--rollout-horizon", type=int, default=96)
    parser.add_argument("--ppo-epochs", type=int, default=3)
    parser.add_argument("--minibatch", type=int, default=768)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--log-std-init", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.005)
    parser.add_argument("--progress-reward", type=float, default=1.0)
    parser.add_argument("--progress-clip", type=float, default=0.25)
    parser.add_argument("--inside-reward", type=float, default=0.08)
    parser.add_argument("--success-reward", type=float, default=3.0)
    parser.add_argument("--action-penalty", type=float, default=0.002)
    parser.add_argument("--corridor-radius", type=float, default=0.12)
    parser.add_argument("--segment-completion-radius", type=float, default=0.15)
    parser.add_argument("--initial-noise", type=float, default=0.025)
    parser.add_argument("--full-initial-noise", type=float, default=0.02)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-segment-trials", type=int, default=8)
    parser.add_argument("--eval-full-count", type=int, default=8)
    parser.add_argument("--final-segment-trials", type=int, default=64)
    parser.add_argument("--final-full-count", type=int, default=32)
    parser.add_argument("--full-evaluation-steps", type=int, default=500)
    parser.add_argument("--shuffle-offset", type=int, default=120)
    parser.add_argument("--test-seed", type=int, default=20260807)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-out", type=str, default="")
    parser.add_argument("--json-out", type=str, default="")
    return parser.parse_args()


def main():
    args = parse_args()
    output = {
        "experiment": "DirectSequenceCorridorActorValidation",
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
