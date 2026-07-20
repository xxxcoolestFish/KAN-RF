"""Train and validate a direct Actor constrained by Oracle Jacobian pullback."""

from __future__ import annotations

import argparse
import copy
import json

import torch
import torch.nn.functional as F

from cpbn import OracleAcrobotDynamics, tip_height
from cpbn.cognitive_pullback import (
    CognitivePullbackActor,
    CognitivePullbackCritic,
    future_route_jacobians,
    local_jacobians_batch,
)
from cpbn.corridor_policy import future_corridor
from cpbn.feedback_phase import bounded_nearest_phase
from cpbn.receding_tube import local_state_distance
from cpbn.time_varying_tube import plan_continuous_cem_route
from scripts.validate_direct_corridor_actor import Rollout, perturbed_reference, reset_batch


def collect_rollout(
    actor, critic, dynamics, reference, route_a, route_b, args, seed,
):
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
            state_jacobian, action_jacobian = future_route_jacobians(
                route_a, route_b, phase, args.corridor_horizon,
            )
            action, log_prob = actor.sample(
                state, corridor, state_jacobian, action_jacobian,
            )
            value = critic(state, corridor, state_jacobian, action_jacobian)
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
        state_jacobian, action_jacobian = future_route_jacobians(
            route_a, route_b, phase, args.corridor_horizon,
        )
        last_value = critic(state, corridor, state_jacobian, action_jacobian)
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
        states.reshape(-1, 6), phases.reshape(-1), actions.reshape(-1, 1),
        log_probs.reshape(-1), advantage.reshape(-1), returns.reshape(-1),
    ), {
        "collected_successes": successes,
        "mean_reward": float(rewards.mean()),
        "return_target_std": float(returns.std(unbiased=False)),
    }


def ppo_update(
    actor, critic, reference, route_a, route_b, rollout,
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
            phase = rollout.phase[index]
            corridor = future_corridor(
                reference, phase, args.corridor_horizon,
            )
            state_jacobian, action_jacobian = future_route_jacobians(
                route_a, route_b, phase, args.corridor_horizon,
            )
            log_prob, entropy = actor.evaluate(
                rollout.state[index], corridor,
                state_jacobian, action_jacobian, rollout.action[index],
            )
            ratio = torch.exp(log_prob - rollout.old_log_prob[index])
            raw = ratio * advantage[index]
            clipped = ratio.clamp(
                1.0 - args.clip_ratio, 1.0 + args.clip_ratio,
            ) * advantage[index]
            actor_loss = -torch.minimum(raw, clipped).mean()
            actor_loss = actor_loss - args.entropy_coef * entropy.mean()
            value_loss = F.smooth_l1_loss(
                critic(
                    rollout.state[index], corridor,
                    state_jacobian, action_jacobian,
                ),
                rollout.returns[index],
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
def evaluate_full(
    actor, dynamics, reference, route_a, route_b, args, count, seed,
    shuffled_corridor=False, shuffled_jacobians=False,
):
    generator = torch.Generator().manual_seed(seed)
    phase = torch.zeros(count, dtype=torch.long)
    state = perturbed_reference(reference, phase, generator, args.full_initial_noise)
    success = torch.zeros(count, dtype=torch.bool)
    success_step = torch.full((count,), -1, dtype=torch.long)
    maximum = torch.full((count,), -2.0)
    for step in range(args.full_evaluation_steps):
        active = ~success
        corridor_phase = phase
        if shuffled_corridor:
            corridor_phase = (
                phase + args.shuffle_offset
            ) % (reference.shape[0] - 1)
        jacobian_phase = phase
        if shuffled_jacobians:
            jacobian_phase = (
                phase + args.shuffle_offset
            ) % (reference.shape[0] - 1)
        corridor = future_corridor(
            reference, corridor_phase, args.corridor_horizon,
        )
        state_jacobian, action_jacobian = future_route_jacobians(
            route_a, route_b, jacobian_phase, args.corridor_horizon,
        )
        action, _ = actor.sample(
            state, corridor, state_jacobian, action_jacobian,
            deterministic=True,
        )
        candidate_state = dynamics(state, action)
        state = torch.where(active.unsqueeze(-1), candidate_state, state)
        height = tip_height(state)
        maximum = torch.maximum(maximum, height)
        newly_successful = active & (height >= 1.0)
        success_step = torch.where(
            newly_successful, torch.full_like(success_step, step + 1), success_step,
        )
        success |= newly_successful
        candidate_phase = bounded_nearest_phase(
            state, reference, phase,
            backtrack=args.phase_backtrack,
            advance=args.phase_advance,
        )
        phase = torch.where(success, phase, candidate_phase)
    successful_steps = success_step[success]
    return {
        "success_count": int(success.sum()),
        "success_rate": float(success.float().mean()),
        "mean_success_step": (
            float(successful_steps.float().mean())
            if successful_steps.numel() else None
        ),
        "mean_maximum_height": float(maximum.mean()),
        "minimum_maximum_height": float(maximum.min()),
    }


def aggregate_evaluation(records, count):
    success_count = sum(item["success_count"] for item in records)
    total = len(records) * count
    weighted_steps = sum(
        item["mean_success_step"] * item["success_count"] for item in records
        if item["mean_success_step"] is not None
    )
    return {
        "success_count": success_count,
        "evaluation_count": total,
        "success_rate": success_count / total,
        "mean_success_step": weighted_steps / max(success_count, 1),
        "per_seed": records,
    }


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
    route_a, route_b = local_jacobians_batch(dynamics, reference)
    actor = CognitivePullbackActor(
        args.hidden_dim, args.log_std_init, args.log_pullback_scale,
    )
    critic = CognitivePullbackCritic(args.hidden_dim)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)
    history = []
    best_score = -float("inf")
    best_actor = None
    best_iteration = 0
    for iteration in range(args.iterations):
        rollout, collection = collect_rollout(
            actor, critic, dynamics, reference, route_a, route_b,
            args, args.seed + iteration,
        )
        update = ppo_update(
            actor, critic, reference, route_a, route_b, rollout,
            actor_optimizer, critic_optimizer, args,
            args.seed + 10000 + iteration,
        )
        if iteration == 0 or (iteration + 1) % args.eval_every == 0:
            evaluation = evaluate_full(
                actor, dynamics, reference, route_a, route_b, args,
                args.eval_count, args.test_seed + iteration,
            )
            record = {
                "iteration": iteration + 1,
                **collection, **update,
                "full_success_rate": evaluation["success_rate"],
                "full_mean_maximum_height": evaluation["mean_maximum_height"],
            }
            history.append(record)
            print(json.dumps(record), flush=True)
            score = evaluation["success_rate"]
            if score > best_score:
                best_score = score
                best_iteration = iteration + 1
                best_actor = copy.deepcopy(actor.state_dict())
    if best_actor is not None:
        actor.load_state_dict(best_actor)
    seeds = [args.test_seed + 50000 + index * 1009 for index in range(args.num_test_seeds)]
    normal, shuffled_corridor, shuffled_jacobians = [], [], []
    for seed in seeds:
        normal.append(evaluate_full(
            actor, dynamics, reference, route_a, route_b, args,
            args.final_count, seed,
        ))
        shuffled_corridor.append(evaluate_full(
            actor, dynamics, reference, route_a, route_b, args,
            args.final_count, seed, shuffled_corridor=True,
        ))
        shuffled_jacobians.append(evaluate_full(
            actor, dynamics, reference, route_a, route_b, args,
            args.final_count, seed, shuffled_jacobians=True,
        ))
    result = {
        "reference_route": {
            "state_count": int(reference.shape[0]),
            "maximum_height": construction.diagnostics.maximum_height,
            "success_step": construction.diagnostics.success_step,
            "actions_exposed_to_actor": False,
        },
        "history": history,
        "best_iteration": best_iteration,
        "normal": aggregate_evaluation(normal, args.final_count),
        "shuffled_corridor": aggregate_evaluation(
            shuffled_corridor, args.final_count,
        ),
        "shuffled_jacobians": aggregate_evaluation(
            shuffled_jacobians, args.final_count,
        ),
        "actor_parameter_count": sum(p.numel() for p in actor.parameters()),
        "zero_B_implies_zero_action_mean": True,
        "action_teacher_used": False,
        "passed": aggregate_evaluation(normal, args.final_count)[
            "success_rate"
        ] >= 0.95,
    }
    if args.checkpoint_out:
        torch.save({
            "actor": actor.state_dict(),
            "reference": reference,
            "route_A": route_a,
            "route_B": route_b,
            "config": vars(args),
        }, args.checkpoint_out)
    return result


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
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--rollout-horizon", type=int, default=96)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--minibatch", type=int, default=1024)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--log-std-init", type=float, default=0.0)
    parser.add_argument("--log-pullback-scale", type=float, default=4.0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.005)
    parser.add_argument("--progress-reward", type=float, default=1.0)
    parser.add_argument("--phase-progress-reward", type=float, default=0.04)
    parser.add_argument("--progress-clip", type=float, default=0.25)
    parser.add_argument("--inside-reward", type=float, default=0.02)
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
    parser.add_argument("--num-test-seeds", type=int, default=5)
    parser.add_argument("--final-count", type=int, default=64)
    parser.add_argument("--full-evaluation-steps", type=int, default=500)
    parser.add_argument("--shuffle-offset", type=int, default=120)
    parser.add_argument("--test-seed", type=int, default=20261001)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--checkpoint-out", default="results/oracle_pullback_actor_seed0.pt",
    )
    parser.add_argument(
        "--json-out", default="results/oracle_pullback_actor_seed0.json",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output = {
        "experiment": "OracleCognitivePullbackActorValidation",
        "config": vars(args),
        "result": run(args),
    }
    text = json.dumps(output, ensure_ascii=False, indent=2)
    with open(args.json_out, "w", encoding="utf-8") as handle:
        handle.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
