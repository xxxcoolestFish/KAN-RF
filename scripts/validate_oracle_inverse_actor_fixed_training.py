"""Train the cognitive inverse under the direct Actor's fixed-clock protocol."""

from __future__ import annotations

import json

import torch

from cpbn import tip_height
from cpbn.cognitive_pullback import future_route_jacobians
from cpbn.corridor_policy import future_corridor
from cpbn.receding_tube import local_state_distance
from scripts import validate_oracle_inverse_actor as inverse
from scripts import validate_oracle_pullback_actor as training_base
from scripts.validate_direct_corridor_actor import Rollout, reset_batch


def collect_fixed_clock(
    actor, critic, dynamics, reference, route_a, route_b, args, seed,
):
    """Match the original direct Actor collection and reward exactly."""
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


def main():
    args = inverse.parse_args()
    args.training_phase_mode = "fixed_clock"
    training_base.collect_rollout = collect_fixed_clock
    output = {
        "experiment": "OracleCognitiveInverseFixedClockTrainingValidation",
        "config": vars(args),
        "result": inverse.run(args),
    }
    text = json.dumps(output, ensure_ascii=False, indent=2)
    with open(args.json_out, "w", encoding="utf-8") as handle:
        handle.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
