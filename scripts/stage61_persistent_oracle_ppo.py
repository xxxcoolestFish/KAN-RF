"""Target-only Oracle PPO with temporally persistent macro actions.

One sampled action is held for ``action_repeat`` physical transitions.  The
macro reward is discounted inside the block and GAE uses ``gamma**repeat``, so
the PPO update remains consistent with the induced semi-MDP.  Policy standard
deviation follows a fixed geometric schedule instead of being entropy-driven.
"""

from __future__ import annotations

import argparse
import json
import math

import torch

from physics_transfer.variants import step
from scripts import stage51_context_cognitive_ppo as base
from scripts.stage23_multistep_terminal_value import GOAL
from scripts.stage60_target_direct_oracle import ExactTargetCognitive


def collect_macro_rollout(actor, critic, cognitive, factor, goal, num_envs,
                          physical_horizon, action_repeat, gamma, gae_lambda,
                          seed):
    torch.manual_seed(seed)
    macro_steps = max(1, physical_horizon // action_repeat)
    state = base.reset_down_states(num_envs)
    context = torch.zeros(num_envs, base.CONTEXT_DIM)
    factor_tensor = torch.tensor(factor).view(1, 4).expand(num_envs, -1)
    goal_batch = goal.expand(num_envs, -1)
    states, actions, log_probs, values, rewards, dones = [], [], [], [], [], []
    successes = 0

    for _ in range(macro_steps):
        actor_state = torch.cat([state, context], dim=-1)
        with torch.no_grad():
            action, log_prob, _ = actor.sample(actor_state, goal_batch)
            value = critic(actor_state, goal_batch)
            macro_reward = torch.zeros(num_envs)
            active = torch.ones(num_envs, dtype=torch.bool)
            next_state = state
            discount = 1.0
            for _ in range(action_repeat):
                candidate = step(
                    next_state, action, factor_tensor[:, 0], factor_tensor[:, 1],
                    factor_tensor[:, 2], factor_tensor[:, 3],
                )
                reward, success = base.reward_fn(candidate, action)
                macro_reward = macro_reward + discount * reward * active.float()
                next_state = torch.where(active.unsqueeze(-1), candidate, next_state)
                newly_done = active & success
                successes += int(newly_done.sum())
                active = active & ~success
                discount *= gamma
            done = ~active

        states.append(actor_state); actions.append(action); log_probs.append(log_prob)
        values.append(value); rewards.append(macro_reward); dones.append(done)
        state = torch.where(done.unsqueeze(-1), base.reset_down_states(num_envs), next_state)
        context = torch.zeros_like(context)

    with torch.no_grad():
        last_actor_state = torch.cat([state, context], dim=-1)
        last_value = critic(last_actor_state, goal_batch)

    states = torch.stack(states); actions = torch.stack(actions)
    log_probs = torch.stack(log_probs); values = torch.stack(values)
    rewards = torch.stack(rewards); dones = torch.stack(dones)
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros(num_envs)
    next_value = last_value
    macro_gamma = gamma ** action_repeat
    for index in reversed(range(macro_steps)):
        nonterminal = 1.0 - dones[index].float()
        delta = rewards[index] + macro_gamma * next_value * nonterminal - values[index]
        gae = delta + macro_gamma * gae_lambda * nonterminal * gae
        advantages[index] = gae
        next_value = values[index]
    returns = advantages + values
    goals = goal_batch.unsqueeze(0).expand(macro_steps, -1, -1)
    rollout = base.Rollout(
        states.reshape(-1, base.ACTOR_STATE_DIM),
        goals.reshape(-1, base.STATE_DIM),
        actions.reshape(-1, base.ACTION_DIM),
        log_probs.reshape(-1), values.reshape(-1), rewards.reshape(-1),
        dones.reshape(-1), advantages.reshape(-1), returns.reshape(-1),
    )
    return rollout, successes


@torch.no_grad()
def evaluate_macro(actor, cognitive, factor, goal, count, physical_steps,
                   action_repeat, seed):
    torch.manual_seed(seed)
    state = base.reset_down_states(count)
    context = torch.zeros(count, base.CONTEXT_DIM)
    factor_tensor = torch.tensor(factor).view(1, 4).expand(count, -1)
    maximum = torch.full((count,), -float("inf"))
    success = torch.zeros(count, dtype=torch.bool)
    elapsed = 0
    while elapsed < physical_steps:
        actor_state = torch.cat([state, context], dim=-1)
        action, _, _ = actor.sample(actor_state, goal.expand(count, -1), deterministic=True)
        for _ in range(min(action_repeat, physical_steps - elapsed)):
            state = step(
                state, action, factor_tensor[:, 0], factor_tensor[:, 1],
                factor_tensor[:, 2], factor_tensor[:, 3],
            )
            height = base.tip_height(state)
            maximum = torch.maximum(maximum, height)
            success |= height >= 1.0
            elapsed += 1
    return {
        "success_count": int(success.sum()),
        "success_rate": float(success.float().mean()),
        "mean_max_height": float(maximum.mean()),
        "max_height": float(maximum.max()),
    }


def scheduled_std(start, end, iteration, total):
    fraction = iteration / max(1, total - 1)
    return math.exp((1.0 - fraction) * math.log(start) + fraction * math.log(end))


def train(args):
    torch.manual_seed(args.seed)
    factor = tuple(args.target_factor)
    cognitive = ExactTargetCognitive(factor)
    actor = base.ContextFiLMActor(cognitive, args.hidden_dim)
    actor.log_std.requires_grad_(False)
    critic = base.ContextValueCritic(args.hidden_dim)
    actor_parameters = [
        parameter for name, parameter in actor.named_parameters()
        if not name.startswith("cognitive.") and name != "log_std"
    ]
    actor_optimizer = torch.optim.Adam(actor_parameters, lr=args.actor_lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)
    goal = GOAL.view(1, -1)
    history = []

    for iteration in range(args.iterations):
        policy_std = scheduled_std(
            args.start_std, args.end_std, iteration, args.iterations,
        )
        actor.log_std.data.fill_(math.log(policy_std))
        rollout, collected_successes = collect_macro_rollout(
            actor, critic, cognitive, factor, goal, args.num_envs,
            args.rollout_horizon, args.action_repeat, args.gamma,
            args.gae_lambda, args.seed + iteration,
        )
        update = base.ppo_update(
            actor, critic, rollout, actor_optimizer, critic_optimizer,
            args.clip_ratio, args.value_coef, 0.0, args.ppo_epochs,
            args.minibatch, args.seed + 10000 + iteration,
        )
        if iteration == 0 or (iteration + 1) % args.eval_every == 0:
            history.append({
                "iteration": iteration + 1,
                "collected_successes": collected_successes,
                "policy_std": policy_std,
                **update,
                **evaluate_macro(
                    actor, cognitive, factor, goal, args.test_count,
                    args.eval_steps, args.action_repeat,
                    args.test_seed + iteration,
                ),
            })

    return {
        "target_factor": factor,
        "action_repeat": args.action_repeat,
        "history": history,
        "final_evaluation": evaluate_macro(
            actor, cognitive, factor, goal, args.test_count, args.eval_steps,
            args.action_repeat, args.test_seed + 1000,
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-factor", type=float, nargs=4,
                        default=[13.475, 0.06, 0.90, 1.10])
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--rollout-horizon", type=int, default=256)
    parser.add_argument("--action-repeat", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--minibatch", type=int, default=512)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--start-std", type=float, default=0.8)
    parser.add_argument("--end-std", type=float, default=0.08)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.0)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--test-count", type=int, default=64)
    parser.add_argument("--test-seed", type=int, default=20260720)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-out", type=str, default="")
    args = parser.parse_args()
    output = {
        "architecture": "ExactTargetDynamics_PersistentAction_ContextFiLM_PPO",
        "config": vars(args),
        "result": train(args),
    }
    text = json.dumps(output, ensure_ascii=False, indent=2)
    print(text)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")


if __name__ == "__main__":
    main()
