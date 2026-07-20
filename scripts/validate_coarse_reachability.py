"""Test cognition-derived coarse state planning with a PPO local controller."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal

from cpbn import OracleAcrobotDynamics, reset_down_states, task_reward, tip_height
from cpbn.reachability import CoarseReachabilityPlanner, state_distance


def mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim), nn.Tanh(),
        nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
        nn.Linear(hidden_dim, output_dim),
    )


class GaussianWaypointActor(nn.Module):
    def __init__(self, hidden_dim: int, log_std: float):
        super().__init__()
        self.net = mlp(12, hidden_dim, 1)
        self.log_std = nn.Parameter(torch.tensor([log_std]))

    def mean(self, state, reference):
        return self.net(torch.cat([state, reference], dim=-1))

    def distribution(self, state, reference):
        mean = self.mean(state, reference)
        std = self.log_std.clamp(-5.0, 1.0).exp().expand_as(mean)
        return Normal(mean, std)

    @staticmethod
    def log_prob(distribution, raw, action):
        return (
            distribution.log_prob(raw)
            - torch.log(1.0 - action.square() + 1e-6)
        ).sum(dim=-1)

    def sample(self, state, reference, deterministic=False):
        distribution = self.distribution(state, reference)
        raw = distribution.mean if deterministic else distribution.rsample()
        action = torch.tanh(raw)
        return action, self.log_prob(distribution, raw, action)

    def evaluate(self, state, reference, action):
        distribution = self.distribution(state, reference)
        bounded = action.clamp(-0.999999, 0.999999)
        raw = torch.atanh(bounded)
        return self.log_prob(distribution, raw, bounded), distribution.entropy().sum(-1)


class WaypointCritic(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = mlp(12, hidden_dim, 1)

    def forward(self, state, reference):
        return self.net(torch.cat([state, reference], dim=-1)).squeeze(-1)


@dataclass
class Rollout:
    state: torch.Tensor
    reference: torch.Tensor
    action: torch.Tensor
    old_log_prob: torch.Tensor
    advantage: torch.Tensor
    returns: torch.Tensor


def collect_rollout(actor, critic, dynamics, planner, args, seed):
    generator = torch.Generator().manual_seed(seed)
    state = reset_down_states(args.num_envs, generator=generator)
    reference, _ = planner.query(state)
    states, references, actions, log_probs = [], [], [], []
    values, rewards, dones = [], [], []
    collected_successes = 0
    absolute_actions = []
    for time in range(args.rollout_horizon):
        if time % args.replan_period == 0:
            reference, _ = planner.query(state)
        with torch.no_grad():
            action, log_prob = actor.sample(state, reference)
            value = critic(state, reference)
            next_state = dynamics(state, action)
            base_reward, done = task_reward(state, next_state, action)
            progress = state_distance(state, reference) - state_distance(
                next_state, reference,
            )
            reward = base_reward + args.plan_reward * progress
        collected_successes += int(done.sum())
        states.append(state)
        references.append(reference)
        actions.append(action)
        log_probs.append(log_prob)
        values.append(value)
        rewards.append(reward)
        dones.append(done)
        absolute_actions.append(action.abs())
        reset = reset_down_states(args.num_envs, generator=generator)
        state = torch.where(done.unsqueeze(-1), reset, next_state)
        if done.any():
            reset_reference, _ = planner.query(state)
            reference = torch.where(done.unsqueeze(-1), reset_reference, reference)

    with torch.no_grad():
        last_value = critic(state, reference)
    states = torch.stack(states)
    references = torch.stack(references)
    actions = torch.stack(actions)
    log_probs = torch.stack(log_probs)
    values = torch.stack(values)
    rewards = torch.stack(rewards)
    dones = torch.stack(dones)
    advantage = torch.zeros_like(rewards)
    gae = torch.zeros(args.num_envs)
    next_value = last_value
    for index in reversed(range(args.rollout_horizon)):
        nonterminal = 1.0 - dones[index].float()
        delta = rewards[index] + args.gamma * next_value * nonterminal - values[index]
        gae = delta + args.gamma * args.gae_lambda * nonterminal * gae
        advantage[index] = gae
        next_value = values[index]
    returns = advantage + values
    return Rollout(
        states.reshape(-1, 6),
        references.reshape(-1, 6),
        actions.reshape(-1, 1),
        log_probs.reshape(-1),
        advantage.reshape(-1),
        returns.reshape(-1),
    ), {
        "collected_successes": collected_successes,
        "mean_absolute_action": float(torch.cat(absolute_actions).mean()),
        "mean_reward": float(rewards.mean()),
        "mean_plan_progress": float(
            (rewards - torch.stack([
                task_reward(states[i], dynamics(states[i], actions[i]), actions[i])[0]
                for i in range(args.rollout_horizon)
            ])).mean() / max(args.plan_reward, 1e-8)
        ),
    }


def ppo_update(actor, critic, rollout, actor_optimizer, critic_optimizer, args, seed):
    torch.manual_seed(seed)
    advantage = rollout.advantage
    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
    count = rollout.state.shape[0]
    actor_losses, critic_losses = [], []
    for _ in range(args.ppo_epochs):
        permutation = torch.randperm(count)
        for start in range(0, count, args.minibatch):
            index = permutation[start:start + args.minibatch]
            log_prob, entropy = actor.evaluate(
                rollout.state[index], rollout.reference[index], rollout.action[index],
            )
            ratio = torch.exp(log_prob - rollout.old_log_prob[index])
            unclipped = ratio * advantage[index]
            clipped = ratio.clamp(
                1.0 - args.clip_ratio, 1.0 + args.clip_ratio,
            ) * advantage[index]
            actor_loss = -torch.minimum(unclipped, clipped).mean()
            actor_loss = actor_loss - args.entropy_coef * entropy.mean()
            value_loss = F.mse_loss(
                critic(rollout.state[index], rollout.reference[index]),
                rollout.returns[index],
            )
            actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            actor_optimizer.step()
            critic_optimizer.zero_grad()
            (args.value_coef * value_loss).backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            critic_optimizer.step()
            actor_losses.append(float(actor_loss.detach()))
            critic_losses.append(float(value_loss.detach()))
    return {
        "actor_loss": sum(actor_losses) / len(actor_losses),
        "critic_loss": sum(critic_losses) / len(critic_losses),
    }


@torch.no_grad()
def evaluate(actor, dynamics, planner, args, seed):
    generator = torch.Generator().manual_seed(seed)
    state = reset_down_states(args.test_count, generator=generator)
    reference, route_distance = planner.query(state)
    initial_routed = torch.isfinite(route_distance)
    maximum = torch.full((args.test_count,), -float("inf"))
    success = torch.zeros(args.test_count, dtype=torch.bool)
    absolute_action = []
    for time in range(args.eval_steps):
        if time % args.replan_period == 0:
            reference, _ = planner.query(state)
        action, _ = actor.sample(state, reference, deterministic=True)
        state = dynamics(state, action)
        height = tip_height(state)
        maximum = torch.maximum(maximum, height)
        success |= height >= 1.0
        absolute_action.append(action.abs())
    return {
        "success_count": int(success.sum()),
        "success_rate": float(success.float().mean()),
        "mean_max_height": float(maximum.mean()),
        "max_height": float(maximum.max()),
        "mean_absolute_action": float(torch.cat(absolute_action).mean()),
        "initial_routed_fraction": float(initial_routed.float().mean()),
    }


def run(args):
    torch.manual_seed(args.seed)
    dynamics = OracleAcrobotDynamics()
    planner = CoarseReachabilityPlanner(
        dynamics,
        anchor_count=args.anchor_count,
        samples_per_anchor=args.samples_per_anchor,
        macro_steps=args.macro_steps,
        action_segments=args.action_segments,
        maximum_snap_error=args.maximum_snap_error,
        seed=args.seed + 500,
    )
    actor = GaussianWaypointActor(args.hidden_dim, args.log_std_init)
    critic = WaypointCritic(args.hidden_dim)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)
    history = []
    for iteration in range(args.iterations):
        rollout, collection = collect_rollout(
            actor, critic, dynamics, planner, args, args.seed + iteration,
        )
        update = ppo_update(
            actor, critic, rollout, actor_optimizer, critic_optimizer,
            args, args.seed + 10000 + iteration,
        )
        if iteration == 0 or (iteration + 1) % args.eval_every == 0:
            result = evaluate(
                actor, dynamics, planner, args, args.test_seed + iteration,
            )
            record = {"iteration": iteration + 1, **collection, **update, **result}
            history.append(record)
            print(json.dumps(record), flush=True)
    return {
        "planner": asdict(planner.diagnostics),
        "history": history,
        "final_evaluation": evaluate(
            actor, dynamics, planner, args, args.test_seed + 10000,
        ),
        "actor_parameters": sum(p.numel() for p in actor.parameters()),
        "critic_parameters": sum(p.numel() for p in critic.parameters()),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", type=int, required=True)
    parser.add_argument("--anchor-count", type=int, default=1024)
    parser.add_argument("--samples-per-anchor", type=int, default=16)
    parser.add_argument("--macro-steps", type=int, default=16)
    parser.add_argument("--action-segments", type=int, default=4)
    parser.add_argument("--maximum-snap-error", type=float, default=0.65)
    parser.add_argument("--replan-period", type=int, default=16)
    parser.add_argument("--plan-reward", type=float, default=1.0)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--rollout-horizon", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch", type=int, default=1024)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--log-std-init", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--test-count", type=int, default=20)
    parser.add_argument("--test-seed", type=int, default=20260802)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-out", type=str, default="")
    return parser.parse_args()


def main():
    args = parse_args()
    output = {
        "experiment": "OracleCoarseReachabilityWaypointPPO",
        "attempt": args.attempt,
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
