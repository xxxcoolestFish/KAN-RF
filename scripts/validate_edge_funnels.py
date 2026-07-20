"""Validate whether coarse cognitive graph edges are feedback executable."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal

from cpbn import OracleAcrobotDynamics
from cpbn.reachability import CoarseReachabilityPlanner
from cpbn.reachability_funnel import EmpiricalReachabilityFunnels


def mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim), nn.Tanh(),
        nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
        nn.Linear(hidden_dim, output_dim),
    )


class FunnelActor(nn.Module):
    def __init__(self, hidden_dim: int, log_std: float):
        super().__init__()
        self.net = mlp(18, hidden_dim, 1)
        self.log_std = nn.Parameter(torch.tensor([log_std]))

    @staticmethod
    def features(state, center, scale):
        return torch.cat([state, center, scale.log()], dim=-1)

    def distribution(self, state, center, scale):
        mean = self.net(self.features(state, center, scale))
        std = self.log_std.clamp(-5.0, 1.0).exp().expand_as(mean)
        return Normal(mean, std)

    @staticmethod
    def log_prob(distribution, raw, action):
        return (
            distribution.log_prob(raw)
            - torch.log(1.0 - action.square() + 1e-6)
        ).sum(dim=-1)

    def sample(self, state, center, scale, deterministic=False):
        distribution = self.distribution(state, center, scale)
        raw = distribution.mean if deterministic else distribution.rsample()
        action = torch.tanh(raw)
        return action, self.log_prob(distribution, raw, action)

    def evaluate(self, state, center, scale, action):
        distribution = self.distribution(state, center, scale)
        action = action.clamp(-0.999999, 0.999999)
        raw = torch.atanh(action)
        return self.log_prob(distribution, raw, action), distribution.entropy().sum(-1)


class FunnelCritic(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = mlp(18, hidden_dim, 1)

    def forward(self, state, center, scale):
        return self.net(torch.cat([state, center, scale.log()], dim=-1)).squeeze(-1)


@dataclass
class Rollout:
    state: torch.Tensor
    center: torch.Tensor
    scale: torch.Tensor
    action: torch.Tensor
    old_log_prob: torch.Tensor
    advantage: torch.Tensor
    returns: torch.Tensor


def reset_batch(funnels, count, generator):
    edge = torch.randint(funnels.edge_count, (count,), generator=generator)
    state, center, scale = funnels.sample_initial(edge, generator)
    return state, center, scale


def collect_rollout(actor, critic, dynamics, funnels, args, seed):
    generator = torch.Generator().manual_seed(seed)
    state, center, scale = reset_batch(funnels, args.num_envs, generator)
    states, centers, scales, actions = [], [], [], []
    log_probs, values, rewards, dones = [], [], [], []
    successes = 0
    for _ in range(args.rollout_horizon):
        with torch.no_grad():
            action, log_prob = actor.sample(state, center, scale)
            value = critic(state, center, scale)
            before = funnels.normalized_distance(state, center, scale)
            next_state = dynamics(state, action)
            after = funnels.normalized_distance(next_state, center, scale)
            done = after <= 1.0
            reward = args.progress_reward * (before - after)
            reward = reward + args.success_reward * done.float()
            reward = reward - args.action_penalty * action.square().squeeze(-1)
        successes += int(done.sum())
        states.append(state); centers.append(center); scales.append(scale)
        actions.append(action); log_probs.append(log_prob); values.append(value)
        rewards.append(reward); dones.append(done)
        reset_state, reset_center, reset_scale = reset_batch(
            funnels, args.num_envs, generator,
        )
        state = torch.where(done.unsqueeze(-1), reset_state, next_state)
        center = torch.where(done.unsqueeze(-1), reset_center, center)
        scale = torch.where(done.unsqueeze(-1), reset_scale, scale)

    with torch.no_grad():
        last_value = critic(state, center, scale)
    states = torch.stack(states); centers = torch.stack(centers)
    scales = torch.stack(scales); actions = torch.stack(actions)
    log_probs = torch.stack(log_probs); values = torch.stack(values)
    rewards = torch.stack(rewards); dones = torch.stack(dones)
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
        states.reshape(-1, 6), centers.reshape(-1, 6), scales.reshape(-1, 6),
        actions.reshape(-1, 1), log_probs.reshape(-1),
        advantage.reshape(-1), returns.reshape(-1),
    ), {
        "collected_edge_completions": successes,
        "mean_reward": float(rewards.mean()),
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
                rollout.state[index], rollout.center[index],
                rollout.scale[index], rollout.action[index],
            )
            ratio = torch.exp(log_prob - rollout.old_log_prob[index])
            raw = ratio * advantage[index]
            clipped = ratio.clamp(
                1.0 - args.clip_ratio, 1.0 + args.clip_ratio,
            ) * advantage[index]
            actor_loss = -torch.minimum(raw, clipped).mean()
            actor_loss = actor_loss - args.entropy_coef * entropy.mean()
            value_loss = F.mse_loss(
                critic(
                    rollout.state[index], rollout.center[index], rollout.scale[index],
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
def evaluate(actor, dynamics, funnels, args, seed, mode):
    generator = torch.Generator().manual_seed(seed)
    edge = torch.arange(funnels.edge_count).repeat_interleave(args.trials_per_edge)
    state, true_center, true_scale = funnels.sample_initial(edge, generator)
    if mode == "correct":
        input_center, input_scale = true_center, true_scale
    elif mode == "shuffled":
        permutation = torch.randperm(edge.shape[0], generator=generator)
        input_center, input_scale = true_center[permutation], true_scale[permutation]
    elif mode == "random_constant":
        input_center, input_scale = true_center, true_scale
        constant_action = torch.rand(edge.shape[0], 1, generator=generator) * 2.0 - 1.0
    else:
        raise ValueError(mode)

    completed = funnels.inside(state, true_center, true_scale)
    for _ in range(funnels.planner.macro_steps):
        if mode == "random_constant":
            action = constant_action
        else:
            action, _ = actor.sample(
                state, input_center, input_scale, deterministic=True,
            )
        state = dynamics(state, action)
        completed |= funnels.inside(state, true_center, true_scale)
    per_edge = completed.view(funnels.edge_count, args.trials_per_edge).float().mean(1)
    start = per_edge[funnels.start_route_mask]
    return {
        "mode": mode,
        "completion_rate": float(completed.float().mean()),
        "median_edge_completion": float(per_edge.median()),
        "minimum_edge_completion": float(per_edge.min()),
        "start_route_completion": float(start.mean()),
        "edges_above_95_percent": int((per_edge >= 0.95).sum()),
    }


@torch.no_grad()
def reference_sensitivity(actor, funnels, seed, count=512):
    generator = torch.Generator().manual_seed(seed)
    edge = torch.randint(funnels.edge_count, (count,), generator=generator)
    state, center, scale = funnels.sample_initial(edge, generator)
    permutation = torch.randperm(count, generator=generator)
    correct = torch.tanh(actor.net(actor.features(state, center, scale)))
    shuffled = torch.tanh(
        actor.net(actor.features(state, center[permutation], scale[permutation]))
    )
    return float((correct - shuffled).abs().mean())


def run(args):
    torch.manual_seed(args.seed)
    dynamics = OracleAcrobotDynamics()
    planner = CoarseReachabilityPlanner(
        dynamics,
        anchor_count=args.anchor_count,
        samples_per_anchor=args.samples_per_anchor,
        macro_steps=args.macro_steps,
        action_segments=1,
        maximum_snap_error=args.maximum_snap_error,
        seed=args.seed + 500,
    )
    funnels = EmpiricalReachabilityFunnels(
        planner, dynamics,
        edge_count=args.edge_count,
        action_grid=args.action_grid,
        perturbations=args.funnel_perturbations,
        perturbation_segments=args.funnel_segments,
        action_noise=args.funnel_action_noise,
        seed=args.seed + 700,
    )
    actor = FunnelActor(args.hidden_dim, args.log_std_init)
    critic = FunnelCritic(args.hidden_dim)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)
    random_baseline = evaluate(
        actor, dynamics, funnels, args, args.test_seed, "random_constant",
    )
    history = []
    for iteration in range(args.iterations):
        rollout, collection = collect_rollout(
            actor, critic, dynamics, funnels, args, args.seed + iteration,
        )
        update = ppo_update(
            actor, critic, rollout, actor_optimizer, critic_optimizer,
            args, args.seed + 10000 + iteration,
        )
        if iteration == 0 or (iteration + 1) % args.eval_every == 0:
            evaluation = evaluate(
                actor, dynamics, funnels, args,
                args.test_seed + iteration, "correct",
            )
            record = {"iteration": iteration + 1, **collection, **update, **evaluation}
            history.append(record)
            print(json.dumps(record), flush=True)
    correct = evaluate(actor, dynamics, funnels, args, args.test_seed + 10000, "correct")
    shuffled = evaluate(actor, dynamics, funnels, args, args.test_seed + 10000, "shuffled")
    return {
        "planner": asdict(planner.diagnostics),
        "funnels": asdict(funnels.diagnostics),
        "random_constant_baseline": random_baseline,
        "history": history,
        "final_correct": correct,
        "final_shuffled": shuffled,
        "reference_action_sensitivity": reference_sensitivity(
            actor, funnels, args.test_seed + 20000,
        ),
        "passed_95_percent_gate": correct["completion_rate"] >= 0.95,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor-count", type=int, default=1024)
    parser.add_argument("--samples-per-anchor", type=int, default=32)
    parser.add_argument("--macro-steps", type=int, default=24)
    parser.add_argument("--maximum-snap-error", type=float, default=0.65)
    parser.add_argument("--edge-count", type=int, default=96)
    parser.add_argument("--action-grid", type=int, default=65)
    parser.add_argument("--funnel-perturbations", type=int, default=64)
    parser.add_argument("--funnel-segments", type=int, default=4)
    parser.add_argument("--funnel-action-noise", type=float, default=0.15)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--rollout-horizon", type=int, default=96)
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch", type=int, default=1024)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--log-std-init", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--progress-reward", type=float, default=1.0)
    parser.add_argument("--success-reward", type=float, default=2.0)
    parser.add_argument("--action-penalty", type=float, default=0.002)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--trials-per-edge", type=int, default=16)
    parser.add_argument("--test-seed", type=int, default=20260803)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-out", type=str, default="")
    return parser.parse_args()


def main():
    args = parse_args()
    output = {
        "experiment": "OracleFeedbackReachabilityFunnelValidation",
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
