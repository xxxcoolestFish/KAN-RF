"""Train and validate a policy on three time-varying cognitive tubes."""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal

from cpbn import OracleAcrobotDynamics
from cpbn.time_varying_tube import TimeVaryingTubeSet, plan_continuous_cem_route


def mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim), nn.Tanh(),
        nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
        nn.Linear(hidden_dim, output_dim),
    )


class TubeActor(nn.Module):
    def __init__(self, hidden_dim: int, log_std: float):
        super().__init__()
        self.net = mlp(15, hidden_dim, 1)
        self.log_std = nn.Parameter(torch.tensor([log_std]))

    def distribution(self, features):
        mean = self.net(features)
        std = self.log_std.clamp(-5.0, 1.0).exp().expand_as(mean)
        return Normal(mean, std)

    @staticmethod
    def log_prob(distribution, raw, action):
        return (
            distribution.log_prob(raw)
            - torch.log(1.0 - action.square() + 1e-6)
        ).sum(dim=-1)

    def sample(self, features, deterministic=False):
        distribution = self.distribution(features)
        raw = distribution.mean if deterministic else distribution.rsample()
        action = torch.tanh(raw)
        return action, self.log_prob(distribution, raw, action)

    def evaluate(self, features, action):
        distribution = self.distribution(features)
        action = action.clamp(-0.999999, 0.999999)
        raw = torch.atanh(action)
        return self.log_prob(distribution, raw, action), distribution.entropy().sum(-1)


class TubeCritic(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = mlp(15, hidden_dim, 1)

    def forward(self, features):
        return self.net(features).squeeze(-1)


@dataclass
class Rollout:
    features: torch.Tensor
    action: torch.Tensor
    old_log_prob: torch.Tensor
    advantage: torch.Tensor
    returns: torch.Tensor


def reset_batch(tubes, count, generator):
    edge = torch.randint(tubes.edge_count, (count,), generator=generator)
    state = tubes.sample_initial(edge, generator)
    phase = torch.zeros(count, dtype=torch.long)
    return state, edge, phase


def collect_rollout(actor, critic, dynamics, tubes, args, seed):
    generator = torch.Generator().manual_seed(seed)
    state, edge, phase = reset_batch(tubes, args.num_envs, generator)
    features_list, actions, log_probs = [], [], []
    values, rewards, terminals = [], [], []
    completed = 0
    for _ in range(args.rollout_horizon):
        with torch.no_grad():
            features = tubes.policy_features(state, edge, phase)
            action, log_prob = actor.sample(features)
            value = critic(features)
            before = tubes.normalized_distance(state, edge, phase)
            next_state = dynamics(state, action)
            next_phase = phase + 1
            after = tubes.normalized_distance(next_state, edge, next_phase)
            inside = after <= 1.0
            terminal = next_phase >= tubes.horizon
            success = terminal & inside
            progress = (before - after).clamp(-args.progress_clip, args.progress_clip)
            reward = args.progress_reward * progress
            reward = reward + args.inside_reward * inside.float()
            reward = reward + args.success_reward * success.float()
            reward = reward - args.action_penalty * action.square().squeeze(-1)
        completed += int(success.sum())
        features_list.append(features); actions.append(action)
        log_probs.append(log_prob); values.append(value)
        rewards.append(reward); terminals.append(terminal)
        reset_state, reset_edge, reset_phase = reset_batch(
            tubes, args.num_envs, generator,
        )
        state = torch.where(terminal.unsqueeze(-1), reset_state, next_state)
        edge = torch.where(terminal, reset_edge, edge)
        phase = torch.where(terminal, reset_phase, next_phase)

    with torch.no_grad():
        last_features = tubes.policy_features(state, edge, phase)
        last_value = critic(last_features)
    features = torch.stack(features_list)
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
        features.reshape(-1, 15), actions.reshape(-1, 1),
        log_probs.reshape(-1), advantage.reshape(-1), returns.reshape(-1),
    ), {
        "collected_completions": completed,
        "mean_reward": float(rewards.mean()),
        "critic_target_std": float(returns.std(unbiased=False)),
    }


def ppo_update(actor, critic, rollout, actor_optimizer, critic_optimizer, args, seed):
    torch.manual_seed(seed)
    advantage = rollout.advantage
    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
    count = rollout.features.shape[0]
    actor_losses, critic_losses = [], []
    for _ in range(args.ppo_epochs):
        permutation = torch.randperm(count)
        for start in range(0, count, args.minibatch):
            index = permutation[start:start + args.minibatch]
            log_prob, entropy = actor.evaluate(
                rollout.features[index], rollout.action[index],
            )
            ratio = torch.exp(log_prob - rollout.old_log_prob[index])
            raw = ratio * advantage[index]
            clipped = ratio.clamp(
                1.0 - args.clip_ratio, 1.0 + args.clip_ratio,
            ) * advantage[index]
            actor_loss = -torch.minimum(raw, clipped).mean()
            actor_loss = actor_loss - args.entropy_coef * entropy.mean()
            value_loss = F.smooth_l1_loss(
                critic(rollout.features[index]), rollout.returns[index],
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
def evaluate(actor, dynamics, tubes, trials_per_edge, seed, mode):
    generator = torch.Generator().manual_seed(seed)
    edge = torch.arange(tubes.edge_count).repeat_interleave(trials_per_edge)
    state = tubes.sample_initial(edge, generator)
    if mode == "shuffled":
        descriptor_edge = torch.roll(edge, trials_per_edge)
    else:
        descriptor_edge = edge
    constant_action = None
    if mode == "random_constant":
        constant_action = torch.rand(edge.shape[0], 1, generator=generator) * 2.0 - 1.0
    stayed_inside = torch.ones(edge.shape[0], dtype=torch.bool)
    for step in range(tubes.horizon):
        phase = torch.full_like(edge, step)
        if mode == "random_constant":
            action = constant_action
        else:
            features = tubes.policy_features(state, descriptor_edge, phase)
            action, _ = actor.sample(features, deterministic=True)
        state = dynamics(state, action)
        next_phase = torch.full_like(edge, step + 1)
        stayed_inside &= tubes.normalized_distance(state, edge, next_phase) <= 1.0
    final_phase = torch.full_like(edge, tubes.horizon)
    completed = tubes.normalized_distance(state, edge, final_phase) <= 1.0
    per_edge = completed.view(tubes.edge_count, trials_per_edge).float().mean(1)
    return {
        "mode": mode,
        "completion_rate": float(completed.float().mean()),
        "per_edge_completion": per_edge.tolist(),
        "minimum_edge_completion": float(per_edge.min()),
        "full_tube_adherence": float(stayed_inside.float().mean()),
    }


@torch.no_grad()
def reference_sensitivity(actor, tubes, seed, count=512):
    generator = torch.Generator().manual_seed(seed)
    edge = torch.randint(tubes.edge_count, (count,), generator=generator)
    state = tubes.sample_initial(edge, generator)
    phase = torch.zeros(count, dtype=torch.long)
    shuffled = torch.roll(edge, 1)
    correct = torch.tanh(actor.net(tubes.policy_features(state, edge, phase)))
    wrong = torch.tanh(actor.net(tubes.policy_features(state, shuffled, phase)))
    return float((correct - wrong).abs().mean())


def run(args):
    torch.manual_seed(args.seed)
    dynamics = OracleAcrobotDynamics()
    route = plan_continuous_cem_route(
        dynamics,
        segment_count=args.route_segments,
        segment_steps=args.segment_steps,
        population=args.cem_population,
        elite_count=args.cem_elite,
        iterations=args.cem_iterations,
        seed=args.seed,
    )
    tubes = TimeVaryingTubeSet(
        dynamics, route,
        construction_samples=args.construction_samples,
        quantile=args.tube_quantile,
        seed=args.seed + 1000,
    )
    hidden_lqr = tubes.evaluate_hidden_lqr(args.trials_per_edge, args.test_seed)
    actor = TubeActor(args.hidden_dim, args.log_std_init)
    critic = TubeCritic(args.hidden_dim)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)
    random_baseline = evaluate(
        actor, dynamics, tubes, args.trials_per_edge,
        args.test_seed, "random_constant",
    )
    history = []
    best_rate = -1.0
    best_iteration = 0
    best_actor = None
    for iteration in range(args.iterations):
        rollout, collection = collect_rollout(
            actor, critic, dynamics, tubes, args, args.seed + iteration,
        )
        update = ppo_update(
            actor, critic, rollout, actor_optimizer, critic_optimizer,
            args, args.seed + 10000 + iteration,
        )
        if iteration == 0 or (iteration + 1) % args.eval_every == 0:
            evaluation = evaluate(
                actor, dynamics, tubes, args.trials_per_edge,
                args.test_seed + iteration, "correct",
            )
            record = {"iteration": iteration + 1, **collection, **update, **evaluation}
            history.append(record)
            print(json.dumps(record), flush=True)
            score = evaluation["minimum_edge_completion"]
            if score > best_rate:
                best_rate = score
                best_iteration = iteration + 1
                best_actor = copy.deepcopy(actor.state_dict())
    if best_actor is not None:
        actor.load_state_dict(best_actor)
    correct = evaluate(
        actor, dynamics, tubes, args.trials_per_edge,
        args.test_seed + 20000, "correct",
    )
    shuffled = evaluate(
        actor, dynamics, tubes, args.trials_per_edge,
        args.test_seed + 20000, "shuffled",
    )
    return {
        "route": asdict(route.diagnostics),
        "tubes": asdict(tubes.diagnostics),
        "hidden_lqr_certifier": hidden_lqr,
        "random_constant_baseline": random_baseline,
        "history": history,
        "best_iteration": best_iteration,
        "final_correct": correct,
        "final_shuffled": shuffled,
        "reference_action_sensitivity": reference_sensitivity(
            actor, tubes, args.test_seed + 30000,
        ),
        "passed_all_edges_95_percent": correct["minimum_edge_completion"] >= 0.95,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--route-segments", type=int, default=20)
    parser.add_argument("--segment-steps", type=int, default=24)
    parser.add_argument("--cem-population", type=int, default=2048)
    parser.add_argument("--cem-elite", type=int, default=128)
    parser.add_argument("--cem-iterations", type=int, default=12)
    parser.add_argument("--construction-samples", type=int, default=1024)
    parser.add_argument("--tube-quantile", type=float, default=0.99)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--rollout-horizon", type=int, default=96)
    parser.add_argument("--iterations", type=int, default=150)
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
    parser.add_argument("--progress-clip", type=float, default=0.25)
    parser.add_argument("--inside-reward", type=float, default=0.05)
    parser.add_argument("--success-reward", type=float, default=2.0)
    parser.add_argument("--action-penalty", type=float, default=0.002)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--trials-per-edge", type=int, default=64)
    parser.add_argument("--test-seed", type=int, default=20260804)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-out", type=str, default="")
    return parser.parse_args()


def main():
    args = parse_args()
    output = {
        "experiment": "OracleTimeVaryingCoupledTubeValidation",
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
