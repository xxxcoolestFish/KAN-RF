"""PPO decision head with an embedded, full ProtoKAN cognition module.

This is the first clean test of the revised architecture:

* the cognition model is trained only with next-state prediction;
* the actor either directly maps ``(state, goal)`` to an action, or first
  proposes an internal action query, passes ``(state, query)`` through the
  complete ProtoKAN model, and maps the predicted state to the final action;
* PPO learns the decision policy from real transition rewards;
* cognition parameters are frozen during PPO updates and can be updated by
  a separate prediction optimizer later.

The embedded actor has no raw-state shortcut after the ProtoKAN block: its
decision head receives only the ProtoKAN prediction and the goal.  Therefore
the complete cognition forward path is required for every action.
"""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal

from physics_transfer.multifactor_data import _random_states
from physics_transfer.variants import step
from scripts.stage15_real_outcome_replay import PRETRAIN_FACTOR
from scripts.stage23_multistep_terminal_value import SimpleCognitiveKAN
from scripts.stage27_parameter_transport import pretrain_cognitive


STATE_DIM = 6
ACTION_DIM = 1
GOAL = torch.tensor([-1.0, 0.0, 1.0, 0.0, 0.0, 0.0])


def tip_height(state: torch.Tensor) -> torch.Tensor:
    c1, s1, c2, s2 = state[:, 0], state[:, 1], state[:, 2], state[:, 3]
    return -c1 - (c1 * c2 - s1 * s2)


def goal_distance(state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
    return (state - goal).square().sum(dim=-1)


def task_reward(state: torch.Tensor, next_state: torch.Tensor,
                action: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
    """Goal-conditioned dense progress plus terminal success reward."""
    previous_distance = goal_distance(state, goal)
    next_distance = goal_distance(next_state, goal)
    success = (tip_height(next_state) >= 1.0).float()
    progress = previous_distance - next_distance
    return 0.20 * progress + 3.0 * success - 0.01 * action.square().sum(dim=-1)


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim), nn.Tanh(),
        nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
        nn.Linear(hidden_dim, output_dim),
    )


class GaussianActorBase(nn.Module):
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.log_std = nn.Parameter(torch.full((ACTION_DIM,), -0.7))
        self.hidden_dim = hidden_dim

    def _distribution(self, mean: torch.Tensor):
        log_std = self.log_std.clamp(-5.0, 1.0).expand_as(mean)
        return Normal(mean, log_std.exp())

    @staticmethod
    def _squashed_log_prob(distribution: Normal, raw: torch.Tensor,
                           action: torch.Tensor) -> torch.Tensor:
        return (
            distribution.log_prob(raw)
            - torch.log(1.0 - action.square() + 1e-6)
        ).sum(dim=-1)

    def sample(self, state: torch.Tensor, goal: torch.Tensor,
               deterministic: bool = False):
        mean = self.mean_action(state, goal)
        distribution = self._distribution(mean)
        raw = mean if deterministic else distribution.rsample()
        action = torch.tanh(raw)
        log_prob = self._squashed_log_prob(distribution, raw, action)
        return action, log_prob, mean

    def evaluate_actions(self, state: torch.Tensor, goal: torch.Tensor,
                         action: torch.Tensor):
        mean = self.mean_action(state, goal)
        distribution = self._distribution(mean)
        bounded = action.clamp(-0.999999, 0.999999)
        raw = torch.atanh(bounded)
        log_prob = self._squashed_log_prob(distribution, raw, bounded)
        entropy = distribution.entropy().sum(dim=-1)
        return log_prob, entropy


class DirectGaussianActor(GaussianActorBase):
    """Ordinary goal-conditioned actor baseline."""

    def __init__(self, hidden_dim: int = 64):
        super().__init__(hidden_dim)
        self.net = _mlp(STATE_DIM * 2, hidden_dim, ACTION_DIM)

    def mean_action(self, state: torch.Tensor, goal: torch.Tensor):
        return self.net(torch.cat([state, goal], dim=-1))


class CognitiveEmbeddedGaussianActor(GaussianActorBase):
    """Actor whose every action path passes through the full ProtoKAN model."""

    def __init__(self, cognitive: nn.Module, hidden_dim: int = 64):
        super().__init__(hidden_dim)
        self.cognitive = cognitive
        self.query_net = _mlp(STATE_DIM * 2, hidden_dim, ACTION_DIM)
        # Deliberately no raw-state input here: this head only receives the
        # predicted next state produced by the full cognitive network.
        self.decision_head = _mlp(STATE_DIM * 2, hidden_dim, ACTION_DIM)
        for parameter in self.cognitive.parameters():
            parameter.requires_grad = False

    def mean_action(self, state: torch.Tensor, goal: torch.Tensor):
        query = torch.tanh(self.query_net(torch.cat([state, goal], dim=-1)))
        predicted_next = self.cognitive(state, query)
        return self.decision_head(torch.cat([predicted_next, goal], dim=-1))

    def cognitive_prediction(self, state: torch.Tensor, goal: torch.Tensor):
        query = torch.tanh(self.query_net(torch.cat([state, goal], dim=-1)))
        return self.cognitive(state, query), query


class ValueCritic(nn.Module):
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.net = _mlp(STATE_DIM * 2, hidden_dim, 1)

    def forward(self, state: torch.Tensor, goal: torch.Tensor):
        return self.net(torch.cat([state, goal], dim=-1)).squeeze(-1)


@dataclass
class Rollout:
    state: torch.Tensor
    goal: torch.Tensor
    action: torch.Tensor
    old_log_prob: torch.Tensor
    value: torch.Tensor
    reward: torch.Tensor
    done: torch.Tensor
    advantage: torch.Tensor | None = None
    returns: torch.Tensor | None = None


def collect_rollout(actor, critic, factor, goal, num_envs, horizon,
                    gamma, gae_lambda, seed):
    torch.manual_seed(seed)
    state = _random_states(num_envs)
    factor_tensor = torch.tensor(factor, dtype=state.dtype).view(1, 4).expand(num_envs, -1)
    goal_batch = goal.expand(num_envs, -1)
    states, actions, log_probs, values, rewards, dones = [], [], [], [], [], []
    for _ in range(horizon):
        with torch.no_grad():
            action, log_prob, _ = actor.sample(state, goal_batch)
            value = critic(state, goal_batch)
            next_state = step(
                state, action,
                factor_tensor[:, 0], factor_tensor[:, 1],
                factor_tensor[:, 2], factor_tensor[:, 3],
            )
            reward = task_reward(state, next_state, action, goal_batch)
            done = (tip_height(next_state) >= 1.0)
        states.append(state)
        actions.append(action)
        log_probs.append(log_prob)
        values.append(value)
        rewards.append(reward)
        dones.append(done)
        reset = _random_states(num_envs)
        state = torch.where(done.unsqueeze(-1), reset, next_state)
    with torch.no_grad():
        last_value = critic(state, goal_batch)
    states = torch.stack(states)
    actions = torch.stack(actions)
    log_probs = torch.stack(log_probs)
    values = torch.stack(values)
    rewards = torch.stack(rewards)
    dones = torch.stack(dones)
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros(num_envs)
    next_value = last_value
    for index in reversed(range(horizon)):
        nonterminal = 1.0 - dones[index].float()
        delta = rewards[index] + gamma * next_value * nonterminal - values[index]
        gae = delta + gamma * gae_lambda * nonterminal * gae
        advantages[index] = gae
        next_value = values[index]
    returns = advantages + values
    return Rollout(
        states.reshape(-1, STATE_DIM), goal_batch.unsqueeze(0).expand(horizon, -1, -1).reshape(-1, STATE_DIM),
        actions.reshape(-1, ACTION_DIM), log_probs.reshape(-1), values.reshape(-1),
        rewards.reshape(-1), dones.reshape(-1), advantages.reshape(-1), returns.reshape(-1),
    )


def ppo_update(actor, critic, rollout, actor_optimizer, critic_optimizer,
               clip_ratio, value_coef, entropy_coef, epochs, minibatch,
               seed):
    torch.manual_seed(seed)
    advantages = rollout.advantage
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    count = rollout.state.shape[0]
    actor_losses, critic_losses = [], []
    for _ in range(epochs):
        permutation = torch.randperm(count)
        for start in range(0, count, minibatch):
            index = permutation[start:start + minibatch]
            state, goal = rollout.state[index], rollout.goal[index]
            action = rollout.action[index]
            old_log_prob = rollout.old_log_prob[index]
            advantage = advantages[index]
            returns = rollout.returns[index]
            log_prob, entropy = actor.evaluate_actions(state, goal, action)
            ratio = torch.exp(log_prob - old_log_prob)
            unclipped = ratio * advantage
            clipped = ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio) * advantage
            actor_loss = -torch.minimum(unclipped, clipped).mean() - entropy_coef * entropy.mean()
            value_loss = F.mse_loss(critic(state, goal), returns)
            actor_optimizer.zero_grad(); actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor_optimizer.param_groups[0]["params"], 1.0)
            actor_optimizer.step()
            critic_optimizer.zero_grad(); (value_coef * value_loss).backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            critic_optimizer.step()
            actor_losses.append(float(actor_loss.detach()))
            critic_losses.append(float(value_loss.detach()))
    return {
        "actor_loss": sum(actor_losses) / max(1, len(actor_losses)),
        "critic_loss": sum(critic_losses) / max(1, len(critic_losses)),
    }


@torch.no_grad()
def evaluate(actor, factor, goal, states, rollout_steps):
    state = states.clone()
    factor_tensor = torch.tensor(factor, dtype=state.dtype).view(1, 4).expand(state.shape[0], -1)
    maxima = torch.full((state.shape[0],), -float("inf"))
    for _ in range(rollout_steps):
        action, _, _ = actor.sample(state, goal.expand(state.shape[0], -1), deterministic=True)
        state = step(
            state, action,
            factor_tensor[:, 0], factor_tensor[:, 1],
            factor_tensor[:, 2], factor_tensor[:, 3],
        )
        maxima = torch.maximum(maxima, tip_height(state))
    success = maxima >= 1.0
    return {
        "success_count": int(success.sum()),
        "success_rate": float(success.float().mean()),
        "mean_max_height": float(maxima.mean()),
    }


def train_variant(args, variant: str, seed: int):
    torch.manual_seed(seed)
    cognitive = SimpleCognitiveKAN()
    cognitive_fit = pretrain_cognitive(cognitive, args.cognitive_steps, args.cognitive_batch, seed)
    goal = GOAL.view(1, -1)
    if variant == "direct":
        actor = DirectGaussianActor(args.hidden_dim)
    elif variant == "embedded":
        actor = CognitiveEmbeddedGaussianActor(cognitive, args.hidden_dim)
    else:
        raise ValueError(variant)
    critic = ValueCritic(args.hidden_dim)
    actor_params = [p for name, p in actor.named_parameters() if not name.startswith("cognitive.")]
    actor_optimizer = torch.optim.Adam(actor_params, lr=args.actor_lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)
    generator = torch.Generator().manual_seed(args.test_seed + seed)
    test_states = _random_states(args.test_count, generator=generator)
    history = []
    for iteration in range(args.iterations):
        rollout = collect_rollout(
            actor, critic, PRETRAIN_FACTOR[0], goal, args.num_envs,
            args.rollout_horizon, args.gamma, args.gae_lambda, seed + iteration,
        )
        update = ppo_update(
            actor, critic, rollout, actor_optimizer, critic_optimizer,
            args.clip_ratio, args.value_coef, args.entropy_coef, args.ppo_epochs,
            args.minibatch, seed + 10000 + iteration,
        )
        if iteration == 0 or (iteration + 1) % args.eval_every == 0:
            evaluation = evaluate(actor, PRETRAIN_FACTOR[0], goal, test_states, args.eval_steps)
            history.append({"iteration": iteration + 1, **update, **evaluation})
    return {
        "variant": variant,
        "cognitive_fit": cognitive_fit,
        "history": history,
        "source_evaluation": evaluate(actor, PRETRAIN_FACTOR[0], goal, test_states, args.eval_steps),
        "heldout_evaluation": evaluate(actor, args.heldout_factor, goal, test_states, args.eval_steps),
        "actor_parameter_count": sum(p.numel() for p in actor_params),
        "cognitive_parameter_count": sum(p.numel() for p in cognitive.parameters()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cognitive-steps", type=int, default=300)
    parser.add_argument("--cognitive-batch", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--rollout-horizon", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch", type=int, default=512)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=64)
    parser.add_argument("--test-count", type=int, default=64)
    parser.add_argument("--test-seed", type=int, default=20260719)
    parser.add_argument("--heldout-factor", type=float, nargs=4, default=[9.80, 0.04, 1.10, 0.90])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json-out", type=str, default="")
    args = parser.parse_args()
    results = {
        "architecture": "PPO_Direct_vs_ProtoKANEmbeddedActor",
        "source_factor": PRETRAIN_FACTOR[0],
        "heldout_factor": args.heldout_factor,
        "config": vars(args),
        "variants": [train_variant(args, variant, args.seed + index * 1000)
                     for index, variant in enumerate(("direct", "embedded"))],
    }
    text = json.dumps(results, ensure_ascii=False, indent=2)
    print(text)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")


if __name__ == "__main__":
    main()
