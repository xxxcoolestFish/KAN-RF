"""History-conditioned ProtoKAN cognition for online physical adaptation.

The context state is inferred from observed real transitions and is not tied to
an explicit list of physical parameters.  The full cognitive predictor remains
inside a FiLM-gated actor; the decision head never receives the query directly.
"""

from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F
from torch import nn

from kanrf._protokan import ProtoKAN
from physics_transfer.transition_data import sample_transition_sequence_batch
from physics_transfer.variants import step
from scripts.stage15_real_outcome_replay import PRETRAIN_FACTOR
from scripts.stage23_multistep_terminal_value import GOAL
from scripts.stage41_ppo_cognitive_actor import (
    GaussianActorBase,
    Rollout,
    ValueCritic,
    _mlp,
    ppo_update,
    tip_height,
)

STATE_DIM = 6
ACTION_DIM = 1
CONTEXT_DIM = 8
ACTOR_STATE_DIM = STATE_DIM + CONTEXT_DIM


class ContextCognitiveKAN(nn.Module):
    def __init__(self, context_dim: int = CONTEXT_DIM, hidden_dim: int = 64):
        super().__init__()
        self.context_dim = context_dim
        self.context_encoder = nn.GRUCell(STATE_DIM + ACTION_DIM + STATE_DIM, context_dim)
        self.network = ProtoKAN(
            [STATE_DIM + ACTION_DIM + context_dim, hidden_dim, STATE_DIM],
            n_prototypes=8,
        )

    def forward(self, state, action, context):
        return self.network(torch.cat([state, action, context], dim=-1))

    def update_context(self, context, state, action, next_state):
        transition = torch.cat([state, action, next_state], dim=-1)
        return self.context_encoder(transition, context)


class ContextFiLMActor(GaussianActorBase):
    def __init__(self, cognitive: ContextCognitiveKAN, hidden_dim: int = 64):
        super().__init__(hidden_dim)
        self.cognitive = cognitive
        self.query_net = _mlp(ACTOR_STATE_DIM + STATE_DIM, hidden_dim, ACTION_DIM)
        self.modulation = nn.Linear(ACTION_DIM, STATE_DIM * 2)
        self.decision_head = _mlp(STATE_DIM * 3, hidden_dim, ACTION_DIM)
        for parameter in self.cognitive.parameters():
            parameter.requires_grad = False

    def mean_action(self, actor_state, goal):
        state = actor_state[:, :STATE_DIM]
        context = actor_state[:, STATE_DIM:]
        query = torch.tanh(self.query_net(torch.cat([actor_state, goal], dim=-1)))
        predicted_next = self.cognitive(state, query, context)
        scale, bias = self.modulation(query).chunk(2, dim=-1)
        scale = 0.5 * torch.tanh(scale)
        bias = 0.5 * torch.tanh(bias)
        modulated = predicted_next * (1.0 + scale) + bias
        delta = modulated - state
        return self.decision_head(torch.cat([modulated, delta, goal], dim=-1))


class ContextValueCritic(ValueCritic):
    def forward(self, actor_state, goal):
        return super().forward(actor_state[:, :STATE_DIM], goal)


def reset_down_states(count: int, noise: float = 0.04):
    angles = torch.randn(count, 2) * noise
    return torch.stack([
        torch.cos(angles[:, 0]), torch.sin(angles[:, 0]),
        torch.cos(angles[:, 1]), torch.sin(angles[:, 1]),
        torch.randn(count) * noise, torch.randn(count) * noise,
    ], dim=-1)


def pretrain_cognitive(cognitive, steps: int, batch_size: int,
                       sequence_steps: int, seed: int):
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(cognitive.parameters(), lr=2e-3)
    losses = []
    for _ in range(steps):
        batch = sample_transition_sequence_batch(
            batch_size, sequence_steps, PRETRAIN_FACTOR,
        )
        state_seq = batch["state"]
        action_seq = batch["action"]
        next_seq = batch["next_state"]
        context = torch.zeros(batch_size, CONTEXT_DIM)
        loss = torch.zeros(())
        for t in range(sequence_steps):
            prediction = cognitive(state_seq[:, t], action_seq[:, t], context)
            loss = loss + F.smooth_l1_loss(prediction, next_seq[:, t])
            context = cognitive.update_context(
                context, state_seq[:, t], action_seq[:, t], next_seq[:, t],
            )
        loss = loss / sequence_steps
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(cognitive.parameters(), 5.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    return {
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "mean_last_20_loss": sum(losses[-20:]) / min(20, len(losses)),
    }


def reward_fn(next_state, action):
    height = tip_height(next_state)
    success = height >= 1.0
    reward = 0.25 * (height + 2.0) + 5.0 * success.float()
    return reward - 0.005 * action.square().sum(dim=-1), success


def collect_rollout(actor, critic, cognitive, factor, goal, num_envs,
                    horizon, gamma, gae_lambda, seed):
    torch.manual_seed(seed)
    state = reset_down_states(num_envs)
    context = torch.zeros(num_envs, CONTEXT_DIM)
    factor_tensor = torch.tensor(factor).view(1, 4).expand(num_envs, -1)
    goal_batch = goal.expand(num_envs, -1)
    states, actions, log_probs, values, rewards, dones = [], [], [], [], [], []
    transition_states, transition_actions, transition_next, transition_done = [], [], [], []
    successes = 0
    for _ in range(horizon):
        actor_state = torch.cat([state, context], dim=-1)
        with torch.no_grad():
            action, log_prob, _ = actor.sample(actor_state, goal_batch)
            value = critic(actor_state, goal_batch)
            next_state = step(
                state, action, factor_tensor[:, 0], factor_tensor[:, 1],
                factor_tensor[:, 2], factor_tensor[:, 3],
            )
            reward, done = reward_fn(next_state, action)
            next_context = cognitive.update_context(context, state, action, next_state)
        successes += int(done.sum())
        states.append(actor_state); actions.append(action); log_probs.append(log_prob)
        values.append(value); rewards.append(reward); dones.append(done)
        transition_states.append(state.clone()); transition_actions.append(action.clone())
        transition_next.append(next_state.clone()); transition_done.append(done.clone())
        state = torch.where(done.unsqueeze(-1), reset_down_states(num_envs), next_state)
        context = torch.where(done.unsqueeze(-1), torch.zeros_like(context), next_context)
    with torch.no_grad():
        last_actor_state = torch.cat([state, context], dim=-1)
        last_value = critic(last_actor_state, goal_batch)
    states = torch.stack(states); actions = torch.stack(actions)
    log_probs = torch.stack(log_probs); values = torch.stack(values)
    rewards = torch.stack(rewards); dones = torch.stack(dones)
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros(num_envs); next_value = last_value
    for index in reversed(range(horizon)):
        nonterminal = 1.0 - dones[index].float()
        delta = rewards[index] + gamma * next_value * nonterminal - values[index]
        gae = delta + gamma * gae_lambda * nonterminal * gae
        advantages[index] = gae
        next_value = values[index]
    returns = advantages + values
    rollout = Rollout(
        states.reshape(-1, ACTOR_STATE_DIM),
        goal_batch.unsqueeze(0).expand(horizon, -1, -1).reshape(-1, STATE_DIM),
        actions.reshape(-1, ACTION_DIM), log_probs.reshape(-1), values.reshape(-1),
        rewards.reshape(-1), dones.reshape(-1), advantages.reshape(-1), returns.reshape(-1),
    )
    transitions = (
        torch.stack(transition_states), torch.stack(transition_actions),
        torch.stack(transition_next), torch.stack(transition_done),
    )
    return rollout, transitions, successes


def update_cognitive(cognitive, optimizer, transitions, epochs, seed):
    states, actions, next_states, dones = transitions
    for parameter in cognitive.parameters():
        parameter.requires_grad = True
    losses = []
    for _ in range(epochs):
        context = torch.zeros(states.shape[1], CONTEXT_DIM)
        loss = torch.zeros(())
        for t in range(states.shape[0]):
            prediction = cognitive(states[t], actions[t], context)
            loss = loss + F.smooth_l1_loss(prediction, next_states[t])
            next_context = cognitive.update_context(
                context, states[t], actions[t], next_states[t],
            )
            context = torch.where(dones[t].unsqueeze(-1), torch.zeros_like(context), next_context)
        loss = loss / states.shape[0]
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(cognitive.parameters(), 5.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    for parameter in cognitive.parameters():
        parameter.requires_grad = False
    return sum(losses) / max(1, len(losses))


@torch.no_grad()
def evaluate(actor, cognitive, factor, goal, count, steps, seed):
    torch.manual_seed(seed)
    state = reset_down_states(count)
    context = torch.zeros(count, CONTEXT_DIM)
    factor_tensor = torch.tensor(factor).view(1, 4).expand(count, -1)
    maximum = torch.full((count,), -float("inf"))
    success = torch.zeros(count, dtype=torch.bool)
    for _ in range(steps):
        actor_state = torch.cat([state, context], dim=-1)
        action, _, _ = actor.sample(actor_state, goal.expand(count, -1), deterministic=True)
        next_state = step(state, action, factor_tensor[:, 0], factor_tensor[:, 1],
                           factor_tensor[:, 2], factor_tensor[:, 3])
        context = cognitive.update_context(context, state, action, next_state)
        state = next_state
        height = tip_height(state)
        maximum = torch.maximum(maximum, height)
        success |= height >= 1.0
    return {
        "success_count": int(success.sum()),
        "success_rate": float(success.float().mean()),
        "mean_max_height": float(maximum.mean()),
        "max_height": float(maximum.max()),
    }


def train(args):
    torch.manual_seed(args.seed)
    cognitive = ContextCognitiveKAN()
    cognitive_fit = pretrain_cognitive(
        cognitive, args.cognitive_steps, args.cognitive_batch,
        args.sequence_steps, args.seed,
    )
    actor = ContextFiLMActor(cognitive, args.hidden_dim)
    actor.log_std.data.fill_(args.log_std_init)
    critic = ContextValueCritic(args.hidden_dim)
    actor_params = [p for name, p in actor.named_parameters()
                    if not name.startswith("cognitive.")]
    actor_optimizer = torch.optim.Adam(actor_params, lr=args.actor_lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)
    cognitive_optimizer = torch.optim.Adam(cognitive.parameters(), lr=args.cognitive_lr)
    goal = GOAL.view(1, -1)

    def phase(factor, iterations, offset, online):
        history = []
        for local in range(iterations):
            rollout, transitions, collected = collect_rollout(
                actor, critic, cognitive, factor, goal, args.num_envs,
                args.rollout_horizon, args.gamma, args.gae_lambda,
                args.seed + offset + local,
            )
            update = ppo_update(
                actor, critic, rollout, actor_optimizer, critic_optimizer,
                args.clip_ratio, args.value_coef, args.entropy_coef,
                args.ppo_epochs, args.minibatch, args.seed + 10000 + offset + local,
            )
            cognitive_loss = None
            if online:
                cognitive_loss = update_cognitive(
                    cognitive, cognitive_optimizer, transitions,
                    args.cognitive_update_epochs, args.seed + 20000 + offset + local,
                )
            if local == 0 or (local + 1) % args.eval_every == 0:
                history.append({
                    "iteration": local + 1,
                    "collected_successes": collected,
                    "cognitive_update_loss": cognitive_loss,
                    **update,
                    **evaluate(actor, cognitive, factor, goal, args.test_count,
                               args.eval_steps, args.test_seed + offset + local),
                })
        return history

    source_history = phase(PRETRAIN_FACTOR[0], args.source_iterations, 0, False)
    heldout_before = evaluate(actor, cognitive, args.heldout_factor, goal,
                              args.test_count, args.eval_steps, args.test_seed + 5000)
    adaptation_history = phase(args.heldout_factor, args.adaptation_iterations, 1000, True)
    heldout_after = evaluate(actor, cognitive, args.heldout_factor, goal,
                             args.test_count, args.eval_steps, args.test_seed + 6000)
    return {
        "seed": args.seed,
        "cognitive_fit": cognitive_fit,
        "source_history": source_history,
        "heldout_before_adaptation": heldout_before,
        "adaptation_history": adaptation_history,
        "heldout_after_adaptation": heldout_after,
        "actor_parameter_count": sum(p.numel() for p in actor_params),
        "cognitive_parameter_count": sum(p.numel() for p in cognitive.parameters()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cognitive-steps", type=int, default=300)
    parser.add_argument("--cognitive-batch", type=int, default=32)
    parser.add_argument("--sequence-steps", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--rollout-horizon", type=int, default=256)
    parser.add_argument("--source-iterations", type=int, default=30)
    parser.add_argument("--adaptation-iterations", type=int, default=30)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--minibatch", type=int, default=512)
    parser.add_argument("--cognitive-update-epochs", type=int, default=1)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--cognitive-lr", type=float, default=2e-3)
    parser.add_argument("--log-std-init", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--test-count", type=int, default=64)
    parser.add_argument("--test-seed", type=int, default=20260719)
    parser.add_argument("--heldout-factor", type=float, nargs=4,
                        default=[13.475, 0.06, 0.90, 1.10])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-out", type=str, default="")
    args = parser.parse_args()
    output = {
        "architecture": "HistoryContext_ProtoKAN_FiLM_PPO",
        "source_factor": PRETRAIN_FACTOR[0],
        "heldout_factor": args.heldout_factor,
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
