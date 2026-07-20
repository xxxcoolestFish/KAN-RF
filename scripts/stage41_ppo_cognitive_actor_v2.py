"""Fixed-start PPO validation for the ProtoKAN-embedded actor.

Stage 41 used uniformly random initial states, which makes Acrobot success
partly a measure of whether the initial state already lies near the goal. This
variant starts every episode near the hanging-down state and therefore tests
actual swing-up learning.
"""

from __future__ import annotations

import argparse
import json

import torch

from physics_transfer.multifactor_data import _random_states
from physics_transfer.variants import step
from scripts.stage15_real_outcome_replay import PRETRAIN_FACTOR
from scripts.stage23_multistep_terminal_value import SimpleCognitiveKAN
from scripts.stage27_parameter_transport import pretrain_cognitive
from scripts.stage41_ppo_cognitive_actor import (
    ACTION_DIM,
    GOAL,
    STATE_DIM,
    CognitiveEmbeddedGaussianActor,
    DirectGaussianActor,
    ValueCritic,
    _random_states,
    ppo_update,
    tip_height,
)


def reset_down_states(count: int, noise: float = 0.04) -> torch.Tensor:
    angles = torch.randn(count, 2) * noise
    state = torch.stack([
        torch.cos(angles[:, 0]), torch.sin(angles[:, 0]),
        torch.cos(angles[:, 1]), torch.sin(angles[:, 1]),
        torch.randn(count) * noise, torch.randn(count) * noise,
    ], dim=-1)
    return state


def dense_reward(state, next_state, action, goal):
    # Goal progress plus height is a task reward, not a hand-designed energy
    # teacher. The success bonus is the only terminal task signal.
    distance_before = (state - goal).square().sum(dim=-1)
    distance_after = (next_state - goal).square().sum(dim=-1)
    height = tip_height(next_state)
    success = (height >= 1.0).float()
    return (
        0.25 * (distance_before - distance_after)
        + 0.03 * height
        + 5.0 * success
        - 0.005 * action.square().sum(dim=-1)
    )


def collect_rollout(actor, critic, factor, goal, num_envs, horizon,
                    gamma, gae_lambda, seed):
    torch.manual_seed(seed)
    state = reset_down_states(num_envs)
    factor_tensor = torch.tensor(factor, dtype=state.dtype).view(1, 4).expand(num_envs, -1)
    goal_batch = goal.expand(num_envs, -1)
    states, actions, log_probs, values, rewards, dones = [], [], [], [], [], []
    for _ in range(horizon):
        with torch.no_grad():
            action, log_prob, _ = actor.sample(state, goal_batch)
            value = critic(state, goal_batch)
            next_state = step(
                state, action, factor_tensor[:, 0], factor_tensor[:, 1],
                factor_tensor[:, 2], factor_tensor[:, 3],
            )
            reward = dense_reward(state, next_state, action, goal_batch)
            done = (tip_height(next_state) >= 1.0)
        states.append(state); actions.append(action); log_probs.append(log_prob)
        values.append(value); rewards.append(reward); dones.append(done)
        state = torch.where(done.unsqueeze(-1), reset_down_states(num_envs), next_state)
    with torch.no_grad():
        last_value = critic(state, goal_batch)
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
    goals = goal_batch.unsqueeze(0).expand(horizon, -1, -1)
    from scripts.stage41_ppo_cognitive_actor import Rollout
    return Rollout(
        states.reshape(-1, STATE_DIM), goals.reshape(-1, STATE_DIM),
        actions.reshape(-1, ACTION_DIM), log_probs.reshape(-1), values.reshape(-1),
        rewards.reshape(-1), dones.reshape(-1), advantages.reshape(-1), returns.reshape(-1),
    )


@torch.no_grad()
def evaluate(actor, factor, goal, count, rollout_steps, seed):
    torch.manual_seed(seed)
    state = reset_down_states(count)
    initial = state.clone()
    factor_tensor = torch.tensor(factor, dtype=state.dtype).view(1, 4).expand(count, -1)
    maxima = torch.full((count,), -float("inf"))
    for _ in range(rollout_steps):
        action, _, _ = actor.sample(state, goal.expand(count, -1), deterministic=True)
        state = step(
            state, action, factor_tensor[:, 0], factor_tensor[:, 1],
            factor_tensor[:, 2], factor_tensor[:, 3],
        )
        maxima = torch.maximum(maxima, tip_height(state))
    success = maxima >= 1.0
    return {
        "success_count": int(success.sum()),
        "success_rate": float(success.float().mean()),
        "mean_max_height": float(maxima.mean()),
        "max_height": float(maxima.max()),
        "initial_mean_height": float(tip_height(initial).mean()),
    }


def train_variant(args, variant, seed):
    torch.manual_seed(seed)
    cognitive = SimpleCognitiveKAN()
    cognitive_fit = pretrain_cognitive(cognitive, args.cognitive_steps, args.cognitive_batch, seed)
    goal = GOAL.view(1, -1)
    actor = DirectGaussianActor(args.hidden_dim) if variant == "direct" else CognitiveEmbeddedGaussianActor(cognitive, args.hidden_dim)
    critic = ValueCritic(args.hidden_dim)
    actor_params = [p for name, p in actor.named_parameters() if not name.startswith("cognitive.")]
    actor_optimizer = torch.optim.Adam(actor_params, lr=args.actor_lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)
    history = []
    for iteration in range(args.iterations):
        rollout = collect_rollout(
            actor, critic, PRETRAIN_FACTOR[0], goal, args.num_envs,
            args.rollout_horizon, args.gamma, args.gae_lambda, seed + iteration,
        )
        update = ppo_update(
            actor, critic, rollout, actor_optimizer, critic_optimizer,
            args.clip_ratio, args.value_coef, args.entropy_coef,
            args.ppo_epochs, args.minibatch, seed + 10000 + iteration,
        )
        if iteration == 0 or (iteration + 1) % args.eval_every == 0:
            evaluation = evaluate(actor, PRETRAIN_FACTOR[0], goal, args.test_count, args.eval_steps, args.test_seed + iteration)
            history.append({"iteration": iteration + 1, **update, **evaluation})
    source = evaluate(actor, PRETRAIN_FACTOR[0], goal, args.test_count, args.eval_steps, args.test_seed + 1000)
    heldout = evaluate(actor, args.heldout_factor, goal, args.test_count, args.eval_steps, args.test_seed + 2000)
    return {
        "variant": variant,
        "cognitive_fit": cognitive_fit,
        "history": history,
        "source_evaluation": source,
        "heldout_evaluation": heldout,
        "actor_parameter_count": sum(p.numel() for p in actor_params),
        "cognitive_parameter_count": sum(p.numel() for p in cognitive.parameters()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cognitive-steps", type=int, default=300)
    parser.add_argument("--cognitive-batch", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--rollout-horizon", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=100)
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
    parser.add_argument("--eval-steps", type=int, default=128)
    parser.add_argument("--test-count", type=int, default=64)
    parser.add_argument("--test-seed", type=int, default=20260719)
    parser.add_argument("--heldout-factor", type=float, nargs=4, default=[9.80, 0.04, 1.10, 0.90])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json-out", type=str, default="")
    args = parser.parse_args()
    output = {
        "architecture": "FixedStartPPO_Direct_vs_ProtoKANEmbeddedActor",
        "start_state": "downward_with_small_noise",
        "source_factor": PRETRAIN_FACTOR[0],
        "heldout_factor": args.heldout_factor,
        "config": vars(args),
        "variants": [train_variant(args, variant, args.seed + i * 1000)
                     for i, variant in enumerate(("direct", "embedded"))],
    }
    text = json.dumps(output, ensure_ascii=False, indent=2)
    print(text)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")


if __name__ == "__main__":
    main()
