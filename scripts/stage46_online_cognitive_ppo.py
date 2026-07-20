"""Online cognitive adaptation with the ProtoKAN-embedded PPO actor.

The policy is first trained on the source dynamics.  After switching to a
held-out dynamics factor, every real transition is used twice but with
separate objectives: PPO updates the decision parameters, while a prediction
loss updates only the cognitive ProtoKAN parameters.  No action teacher or
model-generated transition is used during adaptation.
"""

from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F

from physics_transfer.variants import step
from scripts.stage15_real_outcome_replay import PRETRAIN_FACTOR
from scripts.stage23_multistep_terminal_value import GOAL, SimpleCognitiveKAN
from scripts.stage27_parameter_transport import pretrain_cognitive
from scripts.stage41_ppo_cognitive_actor import (
    CognitiveEmbeddedGaussianActor,
    Rollout,
    ValueCritic,
    ppo_update,
    tip_height,
)


def reset_down_states(count: int, noise: float = 0.04) -> torch.Tensor:
    angles = torch.randn(count, 2) * noise
    return torch.stack([
        torch.cos(angles[:, 0]), torch.sin(angles[:, 0]),
        torch.cos(angles[:, 1]), torch.sin(angles[:, 1]),
        torch.randn(count) * noise, torch.randn(count) * noise,
    ], dim=-1)


def reward_fn(next_state, action):
    height = tip_height(next_state)
    success = height >= 1.0
    reward = 0.25 * (height + 2.0) + 5.0 * success.float()
    return reward - 0.005 * action.square().sum(dim=-1), success


def collect_rollout(actor, critic, factor, goal, num_envs, horizon,
                    gamma, gae_lambda, seed):
    """Collect PPO data and the exact real transitions for cognition update."""
    torch.manual_seed(seed)
    state = reset_down_states(num_envs)
    factor_tensor = torch.tensor(factor, dtype=state.dtype).view(1, 4).expand(num_envs, -1)
    goal_batch = goal.expand(num_envs, -1)
    states, actions, log_probs, values, rewards, dones = [], [], [], [], [], []
    transition_states, transition_actions, transition_next_states = [], [], []
    successes = 0
    for _ in range(horizon):
        with torch.no_grad():
            action, log_prob, _ = actor.sample(state, goal_batch)
            value = critic(state, goal_batch)
            next_state = step(
                state, action, factor_tensor[:, 0], factor_tensor[:, 1],
                factor_tensor[:, 2], factor_tensor[:, 3],
            )
            reward, done = reward_fn(next_state, action)
        successes += int(done.sum())
        states.append(state); actions.append(action); log_probs.append(log_prob)
        values.append(value); rewards.append(reward); dones.append(done)
        transition_states.append(state.clone())
        transition_actions.append(action.clone())
        transition_next_states.append(next_state.clone())
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
    rollout = Rollout(
        states.reshape(-1, 6), goal_batch.unsqueeze(0).expand(horizon, -1, -1).reshape(-1, 6),
        actions.reshape(-1, 1), log_probs.reshape(-1), values.reshape(-1),
        rewards.reshape(-1), dones.reshape(-1), advantages.reshape(-1), returns.reshape(-1),
    )
    transitions = (
        torch.cat(transition_states), torch.cat(transition_actions),
        torch.cat(transition_next_states),
    )
    return rollout, transitions, successes


def update_cognitive(cognitive, optimizer, transitions, epochs, minibatch, seed):
    """Fit only the cognition parameters to observed real transitions."""
    states, actions, next_states = transitions
    for parameter in cognitive.parameters():
        parameter.requires_grad = True
    generator = torch.Generator().manual_seed(seed)
    losses = []
    count = states.shape[0]
    for _ in range(epochs):
        order = torch.randperm(count, generator=generator)
        for start in range(0, count, minibatch):
            index = order[start:start + minibatch]
            prediction = cognitive(states[index], actions[index])
            loss = F.smooth_l1_loss(prediction, next_states[index])
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(cognitive.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
    for parameter in cognitive.parameters():
        parameter.requires_grad = False
    return float(sum(losses) / max(1, len(losses)))


@torch.no_grad()
def evaluate(actor, factor, goal, count, steps, seed):
    torch.manual_seed(seed)
    state = reset_down_states(count)
    factor_tensor = torch.tensor(factor, dtype=state.dtype).view(1, 4).expand(count, -1)
    maximum = torch.full((count,), -float("inf"))
    success = torch.zeros(count, dtype=torch.bool)
    for _ in range(steps):
        action, _, _ = actor.sample(state, goal.expand(count, -1), deterministic=True)
        state = step(state, action, factor_tensor[:, 0], factor_tensor[:, 1],
                     factor_tensor[:, 2], factor_tensor[:, 3])
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
    cognitive = SimpleCognitiveKAN()
    cognitive_fit = pretrain_cognitive(cognitive, args.cognitive_steps,
                                        args.cognitive_batch, args.seed)
    actor = CognitiveEmbeddedGaussianActor(cognitive, args.hidden_dim)
    actor.log_std.data.fill_(args.log_std_init)
    critic = ValueCritic(args.hidden_dim)
    actor_params = [p for name, p in actor.named_parameters()
                    if not name.startswith("cognitive.")]
    actor_optimizer = torch.optim.Adam(actor_params, lr=args.actor_lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)
    cognitive_optimizer = torch.optim.Adam(cognitive.parameters(), lr=args.cognitive_lr)
    goal = GOAL.view(1, -1)

    def run_phase(factor, iterations, offset, update_cognition):
        history = []
        for local_iteration in range(iterations):
            rollout, transitions, collected_successes = collect_rollout(
                actor, critic, factor, goal, args.num_envs, args.rollout_horizon,
                args.gamma, args.gae_lambda, args.seed + offset + local_iteration,
            )
            update = ppo_update(
                actor, critic, rollout, actor_optimizer, critic_optimizer,
                args.clip_ratio, args.value_coef, args.entropy_coef,
                args.ppo_epochs, args.minibatch,
                args.seed + 10000 + offset + local_iteration,
            )
            cognitive_loss = None
            if update_cognition:
                cognitive_loss = update_cognitive(
                    cognitive, cognitive_optimizer, transitions,
                    args.cognitive_update_epochs, args.cognitive_minibatch,
                    args.seed + 20000 + offset + local_iteration,
                )
            if local_iteration == 0 or (local_iteration + 1) % args.eval_every == 0:
                evaluation = evaluate(actor, factor, goal, args.test_count,
                                      args.eval_steps, args.test_seed + offset + local_iteration)
                history.append({
                    "iteration": local_iteration + 1,
                    "collected_successes": collected_successes,
                    "cognitive_update_loss": cognitive_loss,
                    **update, **evaluation,
                })
        return history

    source_history = run_phase(PRETRAIN_FACTOR[0], args.source_iterations, 0, False)
    heldout_before = evaluate(actor, args.heldout_factor, goal, args.test_count,
                              args.eval_steps, args.test_seed + 5000)
    adaptation_history = run_phase(args.heldout_factor, args.adaptation_iterations,
                                   1000, True)
    heldout_after = evaluate(actor, args.heldout_factor, goal, args.test_count,
                             args.eval_steps, args.test_seed + 6000)
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
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--rollout-horizon", type=int, default=256)
    parser.add_argument("--source-iterations", type=int, default=30)
    parser.add_argument("--adaptation-iterations", type=int, default=30)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--minibatch", type=int, default=512)
    parser.add_argument("--cognitive-minibatch", type=int, default=512)
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
    result = {
        "architecture": "OnlineCognitiveUpdate_ProtoKANEmbeddedPPO",
        "source_factor": PRETRAIN_FACTOR[0],
        "heldout_factor": args.heldout_factor,
        "config": vars(args),
        "result": train(args),
    }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")


if __name__ == "__main__":
    main()
