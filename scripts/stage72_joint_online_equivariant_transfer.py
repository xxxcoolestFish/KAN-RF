"""Full online transfer: separate cognition and PPO losses, shared forward."""

from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F

from scripts import stage43_standard_ppo_baseline as ppo
from scripts.stage23_multistep_terminal_value import GOAL
from scripts.stage70_equivariant_causal_update_actor import (
    EquivariantCausalUpdateActor,
)
from scripts.stage71_equivariant_online_cognition import build_actor


def collect_online_chunk(actor, critic, factor, goal, state, horizon,
                         gamma, gae_lambda, seed):
    """Collect one persistent PPO chunk and its ordered real transitions."""
    torch.manual_seed(seed)
    count = state.shape[0]
    factor_tensor = torch.tensor(factor).view(1, 4).expand(count, -1)
    goal_batch = goal.expand(count, -1)
    states, actions, log_probs, values = [], [], [], []
    rewards, dones, next_states = [], [], []
    successes = 0
    for _ in range(horizon):
        with torch.no_grad():
            action, log_prob, _ = actor.sample(state, goal_batch)
            value = critic(state, goal_batch)
            next_state = ppo.step(
                state, action, factor_tensor[:, 0], factor_tensor[:, 1],
                factor_tensor[:, 2], factor_tensor[:, 3],
            )
            reward, done = ppo.reward_fn(state, next_state, action)
        successes += int(done.sum())
        states.append(state)
        actions.append(action)
        log_probs.append(log_prob)
        values.append(value)
        rewards.append(reward)
        dones.append(done)
        next_states.append(next_state)
        reset = ppo.reset_down_states(count)
        state = torch.where(done.unsqueeze(-1), reset, next_state)
    with torch.no_grad():
        last_value = critic(state, goal_batch)
    states = torch.stack(states)
    actions = torch.stack(actions)
    log_probs = torch.stack(log_probs)
    values = torch.stack(values)
    rewards = torch.stack(rewards)
    dones = torch.stack(dones)
    next_states = torch.stack(next_states)
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros(count)
    next_value = last_value
    for index in reversed(range(horizon)):
        nonterminal = 1.0 - dones[index].float()
        delta = rewards[index] + gamma * next_value * nonterminal - values[index]
        gae = delta + gamma * gae_lambda * nonterminal * gae
        advantages[index] = gae
        next_value = values[index]
    returns = advantages + values
    goals = goal_batch.unsqueeze(0).expand(horizon, -1, -1)
    rollout = ppo.Rollout(
        states.reshape(-1, 6), goals.reshape(-1, 6),
        actions.reshape(-1, 1), log_probs.reshape(-1), values.reshape(-1),
        rewards.reshape(-1), dones.reshape(-1), advantages.reshape(-1),
        returns.reshape(-1),
    )
    trajectories = (states, actions, next_states, dones)
    return rollout, trajectories, state, successes


def update_cognitive_sequences(cognitive, optimizer, replay, updates,
                               batch_size, horizon, seed):
    """Free-running prediction update without crossing episode resets."""
    for parameter in cognitive.parameters():
        parameter.requires_grad = True
    valid_rows = []
    for replay_index, (_, _, _, dones) in enumerate(replay):
        for start in range(dones.shape[0] - horizon + 1):
            valid = ~dones[start:start + horizon - 1].any(dim=0)
            envs = valid.nonzero(as_tuple=False).flatten()
            if envs.numel():
                valid_rows.append(torch.stack([
                    torch.full_like(envs, replay_index),
                    torch.full_like(envs, start),
                    envs,
                ], dim=-1))
    valid_rows = torch.cat(valid_rows)
    generator = torch.Generator().manual_seed(seed)
    losses = []
    for _ in range(updates):
        choice = valid_rows[torch.randint(
            valid_rows.shape[0], (batch_size,), generator=generator,
        )]
        predictions, horizon_losses = [], []
        # Replay chunks are few, so grouping by sample keeps this routine clear
        # and avoids ever treating a reset as a physical transition.
        initial = torch.stack([
            replay[int(row[0])][0][int(row[1]), int(row[2])]
            for row in choice
        ])
        prediction = initial
        for offset in range(horizon):
            action = torch.stack([
                replay[int(row[0])][1][int(row[1]) + offset, int(row[2])]
                for row in choice
            ])
            target = torch.stack([
                replay[int(row[0])][2][int(row[1]) + offset, int(row[2])]
                for row in choice
            ])
            prediction = cognitive(prediction, action)
            predictions.append(prediction)
            horizon_losses.append(F.smooth_l1_loss(prediction, target))
        loss = torch.stack(horizon_losses).mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(cognitive.parameters(), 5.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    for parameter in cognitive.parameters():
        parameter.requires_grad = False
    return {
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "mean_last_20_loss": sum(losses[-20:]) / min(20, len(losses)),
        "valid_sequence_starts": int(valid_rows.shape[0]),
    }


def run(args):
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=True,
    )
    config = checkpoint["config"]
    target_factor = tuple(args.target_factor)
    actor = build_actor(config)
    actor.load_state_dict(checkpoint["actor"])
    critic = ppo.ValueCritic(config["hidden_dim"])
    critic.load_state_dict(checkpoint["critic"])
    for parameter in actor.cognitive.parameters():
        parameter.requires_grad = False
    for parameter in actor.router.parameters():
        parameter.requires_grad = False
    decision_parameters = [
        parameter for name, parameter in actor.named_parameters()
        if not name.startswith("cognitive.")
        and not name.startswith("router.")
    ]
    decision_optimizer = torch.optim.Adam(
        decision_parameters, lr=args.actor_lr,
    )
    critic_optimizer = torch.optim.Adam(
        critic.parameters(), lr=args.critic_lr,
    )
    cognitive_optimizer = torch.optim.Adam(
        actor.cognitive.parameters(), lr=args.cognitive_lr,
    )
    goal = GOAL.view(1, -1)
    before = ppo.evaluate(
        actor, target_factor, goal, args.test_count, args.eval_steps,
        args.eval_seed,
    )
    torch.manual_seed(args.collection_seed)
    state = ppo.reset_down_states(args.num_envs)
    replay, history = [], []
    cumulative_successes = 0
    for iteration in range(args.adaptation_iterations):
        rollout, trajectories, state, successes = collect_online_chunk(
            actor, critic, target_factor, goal, state,
            args.rollout_horizon, args.gamma, args.gae_lambda,
            args.collection_seed + iteration,
        )
        cumulative_successes += successes
        decision_update = ppo.ppo_update(
            actor, critic, rollout, decision_optimizer, critic_optimizer,
            args.clip_ratio, args.value_coef, args.entropy_coef,
            args.ppo_epochs, args.minibatch,
            args.update_seed + iteration,
        )
        replay.append(trajectories)
        cognitive_update = update_cognitive_sequences(
            actor.cognitive, cognitive_optimizer, replay,
            args.cognitive_update_steps, args.cognitive_batch,
            args.cognitive_horizon, args.cognitive_seed + iteration,
        )
        if iteration == 0 or (iteration + 1) % args.eval_every == 0:
            history.append({
                "iteration": iteration + 1,
                "cumulative_real_transitions": (
                    (iteration + 1) * args.num_envs
                    * args.rollout_horizon
                ),
                "chunk_successes": successes,
                "cumulative_collected_successes": cumulative_successes,
                "policy_std": float(actor.log_std.detach().exp()),
                "decision_update": decision_update,
                "cognitive_update": cognitive_update,
                **ppo.evaluate(
                    actor, target_factor, goal, args.test_count,
                    args.eval_steps, args.eval_seed,
                ),
            })
    after = ppo.evaluate(
        actor, target_factor, goal, args.test_count, args.eval_steps,
        args.eval_seed,
    )
    if args.checkpoint_out:
        torch.save({
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "target_factor": target_factor,
            "config": vars(args),
            "source_config": config,
        }, args.checkpoint_out)
    return {
        "target_before": before,
        "adaptation_history": history,
        "target_after": after,
        "cognition_loss": "multistep_real_transition_prediction_only",
        "decision_loss": "ppo_task_reward_only",
        "shared_forward": "fixed_semantic_ProtoKAN_causal_route_update",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default="results/stage70_source_seed0_checkpoint.pt")
    parser.add_argument("--target-factor", type=float, nargs=4,
                        default=[9.8, 0.04, 1.1, 0.9])
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--rollout-horizon", type=int, default=128)
    parser.add_argument("--adaptation-iterations", type=int, default=15)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--minibatch", type=int, default=256)
    parser.add_argument("--cognitive-horizon", type=int, default=8)
    parser.add_argument("--cognitive-update-steps", type=int, default=25)
    parser.add_argument("--cognitive-batch", type=int, default=64)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--cognitive-lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--eval-every", type=int, default=3)
    parser.add_argument("--test-count", type=int, default=32)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--eval-seed", type=int, default=20260721)
    parser.add_argument("--collection-seed", type=int, default=20260723)
    parser.add_argument("--update-seed", type=int, default=20260724)
    parser.add_argument("--cognitive-seed", type=int, default=20260725)
    parser.add_argument("--checkpoint-out", type=str, default="")
    parser.add_argument("--json-out", type=str, default="")
    args = parser.parse_args()
    output = {
        "experiment": "FullJointOnlineEquivariantCausalTransfer",
        "config": vars(args),
        "result": run(args),
    }
    text = json.dumps(output, ensure_ascii=False, indent=2)
    print(text)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")


if __name__ == "__main__":
    main()
