"""Training-only TD value guidance for the direct policy.

The deployed action path remains the direct policy.  A critic is used only to
provide long-horizon credit during training, using exact source dynamics as a
diagnostic upper bound.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from collections import deque

import torch
from torch import nn

from physics_transfer.full_parameter_transport import FullParameterTransport
from physics_transfer.multifactor_data import _random_states
from physics_transfer.sensitivity_policy import SensitivityMandatoryPolicy
from physics_transfer.variants import step
from scripts.stage15_real_outcome_replay import PRETRAIN_FACTOR
from scripts.stage23_multistep_terminal_value import GOAL, SimpleCognitiveKAN, smooth_tip_height
from scripts.stage27_parameter_transport import pretrain_cognitive
from scripts.stage34_real_transition_goal_loss import goal_potential
from scripts.stage35_plain_policy_ablation import PlainPolicy


class QCritic(nn.Module):
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(13, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state, goal, action):
        if goal.shape[0] == 1 and state.shape[0] != 1:
            goal = goal.expand(state.shape[0], -1)
        return self.network(torch.cat([state, goal, action], dim=-1)).squeeze(-1)


def exact_step(state, action):
    factor = torch.tensor(PRETRAIN_FACTOR[0], dtype=state.dtype).view(1, 4)
    factor = factor.expand(state.shape[0], -1)
    return step(state, action, factor[:, 0], factor[:, 1], factor[:, 2], factor[:, 3])


def reward_fn(state, next_state, action, goal):
    progress = goal_potential(state, goal) - goal_potential(next_state, goal)
    height = smooth_tip_height(next_state)
    success = (height >= 1.0).float()
    return torch.tanh(progress) + 0.2 * torch.tanh(height) + 5.0 * success - 0.01 * action[:, 0].square()


def push_rollout(buffer, policy, goal, batch_size, horizon):
    current = _random_states(batch_size)
    with torch.no_grad():
        for _ in range(horizon):
            action = policy(current, goal)
            next_state = exact_step(current, action)
            reward = reward_fn(current, next_state, action, goal)
            done = (smooth_tip_height(next_state) >= 1.0).float()
            for item in zip(current, action, next_state, reward, done):
                buffer.append(tuple(value.detach() for value in item))
            current = next_state


def sample_buffer(buffer, batch_size):
    items = random.sample(buffer, batch_size)
    state, action, next_state, reward, done = zip(*items)
    return (
        torch.stack(state), torch.stack(action), torch.stack(next_state),
        torch.stack(reward), torch.stack(done),
    )


def soft_update(target, source, tau):
    with torch.no_grad():
        for target_parameter, parameter in zip(target.parameters(), source.parameters()):
            target_parameter.mul_(1.0 - tau).add_(tau * parameter)


def train_actor_critic(policy, goal, steps, batch_size, rollout_horizon, seed):
    if hasattr(policy, "cognitive"):
        for parameter in policy.cognitive.parameters():
            parameter.requires_grad = False
        policy.transport.freeze()
    actor_parameters = [p for p in policy.parameters() if p.requires_grad]
    critic = QCritic()
    target_policy = copy.deepcopy(policy)
    target_critic = copy.deepcopy(critic)
    for parameter in target_policy.parameters():
        parameter.requires_grad = False
    for parameter in target_critic.parameters():
        parameter.requires_grad = False
    actor_optimizer = torch.optim.Adam(actor_parameters, lr=1e-3)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=2e-3)
    random.seed(seed + 3700); torch.manual_seed(seed + 3700)
    buffer = deque(maxlen=20000)
    critic_losses, actor_losses, rewards = [], [], []
    goal = goal.view(1, -1)
    for index in range(steps):
        push_rollout(buffer, policy, goal, batch_size, rollout_horizon)
        if len(buffer) < batch_size:
            continue
        state, action, next_state, reward, done = sample_buffer(buffer, batch_size)
        with torch.no_grad():
            next_action = target_policy(next_state, goal)
            target = reward + 0.98 * (1.0 - done) * target_critic(
                next_state, goal, next_action,
            )
        prediction = critic(state, goal, action)
        critic_loss = torch.nn.functional.smooth_l1_loss(prediction, target)
        critic_optimizer.zero_grad(); critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 5.0)
        critic_optimizer.step()
        critic_losses.append(float(critic_loss.item()))
        rewards.append(float(reward.mean().item()))
        if index % 2 == 0:
            for parameter in critic.parameters():
                parameter.requires_grad = False
            actor_loss = -critic(state, goal, policy(state, goal)).mean()
            actor_optimizer.zero_grad(); actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor_parameters, 5.0)
            actor_optimizer.step()
            for parameter in critic.parameters():
                parameter.requires_grad = True
            actor_losses.append(float(actor_loss.item()))
            soft_update(target_policy, policy, 0.01)
            soft_update(target_critic, critic, 0.01)
    return {
        "critic_first_loss": critic_losses[0],
        "critic_last_loss": critic_losses[-1],
        "critic_mean_last_50_loss": sum(critic_losses[-50:]) / min(50, len(critic_losses)),
        "actor_first_loss": actor_losses[0],
        "actor_last_loss": actor_losses[-1],
        "mean_last_50_reward": sum(rewards[-50:]) / min(50, len(rewards)),
        "buffer_size": len(buffer),
    }


@torch.no_grad()
def evaluate(policy, states, goal, steps):
    current = states.clone()
    factor = torch.tensor(PRETRAIN_FACTOR[0]).view(1, 4).expand(states.shape[0], -1)
    maxima = torch.full((states.shape[0],), -float("inf")); actions = []
    for _ in range(steps):
        action = policy(current, goal); actions.append(action)
        current = step(current, action, factor[:, 0], factor[:, 1], factor[:, 2], factor[:, 3])
        maxima = torch.maximum(maxima, smooth_tip_height(current))
    success = maxima >= 1.0
    return {
        "success_count": int(success.sum().item()),
        "success_rate": float(success.float().mean().item()),
        "mean_max_height": float(maxima.mean().item()),
        "mean_final_goal_potential": float(goal_potential(current, goal).mean().item()),
        "mean_abs_action": float(torch.stack(actions, dim=1).abs().mean().item()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cognitive-steps", type=int, default=200)
    parser.add_argument("--train-steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--rollout-horizon", type=int, default=8)
    parser.add_argument("--rollout-steps", type=int, default=48)
    parser.add_argument("--test-count", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--test-seed", type=int, default=20260719)
    parser.add_argument("--json-out", type=str, default="")
    args = parser.parse_args()
    goal = GOAL.view(1, -1)
    records = []
    for seed in args.seeds:
        torch.manual_seed(seed)
        cognitive = SimpleCognitiveKAN()
        cognitive_fit = pretrain_cognitive(cognitive, args.cognitive_steps, args.batch_size, seed)
        template = SensitivityMandatoryPolicy(cognitive, FullParameterTransport(cognitive))
        states = _random_states(args.test_count,
                                 generator=torch.Generator().manual_seed(args.test_seed))
        record = {"seed": seed, "cognitive_fit": cognitive_fit, "variants": {}}
        for architecture in ("full", "plain"):
            policy = copy.deepcopy(template) if architecture == "full" else PlainPolicy()
            fit = train_actor_critic(policy, goal, args.train_steps,
                                     args.batch_size, args.rollout_horizon, seed)
            record["variants"][architecture] = {
                "fit": fit,
                "source": evaluate(policy, states, goal, args.rollout_steps),
            }
        records.append(record)
    output = {"architecture": "DirectPolicyWithTrainingOnlyTDValue",
              "experiment": "exact_source_dynamics_actor_critic",
              "config": vars(args), "seeds": records}
    text = json.dumps(output, ensure_ascii=False, indent=2)
    print(text)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")


if __name__ == "__main__":
    main()
