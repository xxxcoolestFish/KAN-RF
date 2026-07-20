"""Measure whether a trained embedded actor actually uses cognition outputs."""

from __future__ import annotations

import argparse
import copy
import json

import torch

from physics_transfer.multifactor_data import _random_states
from scripts.stage15_real_outcome_replay import PRETRAIN_FACTOR
from scripts.stage23_multistep_terminal_value import SimpleCognitiveKAN
from scripts.stage27_parameter_transport import pretrain_cognitive
from scripts.stage41_ppo_cognitive_actor import (
    CognitiveEmbeddedGaussianActor,
    GOAL,
    ValueCritic,
    collect_rollout,
    ppo_update,
)


@torch.no_grad()
def probe(actor, swapped, states, goal):
    goal_batch = goal.expand(states.shape[0], -1)
    action, _, _ = actor.sample(states, goal_batch, deterministic=True)
    swapped_action, _, _ = swapped.sample(states, goal_batch, deterministic=True)
    predicted, query = actor.cognitive_prediction(states, goal_batch)
    swapped_predicted, swapped_query = swapped.cognitive_prediction(states, goal_batch)
    return {
        "mean_abs_action_delta_after_cognitive_swap": float((action - swapped_action).abs().mean()),
        "max_abs_action_delta_after_cognitive_swap": float((action - swapped_action).abs().max()),
        "action_std_across_states": float(action.std()),
        "swapped_action_std_across_states": float(swapped_action.std()),
        "mean_abs_predicted_state_delta": float((predicted - swapped_predicted).abs().mean()),
        "mean_abs_query_delta": float((query - swapped_query).abs().mean()),
        "mean_abs_action": float(action.abs().mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cognitive-steps", type=int, default=200)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--rollout-horizon", type=int, default=64)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--minibatch", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test-count", type=int, default=128)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    cognitive = SimpleCognitiveKAN()
    cognitive_fit = pretrain_cognitive(cognitive, args.cognitive_steps, 32, args.seed)
    actor = CognitiveEmbeddedGaussianActor(cognitive, 64)
    critic = ValueCritic(64)
    actor_params = [p for name, p in actor.named_parameters() if not name.startswith("cognitive.")]
    actor_optimizer = torch.optim.Adam(actor_params, lr=3e-4)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)
    goal = GOAL.view(1, -1)
    for iteration in range(args.iterations):
        rollout = collect_rollout(
            actor, critic, PRETRAIN_FACTOR[0], goal, args.num_envs,
            args.rollout_horizon, 0.99, 0.95, args.seed + iteration,
        )
        ppo_update(
            actor, critic, rollout, actor_optimizer, critic_optimizer,
            0.2, 0.5, 0.01, args.ppo_epochs, args.minibatch,
            args.seed + 10000 + iteration,
        )
    swapped = copy.deepcopy(actor)
    swapped.cognitive = SimpleCognitiveKAN()
    states = _random_states(args.test_count, generator=torch.Generator().manual_seed(20260719))
    output = {
        "architecture": "CognitiveForwardSensitivityProbe",
        "cognitive_fit": cognitive_fit,
        "probe": probe(actor, swapped, states, goal),
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
