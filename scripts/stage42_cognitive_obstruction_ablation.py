"""Ablate whether the embedded cognition parameters obstruct PPO learning.

All conditions use the same PPO implementation, reward, rollout protocol and
decision-head size. Only the state representation entering the decision head
changes:

* direct: raw state + goal actor;
* trained_proto: pretrained ProtoKAN prediction + goal;
* random_proto: untrained ProtoKAN prediction + goal;
* identity_receiver: state passthrough + goal, using the same embedded-head
  shape as the ProtoKAN conditions.

For the trained embedded actor, we also swap the cognitive parameters after
training without changing the decision head. This measures whether the actor
actually depends on the learned physical parameters.
"""

from __future__ import annotations

import argparse
import copy
import json

import torch
from torch import nn

from physics_transfer.multifactor_data import _random_states
from scripts.stage15_real_outcome_replay import PRETRAIN_FACTOR
from scripts.stage23_multistep_terminal_value import SimpleCognitiveKAN
from scripts.stage27_parameter_transport import pretrain_cognitive
from scripts.stage41_ppo_cognitive_actor import (
    GOAL,
    CognitiveEmbeddedGaussianActor,
    DirectGaussianActor,
    ValueCritic,
    collect_rollout,
    evaluate,
    ppo_update,
)


class IdentityCognitive(nn.Module):
    """Architecture-matched receiver that passes the current state through."""

    def forward(self, state, action):
        return state


def make_cognitive(condition: str, steps: int, batch: int, seed: int):
    if condition == "identity_receiver":
        return IdentityCognitive(), None
    model = SimpleCognitiveKAN()
    fit = None
    if condition == "trained_proto":
        fit = pretrain_cognitive(model, steps, batch, seed)
    elif condition != "random_proto":
        raise ValueError(condition)
    return model, fit


def train_condition(args, condition: str, seed: int):
    torch.manual_seed(seed)
    goal = GOAL.view(1, -1)
    if condition == "direct":
        cognitive, cognitive_fit = None, None
        actor = DirectGaussianActor(args.hidden_dim)
    else:
        cognitive, cognitive_fit = make_cognitive(
            condition, args.cognitive_steps, args.cognitive_batch, seed,
        )
        actor = CognitiveEmbeddedGaussianActor(cognitive, args.hidden_dim)
    critic = ValueCritic(args.hidden_dim)
    # Cognition parameters are never included in the PPO optimizer.
    actor_params = [
        p for name, p in actor.named_parameters()
        if not name.startswith("cognitive.")
    ]
    actor_optimizer = torch.optim.Adam(actor_params, lr=args.actor_lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)
    test_states = _random_states(
        args.test_count,
        generator=torch.Generator().manual_seed(args.test_seed + seed),
    )
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
            evaluation = evaluate(
                actor, PRETRAIN_FACTOR[0], goal, test_states, args.eval_steps,
            )
            history.append({"iteration": iteration + 1, **update, **evaluation})
    source = evaluate(actor, PRETRAIN_FACTOR[0], goal, test_states, args.eval_steps)
    result = {
        "condition": condition,
        "seed": seed,
        "cognitive_fit": cognitive_fit,
        "history": history,
        "source_evaluation": source,
        "actor_parameter_count": sum(p.numel() for p in actor_params),
        "cognitive_parameter_count": (
            sum(p.numel() for p in cognitive.parameters())
            if cognitive is not None else 0
        ),
    }
    if condition == "trained_proto":
        # Same trained decision head, only the cognition parameters are
        # swapped after learning. A large drop means the actor actually uses
        # the learned physical parameters; a gain means those parameters may
        # be obstructive.
        random_cognitive = SimpleCognitiveKAN()
        swapped = copy.deepcopy(actor)
        swapped.cognitive = random_cognitive
        result["swap_random_cognition"] = evaluate(
            swapped, PRETRAIN_FACTOR[0], goal, test_states, args.eval_steps,
        )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cognitive-steps", type=int, default=200)
    parser.add_argument("--cognitive-batch", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--rollout-horizon", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--minibatch", type=int, default=256)
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
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--json-out", type=str, default="")
    args = parser.parse_args()
    conditions = ("direct", "identity_receiver", "random_proto", "trained_proto")
    results = []
    for condition_index, condition in enumerate(conditions):
        for seed in args.seeds:
            results.append(train_condition(
                args, condition, seed + condition_index * 10000,
            ))
    output = {
        "architecture": "CognitiveParameterObstructionAblation",
        "source_factor": PRETRAIN_FACTOR[0],
        "conditions": conditions,
        "config": vars(args),
        "results": results,
    }
    text = json.dumps(output, ensure_ascii=False, indent=2)
    print(text)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")


if __name__ == "__main__":
    main()
