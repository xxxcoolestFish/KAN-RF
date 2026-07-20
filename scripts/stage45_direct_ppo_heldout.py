"""Direct PPO baseline with the same held-out evaluation as stage44."""

from __future__ import annotations

import argparse
import json

import torch

from physics_transfer.multifactor_data import _random_states
from physics_transfer.variants import step
from scripts.stage15_real_outcome_replay import PRETRAIN_FACTOR
from scripts.stage23_multistep_terminal_value import GOAL
from scripts.stage41_ppo_cognitive_actor import DirectGaussianActor, ValueCritic, ppo_update
from scripts.stage43_standard_ppo_baseline import collect_rollout, evaluate


def train_one(args, seed):
    torch.manual_seed(seed)
    actor = DirectGaussianActor(args.hidden_dim)
    actor.log_std.data.fill_(args.log_std_init)
    critic = ValueCritic(args.hidden_dim)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)
    goal = GOAL.view(1, -1)
    states = _random_states(args.test_count, generator=torch.Generator().manual_seed(args.test_seed + seed))
    for iteration in range(args.iterations):
        rollout, _ = collect_rollout(
            actor, critic, PRETRAIN_FACTOR[0], goal, args.num_envs,
            args.rollout_horizon, args.gamma, args.gae_lambda, seed + iteration,
        )
        ppo_update(
            actor, critic, rollout, actor_optimizer, critic_optimizer,
            args.clip_ratio, args.value_coef, args.entropy_coef,
            args.ppo_epochs, args.minibatch, seed + 10000 + iteration,
        )
    return {
        "seed": seed,
        "source_evaluation": evaluate(actor, PRETRAIN_FACTOR[0], goal, states, args.eval_steps),
        "heldout_evaluation": evaluate(actor, args.heldout_factor, goal, states, args.eval_steps),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--rollout-horizon", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--minibatch", type=int, default=512)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--log-std-init", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--test-count", type=int, default=64)
    parser.add_argument("--test-seed", type=int, default=20260719)
    parser.add_argument("--heldout-factor", type=float, nargs=4, default=[9.80, 0.04, 1.10, 0.90])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--json-out", type=str, default="")
    args = parser.parse_args()
    output = {
        "architecture": "StandardPPO_DirectMLP_HeldoutMatched",
        "source_factor": PRETRAIN_FACTOR[0],
        "heldout_factor": args.heldout_factor,
        "config": vars(args),
        "results": [train_one(args, seed) for seed in args.seeds],
    }
    text = json.dumps(output, ensure_ascii=False, indent=2)
    print(text)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")


if __name__ == "__main__":
    main()
