"""Fair PPO comparison for the full ProtoKAN-embedded decision actor.

The rollout, reward, initial-state distribution, PPO hyperparameters and
evaluation protocol are identical to stage43. The only addition is a
prediction-pretrained ProtoKAN cognition block in the actor forward path.
Cognition parameters are frozen during PPO updates and remain trained only by
the separate next-state prediction loss.
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
    CognitiveEmbeddedGaussianActor,
    GOAL,
    ValueCritic,
    ppo_update,
)
from scripts.stage43_standard_ppo_baseline import (
    collect_rollout,
    evaluate,
)


def train_one(args, seed):
    torch.manual_seed(seed)
    cognitive = SimpleCognitiveKAN()
    cognitive_fit = pretrain_cognitive(
        cognitive, args.cognitive_steps, args.cognitive_batch, seed,
    )
    actor = CognitiveEmbeddedGaussianActor(cognitive, args.hidden_dim)
    actor.log_std.data.fill_(args.log_std_init)
    critic = ValueCritic(args.hidden_dim)
    actor_params = [
        parameter for name, parameter in actor.named_parameters()
        if not name.startswith("cognitive.")
    ]
    actor_optimizer = torch.optim.Adam(actor_params, lr=args.actor_lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)
    goal = GOAL.view(1, -1)
    test_states = _random_states(
        args.test_count,
        generator=torch.Generator().manual_seed(args.test_seed + seed),
    )
    history = []
    for iteration in range(args.iterations):
        rollout, collected_successes = collect_rollout(
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
                actor, PRETRAIN_FACTOR[0], goal, test_states,
                args.eval_steps,
            )
            history.append({
                "iteration": iteration + 1,
                "collected_successes": collected_successes,
                **update,
                **evaluation,
            })
    return {
        "seed": seed,
        "cognitive_fit": cognitive_fit,
        "history": history,
        "source_evaluation": evaluate(
            actor, PRETRAIN_FACTOR[0], goal, test_states, args.eval_steps,
        ),
        "heldout_evaluation": evaluate(
            actor, args.heldout_factor, goal, test_states, args.eval_steps,
        ),
        "actor_parameter_count": sum(parameter.numel() for parameter in actor_params),
        "cognitive_parameter_count": sum(parameter.numel() for parameter in cognitive.parameters()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cognitive-steps", type=int, default=300)
    parser.add_argument("--cognitive-batch", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--rollout-horizon", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch", type=int, default=1024)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--log-std-init", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--test-count", type=int, default=64)
    parser.add_argument("--test-seed", type=int, default=20260719)
    parser.add_argument("--heldout-factor", type=float, nargs=4,
                        default=[9.80, 0.04, 1.10, 0.90])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--json-out", type=str, default="")
    args = parser.parse_args()
    results = [train_one(args, seed) for seed in args.seeds]
    output = {
        "architecture": "StandardPPO_ProtoKANEmbeddedActor",
        "source_factor": PRETRAIN_FACTOR[0],
        "heldout_factor": args.heldout_factor,
        "training_protocol": "identical_to_stage43_except_cognitive_pretraining_and_embedded_forward",
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
