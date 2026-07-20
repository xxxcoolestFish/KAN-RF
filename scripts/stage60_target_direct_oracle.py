"""Train the current embedded decision architecture directly on target dynamics.

This diagnostic removes transfer and cognitive-identification errors.  The
actor receives the exact target transition function from the first update and
is trained only in that same target environment.  High performance means the
decision architecture has enough capacity and the source-to-target decoder is
the bottleneck; low performance points to the decision architecture/loss.
"""

from __future__ import annotations

import argparse
import json

import torch
from torch import nn

from physics_transfer.variants import step
from scripts import stage51_context_cognitive_ppo as base
from scripts.stage23_multistep_terminal_value import GOAL


class ExactTargetCognitive(nn.Module):
    """Differentiable exact target dynamics with the cognition interface."""

    def __init__(self, factor):
        super().__init__()
        self.factor = tuple(float(value) for value in factor)
        self.anchor = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, state, action, context):
        return step(state, action, *self.factor)

    def update_context(self, context, state, action, next_state):
        return context


def train(args):
    torch.manual_seed(args.seed)
    target_factor = tuple(args.target_factor)
    cognitive = ExactTargetCognitive(target_factor)
    actor = base.ContextFiLMActor(cognitive, args.hidden_dim)
    actor.log_std.data.fill_(args.log_std_init)
    critic = base.ContextValueCritic(args.hidden_dim)
    actor_parameters = [
        parameter for name, parameter in actor.named_parameters()
        if not name.startswith("cognitive.")
    ]
    actor_optimizer = torch.optim.Adam(actor_parameters, lr=args.actor_lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)
    goal = GOAL.view(1, -1)
    history = []

    for iteration in range(args.iterations):
        rollout, _, collected_successes = base.collect_rollout(
            actor, critic, cognitive, target_factor, goal, args.num_envs,
            args.rollout_horizon, args.gamma, args.gae_lambda,
            args.seed + iteration,
        )
        update = base.ppo_update(
            actor, critic, rollout, actor_optimizer, critic_optimizer,
            args.clip_ratio, args.value_coef, args.entropy_coef,
            args.ppo_epochs, args.minibatch, args.seed + 10000 + iteration,
        )
        if iteration == 0 or (iteration + 1) % args.eval_every == 0:
            history.append({
                "iteration": iteration + 1,
                "collected_successes": collected_successes,
                "policy_std": float(actor.log_std.detach().exp().item()),
                **update,
                **base.evaluate(
                    actor, cognitive, target_factor, goal, args.test_count,
                    args.eval_steps, args.test_seed + iteration,
                ),
            })

    return {
        "target_factor": target_factor,
        "history": history,
        "final_evaluation": base.evaluate(
            actor, cognitive, target_factor, goal, args.test_count,
            args.eval_steps, args.test_seed + 1000,
        ),
        "actor_parameter_count": sum(p.numel() for p in actor_parameters),
        "critic_parameter_count": sum(p.numel() for p in critic.parameters()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-factor", type=float, nargs=4,
                        default=[13.475, 0.06, 0.90, 1.10])
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--rollout-horizon", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=100)
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
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--test-count", type=int, default=64)
    parser.add_argument("--test-seed", type=int, default=20260720)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-out", type=str, default="")
    args = parser.parse_args()
    output = {
        "architecture": "ExactTargetDynamics_ContextFiLM_DirectTargetPPO",
        "diagnostic": "decision_capacity_without_transfer_or_identification_error",
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
