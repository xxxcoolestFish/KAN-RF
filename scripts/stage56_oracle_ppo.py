"""Stage 56: oracle target dynamics with the existing decision network.

This is a diagnostic upper-bound experiment.  The cognitive predictor is
replaced by the exact differentiable Acrobot transition function, while the
actor, PPO loss, rollout protocol, and target evaluation remain unchanged.
The actor never receives the hidden physical factors directly.
"""

from __future__ import annotations

import argparse
import json

import torch
from torch import nn

from physics_transfer.variants import step
from scripts import stage51_context_cognitive_ppo as base
from scripts.stage15_real_outcome_replay import PRETRAIN_FACTOR


class OracleCognitive(nn.Module):
    """Exact target transition function with the cognitive-model interface."""

    def __init__(self, factor):
        super().__init__()
        self.factor = tuple(float(value) for value in factor)
        # A frozen placeholder keeps the base optimizer interface valid.
        self.anchor = nn.Parameter(torch.zeros(1), requires_grad=False)

    def set_factor(self, factor):
        self.factor = tuple(float(value) for value in factor)

    def forward(self, state, action, context=None):
        return step(state, action, *self.factor)

    def update_context(self, context, state, action, next_state):
        # The exact oracle already knows the target dynamics; no history
        # inference is needed for this diagnostic.
        return context


def no_pretrain(cognitive, steps, batch_size, sequence_steps, seed):
    return {
        "first_loss": 0.0,
        "last_loss": 0.0,
        "mean_last_20_loss": 0.0,
        "oracle": True,
    }


def no_cognitive_update(cognitive, optimizer, transitions, epochs, seed):
    return 0.0


_base_evaluate = base.evaluate


def evaluate_with_target(actor, cognitive, factor, goal, count, steps, seed):
    cognitive.set_factor(factor)
    return _base_evaluate(actor, cognitive, factor, goal, count, steps, seed)


base.ContextCognitiveKAN = OracleCognitive
base.pretrain_cognitive = no_pretrain
base.update_cognitive = no_cognitive_update
base.evaluate = evaluate_with_target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cognitive-steps", type=int, default=0)
    parser.add_argument("--cognitive-batch", type=int, default=32)
    parser.add_argument("--sequence-steps", type=int, default=1)
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
    result = base.train(args)
    output = {
        "architecture": "ExactTargetDynamics_ExistingContextFiLMDecision",
        "source_factor": PRETRAIN_FACTOR[0],
        "heldout_factor": args.heldout_factor,
        "config": vars(args),
        "result": result,
    }
    text = json.dumps(output, ensure_ascii=False, indent=2)
    print(text)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")


if __name__ == "__main__":
    main()
