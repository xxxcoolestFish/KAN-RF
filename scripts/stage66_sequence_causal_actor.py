"""PPO actor driven by the complete ordered ProtoKAN causal trajectory.

Stage 65 proved that an aggregated causal representation helps at low sample
budgets, but it discarded most temporal order before the decision head.  This
variant retains the signed route, normalized signed route, predicted state,
reachability score and temporal attention weight at every imagined step.
"""

from __future__ import annotations

import argparse
import json

import torch

from kanrf.protokan_causal_router_stable import (
    StableProtoKANNonlinearEdgeRouter,
)
from kanrf.protokan_temporal_route import temporal_causal_route
from scripts import stage43_standard_ppo_baseline as ppo
from scripts import stage64b_nonlinear_causal_router_transfer as router_training
from scripts.stage15_real_outcome_replay import PRETRAIN_FACTOR
from scripts.stage23_multistep_terminal_value import GOAL, SimpleCognitiveKAN
from scripts.stage27_parameter_transport import pretrain_cognitive
from scripts.stage41_ppo_cognitive_actor import GaussianActorBase, _mlp


class ProtoKANSequenceCausalActor(GaussianActorBase):
    """Proposal -> full cognitive causal sequence -> direct action."""

    def __init__(self, cognitive, router, horizon=8, hidden_dim=64,
                 temperature=0.08):
        super().__init__(hidden_dim)
        self.cognitive = cognitive
        self.router = router
        self.horizon = horizon
        self.temperature = temperature
        self.proposal = _mlp(12, hidden_dim, horizon)
        # Per imagined step: raw route (1), normalized signed route (1),
        # predicted state (6), reachability score (1), attention weight (1).
        # The goal is appended once.  Neither the raw current state nor the
        # proposed actions have a direct path to the executed action.
        feature_dim = 10 * horizon + 6
        self.decision_head = _mlp(feature_dim, hidden_dim, 1)

    def causal_features(self, state, goal):
        proposal = torch.tanh(self.proposal(torch.cat([state, goal], dim=-1)))
        routes, predicted_states, scores, weights = temporal_causal_route(
            self.cognitive, state, proposal.unsqueeze(-1), self.temperature,
            self.router,
        )
        # Normalize per sample without deleting physical scale: the raw route
        # remains alongside this phase/sign-focused representation.  Detaching
        # the denominator avoids an unnecessary cross-time gradient shortcut.
        route_scale = routes.detach().square().mean(
            dim=1, keepdim=True,
        ).sqrt().clamp_min(1e-4)
        normalized_routes = routes / route_scale
        ordered = torch.cat([
            routes,
            normalized_routes,
            predicted_states,
            scores.unsqueeze(-1),
            weights.unsqueeze(-1),
        ], dim=-1)
        return torch.cat([ordered.flatten(start_dim=1), goal], dim=-1)

    def mean_action(self, state, goal):
        return self.decision_head(self.causal_features(state, goal))


def train(args):
    torch.manual_seed(args.seed)
    cognitive = SimpleCognitiveKAN(hidden_dim=32, n_prototypes=8)
    cognitive_fit = pretrain_cognitive(
        cognitive, args.cognitive_steps, args.cognitive_batch, args.seed,
    )
    router = StableProtoKANNonlinearEdgeRouter(delta=args.edge_delta)
    router_fit = router_training.experiment.train_router(
        cognitive, router, args.router_steps, args.router_batch,
        args.route_horizon, args.temperature, args.finite_delta,
        args.seed + 100,
    )
    for parameter in cognitive.parameters():
        parameter.requires_grad = False
    for parameter in router.parameters():
        parameter.requires_grad = False

    actor = ProtoKANSequenceCausalActor(
        cognitive, router, args.route_horizon, args.hidden_dim,
        args.temperature,
    )
    actor.log_std.data.fill_(args.log_std_init)
    critic = ppo.ValueCritic(args.hidden_dim)
    actor_parameters = [
        parameter for name, parameter in actor.named_parameters()
        if not name.startswith("cognitive.")
        and not name.startswith("router.")
    ]
    actor_optimizer = torch.optim.Adam(actor_parameters, lr=args.actor_lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)
    goal = GOAL.view(1, -1)
    history = []

    for iteration in range(args.policy_iterations):
        rollout, collected_successes = ppo.collect_rollout(
            actor, critic, PRETRAIN_FACTOR[0], goal, args.num_envs,
            args.rollout_horizon, args.gamma, args.gae_lambda,
            args.seed + 1000 + iteration,
        )
        update = ppo.ppo_update(
            actor, critic, rollout, actor_optimizer, critic_optimizer,
            args.clip_ratio, args.value_coef, args.entropy_coef,
            args.ppo_epochs, args.minibatch,
            args.seed + 10000 + iteration,
        )
        if iteration == 0 or (iteration + 1) % args.eval_every == 0:
            history.append({
                "iteration": iteration + 1,
                "collected_successes": collected_successes,
                "policy_std": float(actor.log_std.detach().exp().item()),
                **update,
                **ppo.evaluate(
                    actor, PRETRAIN_FACTOR[0], goal, args.test_count,
                    args.eval_steps, args.test_seed + iteration,
                ),
            })

    return {
        "cognitive_fit": cognitive_fit,
        "router_fit": router_fit,
        "history": history,
        "final_evaluation": ppo.evaluate(
            actor, PRETRAIN_FACTOR[0], goal, args.test_count,
            args.eval_steps, args.test_seed + 1000,
        ),
        "trainable_actor_parameters": sum(
            parameter.numel() for parameter in actor_parameters
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cognitive-steps", type=int, default=300)
    parser.add_argument("--cognitive-batch", type=int, default=32)
    parser.add_argument("--router-steps", type=int, default=100)
    parser.add_argument("--router-batch", type=int, default=16)
    parser.add_argument("--route-horizon", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.08)
    parser.add_argument("--finite-delta", type=float, default=0.25)
    parser.add_argument("--edge-delta", type=float, default=0.05)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--rollout-horizon", type=int, default=128)
    parser.add_argument("--policy-iterations", type=int, default=60)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--minibatch", type=int, default=256)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--log-std-init", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--eval-every", type=int, default=15)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--test-count", type=int, default=64)
    parser.add_argument("--test-seed", type=int, default=20260720)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-out", type=str, default="")
    args = parser.parse_args()
    output = {
        "architecture": "ProtoKAN_OrderedCausalSequence_DirectActor",
        "source_factor": PRETRAIN_FACTOR[0],
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
