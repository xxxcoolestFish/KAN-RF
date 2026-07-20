"""Actor with a learned PSD preconditioner over the causal route sequence."""

from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F
from torch import nn

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


class PSDCausalPreconditionerActor(GaussianActorBase):
    """Proposal plus a guaranteed ascent-aligned sequence correction.

    A learned positive-semidefinite matrix mixes temporal causal messages.  It
    is conditioned only on sign-invariant route magnitude and cognitive
    trajectory features.  Therefore route sign changes cannot be arbitrarily
    reinterpreted, while the policy can still learn temporal preconditioning.
    """

    def __init__(self, cognitive, router, horizon=8, hidden_dim=64,
                 temperature=0.08, route_scale=0.04, step_size=0.2,
                 metric_rank=2, min_diagonal=0.1, max_diagonal=2.0):
        super().__init__(hidden_dim)
        self.cognitive = cognitive
        self.router = router
        self.horizon = horizon
        self.temperature = temperature
        self.route_scale = route_scale
        self.step_size = step_size
        self.metric_rank = metric_rank
        self.min_diagonal = min_diagonal
        self.max_diagonal = max_diagonal
        self.proposal = _mlp(12, hidden_dim, horizon)
        # Per step: |route|, predicted state(6), score, temporal weight.
        metric_input = 9 * horizon + 6
        self.metric_trunk = nn.Sequential(
            nn.Linear(metric_input, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
        )
        self.diagonal_head = nn.Linear(hidden_dim, horizon)
        self.factor_head = nn.Linear(hidden_dim, horizon * metric_rank)
        nn.init.zeros_(self.factor_head.weight)
        nn.init.zeros_(self.factor_head.bias)

    def plan_and_route(self, state, goal):
        raw_plan = self.proposal(torch.cat([state, goal], dim=-1))
        routes, predicted_states, scores, weights = temporal_causal_route(
            self.cognitive, state, torch.tanh(raw_plan).unsqueeze(-1),
            self.temperature, self.router,
        )
        route_vector = routes.squeeze(-1) / self.route_scale
        metric_input = torch.cat([
            routes.abs(), predicted_states, scores.unsqueeze(-1),
            weights.unsqueeze(-1),
        ], dim=-1).flatten(start_dim=1)
        metric_input = torch.cat([metric_input, goal], dim=-1)
        hidden = self.metric_trunk(metric_input)
        diagonal = self.min_diagonal + self.max_diagonal * torch.sigmoid(
            self.diagonal_head(hidden)
        )
        factor = 0.25 * torch.tanh(self.factor_head(hidden)).view(
            -1, self.horizon, self.metric_rank,
        )
        low_rank = torch.bmm(factor, factor.transpose(1, 2))
        metric = low_rank + torch.diag_embed(diagonal)
        correction = torch.bmm(
            metric, route_vector.unsqueeze(-1),
        ).squeeze(-1)
        corrected_plan = raw_plan + self.step_size * correction
        # This quantity is nonnegative up to floating-point error because the
        # learned metric is PSD.  It is exposed for audit, not used as a loss.
        alignment = (route_vector * correction).sum(dim=-1)
        return corrected_plan, raw_plan, routes, predicted_states, alignment

    def mean_action(self, state, goal):
        corrected, _, _, _, _ = self.plan_and_route(state, goal)
        return corrected[:, 0:1]


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
    actor = PSDCausalPreconditionerActor(
        cognitive, router, args.route_horizon, args.hidden_dim,
        args.temperature, args.route_scale, args.step_size,
        args.metric_rank, args.min_diagonal, args.max_diagonal,
    )
    actor.log_std.data.fill_(args.log_std_init)
    critic = ppo.ValueCritic(args.hidden_dim)
    decision_parameters = [
        parameter for name, parameter in actor.named_parameters()
        if not name.startswith("cognitive.")
        and not name.startswith("router.")
    ]
    actor_optimizer = torch.optim.Adam(
        decision_parameters, lr=args.actor_lr,
    )
    critic_optimizer = torch.optim.Adam(
        critic.parameters(), lr=args.critic_lr,
    )
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
            evaluation = ppo.evaluate(
                actor, PRETRAIN_FACTOR[0], goal, args.test_count,
                args.eval_steps, args.test_seed + iteration,
            )
            with torch.no_grad():
                audit_state = ppo.reset_down_states(args.audit_count)
                audit_goal = goal.expand(args.audit_count, -1)
                _, _, _, _, alignment = actor.plan_and_route(
                    audit_state, audit_goal,
                )
            history.append({
                "iteration": iteration + 1,
                "collected_successes": collected_successes,
                "policy_std": float(actor.log_std.detach().exp()),
                "minimum_causal_alignment": float(alignment.min()),
                "mean_causal_alignment": float(alignment.mean()),
                **update,
                **evaluation,
            })
    final_evaluation = ppo.evaluate(
        actor, PRETRAIN_FACTOR[0], goal, args.test_count, args.eval_steps,
        args.test_seed + 1000,
    )
    if args.checkpoint_out:
        torch.save({
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "source_factor": PRETRAIN_FACTOR[0],
            "config": vars(args),
        }, args.checkpoint_out)
    return {
        "cognitive_fit": cognitive_fit,
        "router_fit": router_fit,
        "history": history,
        "final_evaluation": final_evaluation,
        "trainable_decision_parameters": sum(
            parameter.numel() for parameter in decision_parameters
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
    parser.add_argument("--route-scale", type=float, default=0.04)
    parser.add_argument("--step-size", type=float, default=0.2)
    parser.add_argument("--metric-rank", type=int, default=2)
    parser.add_argument("--min-diagonal", type=float, default=0.1)
    parser.add_argument("--max-diagonal", type=float, default=2.0)
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
    parser.add_argument("--audit-count", type=int, default=64)
    parser.add_argument("--test-seed", type=int, default=20260720)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-out", type=str, default="")
    parser.add_argument("--json-out", type=str, default="")
    args = parser.parse_args()
    output = {
        "architecture": "PSDTemporalCausalPreconditionerActor",
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
