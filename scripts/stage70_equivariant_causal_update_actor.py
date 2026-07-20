"""Direct actor with a fixed-semantic causal route update layer.

The decision network proposes a bounded action sequence.  ProtoKAN rolls that
sequence forward and routes target reachability backward.  The first proposed
action is then corrected by a fixed positive gain times the signed causal
route.  Unlike a learned decoder, this operator has the same meaning when the
route changes magnitude or sign after online cognition updates.
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


class EquivariantCausalUpdateActor(GaussianActorBase):
    """Amortized proposal plus one mandatory signed causal correction."""

    def __init__(self, cognitive, router, horizon=8, hidden_dim=64,
                 temperature=0.08, route_scale=0.04, causal_gain=0.35):
        super().__init__(hidden_dim)
        self.cognitive = cognitive
        self.router = router
        self.horizon = horizon
        self.temperature = temperature
        self.route_scale = route_scale
        self.causal_gain = causal_gain
        self.proposal = _mlp(12, hidden_dim, horizon)

    def plan_and_route(self, state, goal):
        raw_plan = self.proposal(torch.cat([state, goal], dim=-1))
        bounded_plan = torch.tanh(raw_plan)
        routes, predicted_states, scores, weights = temporal_causal_route(
            self.cognitive, state, bounded_plan.unsqueeze(-1),
            self.temperature, self.router,
        )
        # A fixed coordinate system makes the interface Lipschitz and preserves
        # the sign meaning across cognition updates.  No per-sample denominator
        # and no learned decoder can reinterpret or drop this correction.
        signed_route = torch.tanh(routes[:, 0] / self.route_scale)
        corrected_first_raw = (
            raw_plan[:, 0:1] + self.causal_gain * signed_route
        )
        return corrected_first_raw, raw_plan, routes, predicted_states

    def mean_action(self, state, goal):
        corrected, _, _, _ = self.plan_and_route(state, goal)
        return corrected


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

    actor = EquivariantCausalUpdateActor(
        cognitive, router, args.route_horizon, args.hidden_dim,
        args.temperature, args.route_scale, args.causal_gain,
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
    parser.add_argument("--causal-gain", type=float, default=0.35)
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
    parser.add_argument("--checkpoint-out", type=str, default="")
    parser.add_argument("--json-out", type=str, default="")
    args = parser.parse_args()
    output = {
        "architecture": "EquivariantFixedSemanticCausalUpdateActor",
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
