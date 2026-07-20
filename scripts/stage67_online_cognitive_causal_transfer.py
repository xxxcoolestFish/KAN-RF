"""Online physics transfer through an ordered ProtoKAN causal policy.

The source policy is trained exactly as in Stage 66.  After the dynamics are
changed, decision and nonlinear-router parameters are frozen.  Only the
embedded cognitive ProtoKAN is fitted to transitions actually observed in the
new environment.  Any policy recovery therefore has to travel through the
changed cognitive trajectory and its native temporal causal graph.
"""

from __future__ import annotations

import argparse
import json

import torch

from kanrf.protokan_causal_router_stable import (
    StableProtoKANNonlinearEdgeRouter,
)
from scripts import stage43_standard_ppo_baseline as ppo
from scripts import stage46_online_cognitive_ppo as online
from scripts import stage64b_nonlinear_causal_router_transfer as router_training
from scripts.stage15_real_outcome_replay import PRETRAIN_FACTOR
from scripts.stage23_multistep_terminal_value import GOAL, SimpleCognitiveKAN
from scripts.stage27_parameter_transport import pretrain_cognitive
from scripts.stage66_sequence_causal_actor import ProtoKANSequenceCausalActor


def train(args):
    torch.manual_seed(args.seed)
    source_factor = PRETRAIN_FACTOR[0]
    target_factor = tuple(args.target_factor)
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
    source_history = []

    for iteration in range(args.source_iterations):
        rollout, collected_successes = ppo.collect_rollout(
            actor, critic, source_factor, goal, args.num_envs,
            args.rollout_horizon, args.gamma, args.gae_lambda,
            args.seed + 1000 + iteration,
        )
        update = ppo.ppo_update(
            actor, critic, rollout, actor_optimizer, critic_optimizer,
            args.clip_ratio, args.value_coef, args.entropy_coef,
            args.ppo_epochs, args.minibatch,
            args.seed + 10000 + iteration,
        )
        if iteration == 0 or (iteration + 1) % args.source_eval_every == 0:
            source_history.append({
                "iteration": iteration + 1,
                "collected_successes": collected_successes,
                "policy_std": float(actor.log_std.detach().exp().item()),
                **update,
                **ppo.evaluate(
                    actor, source_factor, goal, args.test_count,
                    args.eval_steps, args.test_seed + iteration,
                ),
            })

    source_final = ppo.evaluate(
        actor, source_factor, goal, args.test_count, args.eval_steps,
        args.test_seed + 1000,
    )
    target_before = ppo.evaluate(
        actor, target_factor, goal, args.test_count, args.eval_steps,
        args.test_seed + 2000,
    )

    # From this point onward no decision parameter receives a gradient.
    for parameter in decision_parameters:
        parameter.requires_grad = False
    for parameter in critic.parameters():
        parameter.requires_grad = False
    cognitive_optimizer = torch.optim.Adam(
        cognitive.parameters(), lr=args.cognitive_lr,
    )
    adaptation_history = []
    for iteration in range(args.adaptation_iterations):
        _, transitions, collected_successes = online.collect_rollout(
            actor, critic, target_factor, goal, args.num_envs,
            args.rollout_horizon, args.gamma, args.gae_lambda,
            args.seed + 3000 + iteration,
        )
        cognitive_loss = online.update_cognitive(
            cognitive, cognitive_optimizer, transitions,
            args.cognitive_update_epochs, args.cognitive_minibatch,
            args.seed + 20000 + iteration,
        )
        if iteration == 0 or (iteration + 1) % args.adaptation_eval_every == 0:
            adaptation_history.append({
                "iteration": iteration + 1,
                "real_transition_count": (
                    (iteration + 1) * args.num_envs * args.rollout_horizon
                ),
                "collected_successes": collected_successes,
                "cognitive_update_loss": cognitive_loss,
                **ppo.evaluate(
                    actor, target_factor, goal, args.test_count,
                    args.eval_steps, args.test_seed + 3000 + iteration,
                ),
            })

    target_after = ppo.evaluate(
        actor, target_factor, goal, args.test_count, args.eval_steps,
        args.test_seed + 4000,
    )
    if args.checkpoint_out:
        torch.save({
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "source_factor": source_factor,
            "target_factor": target_factor,
            "config": vars(args),
        }, args.checkpoint_out)
    return {
        "cognitive_fit": cognitive_fit,
        "router_fit": router_fit,
        "source_history": source_history,
        "source_final": source_final,
        "target_before_online_cognition": target_before,
        "adaptation_history": adaptation_history,
        "target_after_online_cognition": target_after,
        "decision_updated_on_target": False,
        "router_updated_on_target": False,
        "cognitive_updated_on_target": True,
        "trainable_decision_parameters": sum(
            parameter.numel() for parameter in decision_parameters
        ),
        "cognitive_parameters": sum(
            parameter.numel() for parameter in cognitive.parameters()
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
    parser.add_argument("--source-iterations", type=int, default=60)
    parser.add_argument("--adaptation-iterations", type=int, default=15)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--minibatch", type=int, default=256)
    parser.add_argument("--cognitive-minibatch", type=int, default=512)
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
    parser.add_argument("--source-eval-every", type=int, default=30)
    parser.add_argument("--adaptation-eval-every", type=int, default=5)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--test-count", type=int, default=32)
    parser.add_argument("--test-seed", type=int, default=20260720)
    parser.add_argument("--target-factor", type=float, nargs=4,
                        default=[9.80, 0.04, 1.10, 0.90])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-out", type=str, default="")
    parser.add_argument("--json-out", type=str, default="")
    args = parser.parse_args()
    output = {
        "architecture": "OrderedCausalActor_OnlineCognitionOnlyTransfer",
        "source_factor": PRETRAIN_FACTOR[0],
        "target_factor": args.target_factor,
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
