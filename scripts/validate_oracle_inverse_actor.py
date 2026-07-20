"""Train the regularized multi-step cognitive inverse Actor in Oracle dynamics."""

from __future__ import annotations

import argparse
import copy
import json

import torch

from cpbn import OracleAcrobotDynamics
from cpbn.cognitive_inverse import CognitiveInverseActor, CognitiveInverseCritic
from cpbn.cognitive_pullback import local_jacobians_batch
from cpbn.time_varying_tube import plan_continuous_cem_route
from scripts import validate_oracle_pullback_actor as base


def run(args):
    torch.manual_seed(args.seed)
    dynamics = OracleAcrobotDynamics()
    construction = plan_continuous_cem_route(
        dynamics,
        segment_count=args.reference_segments,
        segment_steps=args.segment_steps,
        population=args.reference_population,
        elite_count=args.reference_elite,
        iterations=args.reference_iterations,
        seed=args.reference_seed,
    )
    reference = construction.states.detach().clone()
    route_a, route_b = local_jacobians_batch(dynamics, reference)
    actor = CognitiveInverseActor(
        args.hidden_dim, args.log_std_init, args.effect_scale, args.ridge,
    )
    critic = CognitiveInverseCritic(args.hidden_dim)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)
    history = []
    best_score = -float("inf")
    best_actor = None
    best_iteration = 0
    for iteration in range(args.iterations):
        rollout, collection = base.collect_rollout(
            actor, critic, dynamics, reference, route_a, route_b,
            args, args.seed + iteration,
        )
        update = base.ppo_update(
            actor, critic, reference, route_a, route_b, rollout,
            actor_optimizer, critic_optimizer, args,
            args.seed + 10000 + iteration,
        )
        if iteration == 0 or (iteration + 1) % args.eval_every == 0:
            evaluation = base.evaluate_full(
                actor, dynamics, reference, route_a, route_b, args,
                args.eval_count, args.test_seed + iteration,
            )
            record = {
                "iteration": iteration + 1,
                **collection, **update,
                "full_success_rate": evaluation["success_rate"],
                "full_mean_maximum_height": evaluation["mean_maximum_height"],
            }
            history.append(record)
            print(json.dumps(record), flush=True)
            score = (
                10.0 * evaluation["success_rate"]
                + evaluation["mean_maximum_height"]
            )
            if score > best_score:
                best_score = score
                best_iteration = iteration + 1
                best_actor = copy.deepcopy(actor.state_dict())
    if best_actor is not None:
        actor.load_state_dict(best_actor)
    seeds = [
        args.test_seed + 50000 + index * 1009
        for index in range(args.num_test_seeds)
    ]
    normal, shuffled_corridor, shuffled_jacobians = [], [], []
    for seed in seeds:
        normal.append(base.evaluate_full(
            actor, dynamics, reference, route_a, route_b, args,
            args.final_count, seed,
        ))
        shuffled_corridor.append(base.evaluate_full(
            actor, dynamics, reference, route_a, route_b, args,
            args.final_count, seed, shuffled_corridor=True,
        ))
        shuffled_jacobians.append(base.evaluate_full(
            actor, dynamics, reference, route_a, route_b, args,
            args.final_count, seed, shuffled_jacobians=True,
        ))
    normal_result = base.aggregate_evaluation(normal, args.final_count)
    result = {
        "reference_route": {
            "state_count": int(reference.shape[0]),
            "maximum_height": construction.diagnostics.maximum_height,
            "success_step": construction.diagnostics.success_step,
            "actions_exposed_to_actor": False,
        },
        "history": history,
        "best_iteration": best_iteration,
        "normal": normal_result,
        "shuffled_corridor": base.aggregate_evaluation(
            shuffled_corridor, args.final_count,
        ),
        "shuffled_jacobians": base.aggregate_evaluation(
            shuffled_jacobians, args.final_count,
        ),
        "actor_parameter_count": sum(p.numel() for p in actor.parameters()),
        "zero_control_map_implies_zero_action_mean": True,
        "weak_control_map_compensation_direction": "larger_action",
        "action_teacher_used": False,
        "passed": normal_result["success_rate"] >= 0.95,
    }
    if args.checkpoint_out:
        torch.save({
            "actor": actor.state_dict(),
            "reference": reference,
            "route_A": route_a,
            "route_B": route_b,
            "config": vars(args),
        }, args.checkpoint_out)
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--effect-scale", type=float, default=0.02)
    parser.add_argument("--ridge", type=float, default=1e-4)
    parser.add_argument("--corridor-horizon", type=int, default=12)
    parser.add_argument("--reference-segments", type=int, default=20)
    parser.add_argument("--segment-steps", type=int, default=24)
    parser.add_argument("--reference-population", type=int, default=2048)
    parser.add_argument("--reference-elite", type=int, default=128)
    parser.add_argument("--reference-iterations", type=int, default=12)
    parser.add_argument("--reference-seed", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--rollout-horizon", type=int, default=96)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--minibatch", type=int, default=1024)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--log-std-init", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.005)
    parser.add_argument("--progress-reward", type=float, default=1.0)
    parser.add_argument("--phase-progress-reward", type=float, default=0.04)
    parser.add_argument("--progress-clip", type=float, default=0.25)
    parser.add_argument("--inside-reward", type=float, default=0.02)
    parser.add_argument("--stagnation-penalty", type=float, default=0.01)
    parser.add_argument("--success-reward", type=float, default=3.0)
    parser.add_argument("--action-penalty", type=float, default=0.002)
    parser.add_argument("--corridor-radius", type=float, default=0.12)
    parser.add_argument("--initial-noise", type=float, default=0.025)
    parser.add_argument("--full-initial-noise", type=float, default=0.02)
    parser.add_argument("--phase-backtrack", type=int, default=4)
    parser.add_argument("--phase-advance", type=int, default=12)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-count", type=int, default=16)
    parser.add_argument("--num-test-seeds", type=int, default=5)
    parser.add_argument("--final-count", type=int, default=64)
    parser.add_argument("--full-evaluation-steps", type=int, default=500)
    parser.add_argument("--shuffle-offset", type=int, default=120)
    parser.add_argument("--test-seed", type=int, default=20261001)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--checkpoint-out", default="results/oracle_inverse_actor_seed0.pt",
    )
    parser.add_argument(
        "--json-out", default="results/oracle_inverse_actor_seed0.json",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output = {
        "experiment": "OracleRegularizedCognitiveInverseActorValidation",
        "config": vars(args),
        "result": run(args),
    }
    text = json.dumps(output, ensure_ascii=False, indent=2)
    with open(args.json_out, "w", encoding="utf-8") as handle:
        handle.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
