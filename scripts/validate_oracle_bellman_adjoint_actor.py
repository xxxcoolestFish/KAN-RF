"""Validate a one-step Bellman-adjoint Actor with Oracle cognition."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from cpbn import OracleAcrobotDynamics
from cpbn.cognitive_adjoint import BellmanAdjointActor
from cpbn.corridor_policy import CorridorCritic, future_corridor
from cpbn.time_varying_tube import plan_continuous_cem_route
from scripts import validate_direct_corridor_actor as direct
from scripts import validate_target_online_adaptation as online


class BoundCognitionActor:
    """Expose the direct Actor API while keeping cognition separately owned."""

    def __init__(self, actor, cognition):
        self.actor = actor
        self.cognition = cognition

    def sample(self, state, corridor, deterministic=False):
        return self.actor.sample(
            state, corridor, self.cognition, deterministic=deterministic,
        )


def ppo_update(
    actor, critic, cognition, reference, rollout,
    actor_optimizer, critic_optimizer, args, seed,
):
    generator = torch.Generator().manual_seed(seed)
    advantage = rollout.advantage
    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
    count = rollout.state.shape[0]
    actor_losses, potential_losses, critic_losses = [], [], []
    for _ in range(args.ppo_epochs):
        order = torch.randperm(count, generator=generator)
        for start in range(0, count, args.minibatch):
            index = order[start:start + args.minibatch]
            phase = rollout.phase[index]
            corridor = future_corridor(
                reference, phase, args.corridor_horizon,
            )
            log_prob, entropy = actor.evaluate(
                rollout.state[index], corridor,
                cognition, rollout.action[index],
            )
            ratio = torch.exp(log_prob - rollout.old_log_prob[index])
            raw = ratio * advantage[index]
            clipped = ratio.clamp(
                1.0 - args.clip_ratio, 1.0 + args.clip_ratio,
            ) * advantage[index]
            policy_loss = -torch.minimum(raw, clipped).mean()
            policy_loss = policy_loss - args.entropy_coef * entropy.mean()
            potential_loss = F.smooth_l1_loss(
                actor.potential_value(rollout.state[index], corridor),
                rollout.returns[index],
            )
            actor_loss = policy_loss + args.potential_coef * potential_loss
            value_loss = F.smooth_l1_loss(
                critic(rollout.state[index], corridor),
                rollout.returns[index],
            )
            actor_optimizer.zero_grad(); actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            actor_optimizer.step()
            critic_optimizer.zero_grad(); value_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            critic_optimizer.step()
            actor_losses.append(float(policy_loss.detach()))
            potential_losses.append(float(potential_loss.detach()))
            critic_losses.append(float(value_loss.detach()))
    return {
        "actor_loss": sum(actor_losses) / len(actor_losses),
        "potential_loss": sum(potential_losses) / len(potential_losses),
        "critic_loss": sum(critic_losses) / len(critic_losses),
    }


@torch.no_grad()
def cognition_action_sensitivity(
    actor, correct_cognition, wrong_cognition, reference, args, count, seed,
):
    generator = torch.Generator().manual_seed(seed)
    state, phase = direct.reset_batch(
        reference, count, generator, args.initial_noise,
    )
    corridor = future_corridor(reference, phase, args.corridor_horizon)
    correct, _ = actor.sample(
        state, corridor, correct_cognition, deterministic=True,
    )
    wrong, _ = actor.sample(
        state, corridor, wrong_cognition, deterministic=True,
    )
    return {
        "mean_absolute_action_change": float((correct - wrong).abs().mean()),
        "maximum_absolute_action_change": float((correct - wrong).abs().max()),
    }


def aggregate_feedback(actor, cognition, dynamics, reference, args, offset):
    return online.aggregate_actor(
        BoundCognitionActor(actor, cognition), dynamics, reference,
        args, args, args.final_count, offset,
    )


def run(args):
    torch.manual_seed(args.seed)
    source_cognition = OracleAcrobotDynamics()
    construction = plan_continuous_cem_route(
        source_cognition,
        segment_count=args.reference_segments,
        segment_steps=args.segment_steps,
        population=args.reference_population,
        elite_count=args.reference_elite,
        iterations=args.reference_iterations,
        seed=args.reference_seed,
    )
    reference = construction.states.detach().clone()
    actor = BellmanAdjointActor(
        args.hidden_dim, args.corridor_horizon, args.log_std_init,
        args.ridge, args.log_gain_init,
    )
    critic = CorridorCritic(args.hidden_dim)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic_optimizer = torch.optim.Adam(
        critic.parameters(), lr=args.critic_lr,
    )
    bound_actor = BoundCognitionActor(actor, source_cognition)
    history = []
    best_score = -float("inf")
    best_iteration = 0
    best_actor = copy.deepcopy(actor.state_dict())
    for iteration in range(args.iterations):
        rollout, collection = direct.collect_rollout(
            bound_actor, critic, source_cognition, reference, args,
            args.seed + iteration,
        )
        update = ppo_update(
            actor, critic, source_cognition, reference, rollout,
            actor_optimizer, critic_optimizer, args,
            args.seed + 10000 + iteration,
        )
        if iteration == 0 or (iteration + 1) % args.eval_every == 0:
            evaluation = online.evaluate_once(
                bound_actor, source_cognition, reference, args, args,
                args.eval_count, args.test_seed + iteration,
            )
            record = {
                "iteration": iteration + 1,
                **collection,
                **update,
                "success_rate": evaluation["success_rate"],
                "mean_maximum_height": evaluation["mean_maximum_height"],
                "mean_final_phase": evaluation["mean_final_phase"],
                "log_gain": float(actor.log_gain.detach()),
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
    actor.load_state_dict(best_actor)

    correct = aggregate_feedback(
        actor, source_cognition, source_cognition, reference, args, 100000,
    )
    wrong_cognition = OracleAcrobotDynamics(online.HEAVY_INERTIA_FACTOR)
    wrong = aggregate_feedback(
        actor, wrong_cognition, source_cognition, reference, args, 100000,
    )
    sensitivity = cognition_action_sensitivity(
        actor, source_cognition, wrong_cognition, reference, args,
        512, args.test_seed + 200000,
    )
    result = {
        "reference_route": {
            "state_count": int(reference.shape[0]),
            "maximum_height": construction.diagnostics.maximum_height,
            "success_step": construction.diagnostics.success_step,
            "actions_exposed_to_actor": False,
        },
        "architecture": {
            "time_structure": "one_step_transition_plus_learned_scalar_value",
            "free_vector_costate": False,
            "action_bypass": False,
            "cognition_updated_by_policy_loss": False,
            "action_formula": "gain * dV(F(s,a))/da / (B^T B + ridge) at a=0",
        },
        "history": history,
        "best_iteration": best_iteration,
        "correct_source_cognition": correct,
        "wrong_heavy_inertia_cognition_in_source_environment": wrong,
        "cognition_action_sensitivity": sensitivity,
        "actor_parameter_count": sum(p.numel() for p in actor.parameters()),
        "source_gate_passed": correct["success_rate"] >= args.source_gate,
        "cognition_is_functionally_used": (
            sensitivity["mean_absolute_action_change"]
            >= args.minimum_cognition_action_change
        ),
    }
    if args.checkpoint_out:
        torch.save({
            "actor": actor.state_dict(),
            "reference": reference,
            "config": vars(args),
        }, args.checkpoint_out)
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--corridor-horizon", type=int, default=12)
    parser.add_argument("--ridge", type=float, default=1e-5)
    parser.add_argument("--log-gain-init", type=float, default=-3.0)
    parser.add_argument("--potential-coef", type=float, default=0.25)
    parser.add_argument("--source-gate", type=float, default=0.90)
    parser.add_argument(
        "--minimum-cognition-action-change", type=float, default=0.02,
    )
    parser.add_argument("--reference-segments", type=int, default=20)
    parser.add_argument("--segment-steps", type=int, default=24)
    parser.add_argument("--reference-population", type=int, default=2048)
    parser.add_argument("--reference-elite", type=int, default=128)
    parser.add_argument("--reference-iterations", type=int, default=12)
    parser.add_argument("--reference-seed", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=45)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--rollout-horizon", type=int, default=96)
    parser.add_argument("--ppo-epochs", type=int, default=3)
    parser.add_argument("--minibatch", type=int, default=768)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--log-std-init", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.005)
    parser.add_argument("--progress-reward", type=float, default=1.0)
    parser.add_argument("--progress-clip", type=float, default=0.25)
    parser.add_argument("--inside-reward", type=float, default=0.08)
    parser.add_argument("--success-reward", type=float, default=3.0)
    parser.add_argument("--action-penalty", type=float, default=0.002)
    parser.add_argument("--corridor-radius", type=float, default=0.12)
    parser.add_argument("--initial-noise", type=float, default=0.025)
    parser.add_argument("--full-initial-noise", type=float, default=0.02)
    parser.add_argument("--phase-backtrack", type=int, default=4)
    parser.add_argument("--phase-advance", type=int, default=12)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--eval-count", type=int, default=16)
    parser.add_argument("--num-test-seeds", type=int, default=3)
    parser.add_argument("--final-count", type=int, default=32)
    parser.add_argument("--evaluation-steps", type=int, default=500)
    parser.add_argument("--test-seed", type=int, default=20261301)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--checkpoint-out",
        default="results/oracle_bellman_adjoint_actor_seed0.pt",
    )
    parser.add_argument(
        "--json-out",
        default="results/oracle_bellman_adjoint_actor_seed0.json",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output = {
        "experiment": "OracleBellmanAdjointActorValidation",
        "config": vars(args),
        "result": run(args),
    }
    text = json.dumps(output, ensure_ascii=False, indent=2)
    Path(args.json_out).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
