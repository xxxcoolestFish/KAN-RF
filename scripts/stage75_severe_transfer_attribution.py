"""Attribute severe-target failure to critic, cognition, or decision updates."""

from __future__ import annotations

import argparse
import json

import torch

from kanrf.protokan_causal_router_stable import (
    StableProtoKANNonlinearEdgeRouter,
)
from scripts import stage43_standard_ppo_baseline as ppo
from scripts.stage23_multistep_terminal_value import GOAL, SimpleCognitiveKAN
from scripts.stage72_joint_online_equivariant_transfer import (
    collect_online_chunk,
    update_cognitive_sequences,
)
from scripts.stage73_psd_causal_preconditioner_actor import (
    PSDCausalPreconditionerActor,
)


def build_actor(config):
    cognitive = SimpleCognitiveKAN(hidden_dim=32, n_prototypes=8)
    router = StableProtoKANNonlinearEdgeRouter(delta=config["edge_delta"])
    return PSDCausalPreconditionerActor(
        cognitive, router, config["route_horizon"], config["hidden_dim"],
        config["temperature"], config["route_scale"], config["step_size"],
        config["metric_rank"], config["min_diagonal"],
        config["max_diagonal"],
    )


def _correlation(left, right):
    left = left - left.mean()
    right = right - right.mean()
    return float(
        (left * right).mean()
        / (left.square().mean().sqrt() * right.square().mean().sqrt())
        .clamp_min(1e-8)
    )


@torch.no_grad()
def audit_critic(actor, critic, factor, count, horizon, gamma, seed):
    """Compare critic values with target TD residuals and finite MC returns."""
    torch.manual_seed(seed)
    state = ppo.reset_down_states(count)
    factor_tensor = torch.tensor(factor).view(1, 4).expand(count, -1)
    goal = GOAL.view(1, -1).expand(count, -1)
    values, rewards, dones, next_values = [], [], [], []
    successes = 0
    for _ in range(horizon):
        action, _, _ = actor.sample(state, goal)
        value = critic(state, goal)
        next_state = ppo.step(
            state, action, factor_tensor[:, 0], factor_tensor[:, 1],
            factor_tensor[:, 2], factor_tensor[:, 3],
        )
        reward, done = ppo.reward_fn(state, next_state, action)
        next_value = critic(next_state, goal)
        values.append(value)
        rewards.append(reward)
        dones.append(done)
        next_values.append(next_value)
        successes += int(done.sum())
        state = torch.where(
            done.unsqueeze(-1), ppo.reset_down_states(count), next_state,
        )
    values = torch.stack(values)
    rewards = torch.stack(rewards)
    dones = torch.stack(dones)
    next_values = torch.stack(next_values)
    td_error = rewards + gamma * next_values * (~dones).float() - values

    # Finite-horizon Monte Carlo return.  The final 100 time steps are excluded
    # from the correlation so truncation does not dominate the audit.
    returns = torch.zeros_like(rewards)
    running = torch.zeros(count)
    for index in reversed(range(horizon)):
        running = rewards[index] + gamma * running * (~dones[index]).float()
        returns[index] = running
    usable = max(1, horizon - min(100, horizon // 4))
    flat_value = values[:usable].reshape(-1)
    flat_return = returns[:usable].reshape(-1)
    residual = flat_return - flat_value
    return {
        "factor": list(factor),
        "collected_successes": successes,
        "td_error_mean": float(td_error.mean()),
        "td_error_rmse": float(td_error.square().mean().sqrt()),
        "td_error_abs_mean": float(td_error.abs().mean()),
        "mc_value_return_correlation": _correlation(flat_value, flat_return),
        "mc_value_bias": float(residual.mean()),
        "mc_value_rmse": float(residual.square().mean().sqrt()),
        "mc_explained_variance": float(
            1.0 - residual.var(unbiased=False)
            / flat_return.var(unbiased=False).clamp_min(1e-8)
        ),
    }


def run_mode(name, checkpoint, source_config, args, reset_critic,
             update_decision, update_cognition, mode_index):
    actor = build_actor(source_config)
    actor.load_state_dict(checkpoint["actor"])
    if reset_critic:
        torch.manual_seed(args.seed + 50000 + mode_index)
        critic = ppo.ValueCritic(source_config["hidden_dim"])
    else:
        critic = ppo.ValueCritic(source_config["hidden_dim"])
        critic.load_state_dict(checkpoint["critic"])

    for parameter in actor.cognitive.parameters():
        parameter.requires_grad = False
    for parameter in actor.router.parameters():
        parameter.requires_grad = False
    decision_parameters = [
        parameter for parameter_name, parameter in actor.named_parameters()
        if not parameter_name.startswith("cognitive.")
        and not parameter_name.startswith("router.")
    ]
    for parameter in decision_parameters:
        parameter.requires_grad = update_decision
    for parameter in critic.parameters():
        parameter.requires_grad = update_decision
    decision_optimizer = None
    critic_optimizer = None
    if update_decision:
        decision_optimizer = torch.optim.Adam(
            decision_parameters, lr=args.actor_lr,
        )
        critic_optimizer = torch.optim.Adam(
            critic.parameters(), lr=args.critic_lr,
        )
    cognitive_optimizer = None
    if update_cognition:
        cognitive_optimizer = torch.optim.Adam(
            actor.cognitive.parameters(), lr=args.cognitive_lr,
        )

    target_factor = tuple(args.target_factor)
    goal = GOAL.view(1, -1)
    before = ppo.evaluate(
        actor, target_factor, goal, args.test_count, args.eval_steps,
        args.eval_seed,
    )
    torch.manual_seed(args.collection_seed)
    state = ppo.reset_down_states(args.num_envs)
    replay, history = [], []
    cumulative_successes = 0
    for iteration in range(args.adaptation_iterations):
        rollout, trajectories, state, successes = collect_online_chunk(
            actor, critic, target_factor, goal, state,
            args.rollout_horizon, args.gamma, args.gae_lambda,
            args.collection_seed + iteration,
        )
        cumulative_successes += successes
        decision_update = None
        if update_decision:
            decision_update = ppo.ppo_update(
                actor, critic, rollout, decision_optimizer, critic_optimizer,
                args.clip_ratio, args.value_coef, args.entropy_coef,
                args.ppo_epochs, args.minibatch,
                args.update_seed + iteration,
            )
        cognitive_update = None
        if update_cognition:
            replay.append(trajectories)
            cognitive_update = update_cognitive_sequences(
                actor.cognitive, cognitive_optimizer, replay,
                args.cognitive_update_steps, args.cognitive_batch,
                args.cognitive_horizon, args.cognitive_seed + iteration,
            )
        if iteration == 0 or (iteration + 1) % args.eval_every == 0:
            history.append({
                "iteration": iteration + 1,
                "cumulative_real_transitions": (
                    (iteration + 1) * args.num_envs
                    * args.rollout_horizon
                ),
                "chunk_successes": successes,
                "cumulative_collected_successes": cumulative_successes,
                "policy_std": float(actor.log_std.detach().exp()),
                "decision_update": decision_update,
                "cognitive_update": cognitive_update,
                **ppo.evaluate(
                    actor, target_factor, goal, args.test_count,
                    args.eval_steps, args.eval_seed,
                ),
            })
    return {
        "name": name,
        "reset_critic": reset_critic,
        "update_decision": update_decision,
        "update_cognition": update_cognition,
        "target_before": before,
        "history": history,
        "target_after": ppo.evaluate(
            actor, target_factor, goal, args.test_count, args.eval_steps,
            args.eval_seed,
        ),
    }


def run(args):
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=True,
    )
    source_config = checkpoint["config"]
    audit_actor = build_actor(source_config)
    audit_actor.load_state_dict(checkpoint["actor"])
    audit_critic_model = ppo.ValueCritic(source_config["hidden_dim"])
    audit_critic_model.load_state_dict(checkpoint["critic"])
    critic_audit = {
        "source_environment": audit_critic(
            audit_actor, audit_critic_model,
            tuple(checkpoint["source_factor"]), args.audit_envs,
            args.audit_horizon, args.gamma, args.audit_seed,
        ),
        "severe_target": audit_critic(
            audit_actor, audit_critic_model, tuple(args.target_factor),
            args.audit_envs, args.audit_horizon, args.gamma,
            args.audit_seed,
        ),
    }
    modes = [
        ("decision_only_keep_source_critic", False, True, False),
        ("decision_only_reset_critic", True, True, False),
        ("joint_reset_critic", True, True, True),
        ("cognition_only", False, False, True),
    ]
    results = []
    for index, (name, reset, decision, cognition) in enumerate(modes):
        results.append(run_mode(
            name, checkpoint, source_config, args, reset, decision,
            cognition, index,
        ))
    return {
        "critic_calibration": critic_audit,
        "pilot_modes": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default="results/stage73_source_seed0_checkpoint.pt")
    parser.add_argument("--target-factor", type=float, nargs=4,
                        default=[13.475, 0.06, 0.90, 1.10])
    parser.add_argument("--audit-envs", type=int, default=32)
    parser.add_argument("--audit-horizon", type=int, default=500)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--rollout-horizon", type=int, default=128)
    parser.add_argument("--adaptation-iterations", type=int, default=6)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--minibatch", type=int, default=256)
    parser.add_argument("--cognitive-horizon", type=int, default=8)
    parser.add_argument("--cognitive-update-steps", type=int, default=25)
    parser.add_argument("--cognitive-batch", type=int, default=64)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--cognitive-lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--eval-every", type=int, default=3)
    parser.add_argument("--test-count", type=int, default=32)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--eval-seed", type=int, default=20260721)
    parser.add_argument("--audit-seed", type=int, default=20260726)
    parser.add_argument("--collection-seed", type=int, default=20260723)
    parser.add_argument("--update-seed", type=int, default=20260724)
    parser.add_argument("--cognitive-seed", type=int, default=20260725)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-out", type=str, default="")
    args = parser.parse_args()
    output = {
        "experiment": "SevereTargetCriticCognitionDecisionAttribution",
        "config": vars(args),
        "result": run(args),
    }
    text = json.dumps(output, ensure_ascii=False, indent=2)
    print(text)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")


if __name__ == "__main__":
    main()
