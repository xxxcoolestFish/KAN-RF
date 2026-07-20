"""Audit why online prediction fitting can hurt the causal policy."""

from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F

from kanrf.protokan_causal_router_stable import (
    StableProtoKANNonlinearEdgeRouter,
)
from physics_transfer.transition_data import (
    sample_transition_batch,
    sample_transition_sequence_batch,
)
from scripts import stage43_standard_ppo_baseline as ppo
from scripts.stage23_multistep_terminal_value import GOAL, SimpleCognitiveKAN
from scripts.stage27_parameter_transport import pretrain_cognitive
from scripts.stage66_sequence_causal_actor import ProtoKANSequenceCausalActor


def build_actor(config):
    cognitive = SimpleCognitiveKAN(hidden_dim=32, n_prototypes=8)
    router = StableProtoKANNonlinearEdgeRouter(delta=config["edge_delta"])
    return ProtoKANSequenceCausalActor(
        cognitive, router, config["route_horizon"], config["hidden_dim"],
        config["temperature"],
    )


@torch.no_grad()
def one_step_metrics(cognitive, factor, batch_size, seed):
    torch.manual_seed(seed)
    batch = sample_transition_batch(batch_size, [factor])
    prediction = cognitive(batch["state"], batch["action"])
    return {
        "smooth_l1": float(F.smooth_l1_loss(
            prediction, batch["next_state"],
        )),
        "mse": float(F.mse_loss(prediction, batch["next_state"])),
    }


@torch.no_grad()
def free_rollout_metrics(cognitive, factor, batch_size, horizon, seed):
    torch.manual_seed(seed)
    batch = sample_transition_sequence_batch(batch_size, horizon, [factor])
    state = batch["state"][:, 0]
    errors = []
    for index in range(horizon):
        state = cognitive(state, batch["action"][:, index])
        errors.append((state - batch["next_state"][:, index]).square().mean())
    return {
        "per_horizon_mse": [float(value) for value in errors],
        "final_mse": float(errors[-1]),
        "mean_mse": float(torch.stack(errors).mean()),
    }


@torch.no_grad()
def proposal_rollout_metrics(actor, factor, batch_size, seed):
    torch.manual_seed(seed)
    state = ppo.reset_down_states(batch_size)
    goal = GOAL.view(1, -1).expand(batch_size, -1)
    actions = torch.tanh(actor.proposal(torch.cat([state, goal], dim=-1)))
    batch_factor = torch.tensor(factor).view(1, 4).expand(batch_size, -1)
    model_state = state
    true_state = state
    errors = []
    for index in range(actor.horizon):
        action = actions[:, index:index + 1]
        model_state = actor.cognitive(model_state, action)
        true_state = ppo.step(
            true_state, action, batch_factor[:, 0], batch_factor[:, 1],
            batch_factor[:, 2], batch_factor[:, 3],
        )
        errors.append((model_state - true_state).square().mean())
    return {
        "per_horizon_mse": [float(value) for value in errors],
        "final_mse": float(errors[-1]),
        "mean_mse": float(torch.stack(errors).mean()),
    }


@torch.no_grad()
def interface_drift(source_actor, adapted_actor, batch_size, seed):
    torch.manual_seed(seed)
    state = ppo.reset_down_states(batch_size)
    goal = GOAL.view(1, -1).expand(batch_size, -1)
    source_features = source_actor.causal_features(state, goal)
    adapted_features = adapted_actor.causal_features(state, goal)
    source_action = source_actor.mean_action(state, goal)
    adapted_action = adapted_actor.mean_action(state, goal)
    feature_delta = adapted_features - source_features
    cosine = F.cosine_similarity(source_features, adapted_features, dim=-1)
    return {
        "feature_rmse": float(feature_delta.square().mean().sqrt()),
        "feature_relative_rmse": float(
            feature_delta.square().mean().sqrt()
            / source_features.square().mean().sqrt().clamp_min(1e-8)
        ),
        "feature_cosine_mean": float(cosine.mean()),
        "feature_cosine_min": float(cosine.min()),
        "mean_absolute_action_change": float(
            (adapted_action - source_action).abs().mean()
        ),
        "maximum_absolute_action_change": float(
            (adapted_action - source_action).abs().max()
        ),
    }


def run(args):
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=True,
    )
    config = checkpoint["config"]
    adapted_actor = build_actor(config)
    adapted_actor.load_state_dict(checkpoint["actor"])

    # The cognition module was frozen throughout source PPO.  Replaying its
    # deterministic pretraining therefore reconstructs the exact pre-adaptation
    # cognitive state while retaining the learned decision/router checkpoint.
    torch.manual_seed(config["seed"])
    source_cognitive = SimpleCognitiveKAN(hidden_dim=32, n_prototypes=8)
    source_fit = pretrain_cognitive(
        source_cognitive, config["cognitive_steps"],
        config["cognitive_batch"], config["seed"],
    )
    source_actor = build_actor(config)
    source_actor.load_state_dict(checkpoint["actor"])
    source_actor.cognitive.load_state_dict(source_cognitive.state_dict())
    source_actor.eval(); adapted_actor.eval()
    target_factor = tuple(checkpoint["target_factor"])
    goal = GOAL.view(1, -1)

    return {
        "source_cognitive_reconstruction_fit": source_fit,
        "target_evaluation_same_starts": {
            "before_online_update": ppo.evaluate(
                source_actor, target_factor, goal, args.test_count,
                args.eval_steps, args.eval_seed,
            ),
            "after_online_update": ppo.evaluate(
                adapted_actor, target_factor, goal, args.test_count,
                args.eval_steps, args.eval_seed,
            ),
        },
        "broad_random_target_one_step": {
            "before_online_update": one_step_metrics(
                source_actor.cognitive, target_factor, args.audit_batch,
                args.audit_seed,
            ),
            "after_online_update": one_step_metrics(
                adapted_actor.cognitive, target_factor, args.audit_batch,
                args.audit_seed,
            ),
        },
        "broad_random_target_free_rollout": {
            "before_online_update": free_rollout_metrics(
                source_actor.cognitive, target_factor, args.audit_batch,
                args.rollout_audit_horizon, args.audit_seed + 1,
            ),
            "after_online_update": free_rollout_metrics(
                adapted_actor.cognitive, target_factor, args.audit_batch,
                args.rollout_audit_horizon, args.audit_seed + 1,
            ),
        },
        "policy_proposal_target_rollout": {
            "before_online_update": proposal_rollout_metrics(
                source_actor, target_factor, args.audit_batch,
                args.audit_seed + 2,
            ),
            "after_online_update": proposal_rollout_metrics(
                adapted_actor, target_factor, args.audit_batch,
                args.audit_seed + 2,
            ),
        },
        "interface_drift_on_down_states": interface_drift(
            source_actor, adapted_actor, args.audit_batch,
            args.audit_seed + 3,
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default="results/stage67_seed0_checkpoint.pt")
    parser.add_argument("--audit-batch", type=int, default=512)
    parser.add_argument("--rollout-audit-horizon", type=int, default=8)
    parser.add_argument("--test-count", type=int, default=64)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--eval-seed", type=int, default=20260721)
    parser.add_argument("--audit-seed", type=int, default=20260722)
    parser.add_argument("--json-out", type=str, default="")
    args = parser.parse_args()
    output = {
        "experiment": "OnlineCognitionDistributionAndInterfaceDriftAudit",
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
