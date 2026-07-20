"""First validation of full-parameter cognitive-to-policy transport.

The cognitive ProtoKAN is trained only on one-step transition prediction.  A
parameter transport module then consumes the complete cognitive parameter
vector and generates low-rank updates for a direct action policy.  The
transport is calibrated with a frozen cognitive teacher, after which the
decision loss is optimized separately.  No action probes or runtime action
optimization are used.
"""

from __future__ import annotations

import argparse
import copy
import json
import math

import torch
import torch.nn.functional as F

from physics_transfer.multifactor_data import _random_states
from physics_transfer.parameter_transport import (
    ParameterTransport,
    TransportReconstructionHead,
    TransportedMLPPolicy,
)
from physics_transfer.transition_data import sample_transition_batch
from physics_transfer.variants import step
from scripts.stage15_real_outcome_replay import PRETRAIN_FACTOR
from scripts.stage2_lowrank_loss_ablation import FACTORS, HELDOUT
from scripts.stage23_multistep_terminal_value import (
    GOAL,
    SimpleCognitiveKAN,
    smooth_tip_height,
    task_cost,
)


def _stats(losses):
    return {
        "first_loss": float(losses[0]),
        "last_loss": float(losses[-1]),
        "mean_last_20_loss": float(sum(losses[-20:]) / min(20, len(losses))),
    }


def pretrain_cognitive(model, steps: int, batch_size: int, seed: int):
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    losses = []
    for _ in range(steps):
        batch = sample_transition_batch(batch_size, PRETRAIN_FACTOR)
        prediction = model(batch["state"], batch["action"])
        loss = F.smooth_l1_loss(prediction, batch["next_state"])
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        losses.append(loss.item())
    return _stats(losses)


def fit_transport(cognitive, transport, steps: int, batch_size: int, seed: int):
    """Calibrate transport without mixing it into the policy objective."""
    for parameter in cognitive.parameters():
        parameter.requires_grad = False
    cognitive.eval()
    readout = TransportReconstructionHead(transport.code_dim)
    optimizer = torch.optim.Adam(
        list(transport.parameters()) + list(readout.parameters()), lr=2e-3,
    )
    torch.manual_seed(seed + 100)
    losses = []
    for _ in range(steps):
        batch = sample_transition_batch(batch_size, PRETRAIN_FACTOR)
        with torch.no_grad():
            teacher = cognitive(batch["state"], batch["action"])
        code = transport(cognitive)["code"]
        prediction = readout(batch["state"], batch["action"], code)
        loss = F.smooth_l1_loss(prediction, teacher)
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(transport.parameters()) + list(readout.parameters()), 5.0,
        )
        optimizer.step()
        losses.append(loss.item())
    return readout, _stats(losses)


def train_policy(policy, cognitive, goal, steps: int, batch_size: int,
                 max_horizon: int, seed: int, use_transport: bool):
    for parameter in cognitive.parameters():
        parameter.requires_grad = False
    for parameter in policy.transport.parameters():
        parameter.requires_grad = False
    parameters = [
        policy.base_weight_1, policy.base_bias_1,
        policy.base_weight_2, policy.base_bias_2,
    ]
    optimizer = torch.optim.Adam(parameters, lr=2e-3)
    torch.manual_seed(seed + (200 if use_transport else 300))
    losses, horizons = [], []
    levels = [h for h in (1, 2, 4, 8) if h <= max_horizon]
    for index in range(steps):
        horizon = levels[min(len(levels) - 1,
                             int(index / max(1, steps - 1) * len(levels)))]
        current = _random_states(batch_size)
        costs, actions = [], []
        for _ in range(horizon):
            action = policy(current, goal, use_transport=use_transport)
            next_state = cognitive(current, action)
            costs.append(task_cost(next_state, action))
            actions.append(action)
            current = next_state
        discounted = torch.stack(
            [(0.95 ** t) * value for t, value in enumerate(costs)], dim=1,
        ).sum(dim=1)
        action_stack = torch.stack(actions, dim=1)
        smooth = (
            (action_stack[:, 1:] - action_stack[:, :-1]).square().mean()
            if horizon > 1 else torch.zeros((), dtype=current.dtype)
        )
        loss = discounted.mean() + 0.02 * smooth
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 5.0)
        optimizer.step()
        losses.append(loss.item()); horizons.append(horizon)
    return {**_stats(losses), "final_horizon": horizons[-1]}


@torch.no_grad()
def prediction_error(cognitive, factor, batch_size=512):
    batch = sample_transition_batch(batch_size, (factor,))
    prediction = cognitive(batch["state"], batch["action"])
    return {
        "one_step_mse": float(F.mse_loss(prediction, batch["next_state"]).item()),
        "one_step_smooth_l1": float(
            F.smooth_l1_loss(prediction, batch["next_state"]).item()
        ),
    }


@torch.no_grad()
def evaluate(policy, states, factor, goal, rollout_steps, use_transport=True):
    current = states.detach().clone()
    factor_tensor = torch.tensor(factor, dtype=current.dtype).view(1, 4)
    factor_tensor = factor_tensor.expand(current.shape[0], -1)
    maxima = torch.full((current.shape[0],), -float("inf"))
    actions = []
    for _ in range(rollout_steps):
        action = policy(current, goal, use_transport=use_transport)
        actions.append(action)
        current = step(
            current, action, factor_tensor[:, 0], factor_tensor[:, 1],
            factor_tensor[:, 2], factor_tensor[:, 3],
        )
        maxima = torch.maximum(maxima, smooth_tip_height(current))
    success = maxima >= 1.0
    return {
        "success_count": int(success.sum().item()),
        "success_rate": float(success.float().mean().item()),
        "mean_max_height": float(maxima.mean().item()),
        "mean_abs_action": float(torch.stack(actions, 1).abs().mean().item()),
    }


def adapt_cognitive(cognitive, factor, steps: int, batch_size: int, seed: int):
    """Online prediction-only update on a changed physical environment."""
    torch.manual_seed(seed + 400)
    optimizer = torch.optim.Adam(cognitive.parameters(), lr=1e-3)
    losses = []
    for _ in range(steps):
        batch = sample_transition_batch(batch_size, (factor,))
        prediction = cognitive(batch["state"], batch["action"])
        loss = F.smooth_l1_loss(prediction, batch["next_state"])
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(cognitive.parameters(), 5.0)
        optimizer.step()
        losses.append(loss.item())
    return _stats(losses)


def factor_distance(source, factor):
    scale = (5.0, 0.05, 0.2, 0.2)
    return float(math.sqrt(sum(((a - b) / s) ** 2
                               for a, b, s in zip(source, factor, scale))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cognitive-steps", type=int, default=250)
    parser.add_argument("--transport-steps", type=int, default=200)
    parser.add_argument("--policy-steps", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-horizon", type=int, default=4)
    parser.add_argument("--rollout-steps", type=int, default=64)
    parser.add_argument("--test-count", type=int, default=20)
    parser.add_argument("--test-seed", type=int, default=20260718)
    parser.add_argument("--adapt-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json-out", type=str, default="")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    cognitive = SimpleCognitiveKAN()
    cognitive_fit = pretrain_cognitive(
        cognitive, args.cognitive_steps, args.batch_size, args.seed,
    )
    transport = ParameterTransport(
        cognitive, layer_shapes=((32, 12), (1, 32)), rank=4,
    )
    readout, transport_fit = fit_transport(
        cognitive, transport, args.transport_steps, args.batch_size, args.seed,
    )
    del readout

    transported = TransportedMLPPolicy(cognitive, transport)
    plain = copy.deepcopy(transported)
    goal = GOAL.view(1, -1)
    transported_fit = train_policy(
        transported, cognitive, goal, args.policy_steps, args.batch_size,
        args.max_horizon, args.seed, True,
    )
    plain_fit = train_policy(
        plain, plain.cognitive, goal, args.policy_steps, args.batch_size,
        args.max_horizon, args.seed, False,
    )

    generator = torch.Generator().manual_seed(args.test_seed)
    states = _random_states(args.test_count, generator=generator)
    source = PRETRAIN_FACTOR[0]
    factors = [("source", source)]
    factors += [(f"factor_{i + 1}", factor)
                for i, factor in enumerate(FACTORS[1:])]
    factors += [(f"heldout_{i + 1}", factor)
                for i, factor in enumerate(HELDOUT)]
    results = []
    for label, factor in factors:
        results.append({
            "label": label,
            "factor": factor,
            "normalized_distance_from_source": factor_distance(source, factor),
            "cognitive_prediction": prediction_error(cognitive, factor),
            "transport_policy": evaluate(
                transported, states, factor, goal, args.rollout_steps, True,
            ),
            "plain_policy": evaluate(
                plain, states, factor, goal, args.rollout_steps, False,
            ),
        })

    changed = HELDOUT[0]
    adapted_cognitive = copy.deepcopy(cognitive)
    before_code = transported.transport_code().detach()
    adaptation_fit = adapt_cognitive(
        adapted_cognitive, changed, args.adapt_steps, args.batch_size, args.seed,
    )
    adapted_policy = copy.deepcopy(transported)
    adapted_policy.cognitive = adapted_cognitive
    after_code = adapted_policy.transport_code().detach()
    online_result = {
        "factor": changed,
        "prediction_before": prediction_error(cognitive, changed),
        "prediction_after": prediction_error(adapted_cognitive, changed),
        "adaptation_fit": adaptation_fit,
        "transport_code_shift_l2": float(torch.linalg.vector_norm(after_code - before_code).item()),
        "decision_before": evaluate(transported, states, changed, goal, args.rollout_steps, True),
        "decision_after": evaluate(adapted_policy, states, changed, goal, args.rollout_steps, True),
    }

    output = {
        "architecture": "FullParameterTransportLowRankPolicy",
        "training_factor": source,
        "cognitive_loss_and_decision_loss_separate": True,
        "runtime_action_probes": "none",
        "cognitive_parameter_count": sum(p.numel() for p in cognitive.parameters()),
        "transport_fit": transport_fit,
        "cognitive_fit": cognitive_fit,
        "transported_policy_fit": transported_fit,
        "plain_policy_fit": plain_fit,
        "results": results,
        "online_parameter_refresh": online_result,
        "test_seed": args.test_seed,
    }
    text = json.dumps(output, ensure_ascii=False, indent=2)
    print(text)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")


if __name__ == "__main__":
    main()
