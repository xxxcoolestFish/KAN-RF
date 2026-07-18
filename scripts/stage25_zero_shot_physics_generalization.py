"""Zero-shot physical-parameter generalization benchmark.

Only the source physical factor is used during cognitive and decision
training.  The trained embedded controller is then frozen and evaluated in
several target environments.  The same initial states are used for every
factor so that changes in success rate are attributable to physics changes,
not to a different test-state sample.
"""

from __future__ import annotations

import argparse
import json
import math

import torch
import torch.nn.functional as F

from physics_transfer.multifactor_data import _random_states
from physics_transfer.transition_data import sample_transition_batch
from scripts.stage15_real_outcome_replay import PRETRAIN_FACTOR
from scripts.stage2_lowrank_loss_ablation import FACTORS, HELDOUT
from scripts.stage24_bellman_energy_controller import (
    EnergyKAN,
    PositiveCostToGoKAN,
    SimpleCognitiveKAN,
    alternate_td_energy,
    evaluate,
    fit_one_step_energy,
    pretrain_cognitive_multistep,
    warmup_value,
)
from scripts.stage24_bellman_energy_controller import GOAL


def prediction_error(cognitive, factor, batch_size=512):
    batch = sample_transition_batch(batch_size, (factor,))
    with torch.no_grad():
        prediction = cognitive(batch["state"], batch["action"])
        smooth_l1 = F.smooth_l1_loss(prediction, batch["next_state"])
        mse = F.mse_loss(prediction, batch["next_state"])
    return {
        "one_step_smooth_l1": float(smooth_l1.item()),
        "one_step_mse": float(mse.item()),
    }


def factor_distance(source, factor):
    # Normalize by representative scales so mass, damping and link lengths
    # contribute comparably to the reported distance.
    scale = (5.0, 0.05, 0.2, 0.2)
    return float(math.sqrt(sum(((a - b) / s) ** 2 for a, b, s in zip(source, factor, scale))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cognitive-steps", type=int, default=300)
    parser.add_argument("--max-cognitive-horizon", type=int, default=8)
    parser.add_argument("--energy-fit-steps", type=int, default=100)
    parser.add_argument("--value-warmup-steps", type=int, default=100)
    parser.add_argument("--alternate-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--inner-steps", type=int, default=3)
    parser.add_argument("--action-step-size", type=float, default=0.20)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--rollout-steps", type=int, default=64)
    parser.add_argument("--test-count", type=int, default=20)
    parser.add_argument("--test-seed", type=int, default=20260718)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source = PRETRAIN_FACTOR[0]
    goal = GOAL.view(1, -1)

    # The only factor visible to all training routines is PRETRAIN_FACTOR.
    cognitive = SimpleCognitiveKAN()
    cognitive_fit = pretrain_cognitive_multistep(
        cognitive, args.cognitive_steps, 32,
        args.max_cognitive_horizon, args.seed,
    )
    energy = EnergyKAN()
    energy_fit = fit_one_step_energy(
        cognitive, energy, args.energy_fit_steps, args.batch_size, args.seed, goal,
    )
    value = PositiveCostToGoKAN()
    target_value, value_fit = warmup_value(
        cognitive, energy, value, args.value_warmup_steps, args.batch_size,
        args.seed, goal, args.gamma, args.inner_steps, args.action_step_size,
    )
    decision_fit = alternate_td_energy(
        cognitive, energy, value, target_value, args.alternate_steps,
        args.batch_size, args.horizon, args.seed, goal, args.gamma,
        args.inner_steps, args.action_step_size,
    )

    generator = torch.Generator().manual_seed(args.test_seed)
    test_states = _random_states(args.test_count, generator=generator)
    factors = [("source", source)]
    factors.extend((f"factor_{index + 1}", factor) for index, factor in enumerate(FACTORS[1:]))
    factors.extend((f"heldout_{index + 1}", factor) for index, factor in enumerate(HELDOUT))

    results = []
    for label, factor in factors:
        results.append({
            "label": label,
            "factor": factor,
            "normalized_distance_from_source": factor_distance(source, factor),
            "cognitive_prediction": prediction_error(cognitive, factor),
            "decision": evaluate(
                cognitive, energy, value, test_states, factor, goal,
                args.rollout_steps, args.gamma, args.inner_steps,
                args.action_step_size,
            ),
        })

    print(json.dumps({
        "architecture": "EmbeddedControllerZeroShotPhysicsGeneralization",
        "training_factor": source,
        "training_uses_only_source_factor": True,
        "adaptation_during_evaluation": "none",
        "teacher_usage": "none",
        "cognitive_fit": cognitive_fit,
        "energy_fit": energy_fit,
        "value_fit": value_fit,
        "decision_fit": decision_fit,
        "results": results,
        "test_seed": args.test_seed,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
