"""Gradient-free CEM reachability diagnostic on the target dynamics."""

from __future__ import annotations

import argparse
import json

import torch

from physics_transfer.variants import step
from scripts import stage51_context_cognitive_ppo as base


@torch.no_grad()
def score_sequences(initial_states, factor, action_logits, action_repeat):
    state_count, population, blocks = action_logits.shape
    state = initial_states[:, None, :].expand(-1, population, -1).reshape(-1, 6).clone()
    factor_tensor = torch.tensor(factor).view(1, 4).expand(state.shape[0], -1)
    maximum = torch.full((state.shape[0],), -float("inf"))
    actions = torch.tanh(action_logits)
    for block in range(blocks):
        action = actions[:, :, block].reshape(-1, 1)
        for _ in range(action_repeat):
            state = step(
                state, action, factor_tensor[:, 0], factor_tensor[:, 1],
                factor_tensor[:, 2], factor_tensor[:, 3],
            )
            maximum = torch.maximum(maximum, base.tip_height(state))
    return maximum.view(state_count, population)


@torch.no_grad()
def cem_search(initial_states, factor, blocks, action_repeat, population,
               elites, iterations, seed):
    torch.manual_seed(seed)
    state_count = initial_states.shape[0]
    mean = torch.zeros(state_count, blocks)
    std = torch.full_like(mean, 1.5)
    best_height = torch.full((state_count,), -float("inf"))
    best_actions = torch.zeros(state_count, blocks)
    history = []

    for iteration in range(iterations):
        samples = mean[:, None, :] + std[:, None, :] * torch.randn(
            state_count, population, blocks,
        )
        scores = score_sequences(initial_states, factor, samples, action_repeat)
        values, indices = scores.topk(elites, dim=1)
        elite_logits = samples.gather(1, indices.unsqueeze(-1).expand(-1, -1, blocks))
        mean = elite_logits.mean(dim=1)
        std = elite_logits.std(dim=1).clamp_min(0.08)
        improved = values[:, 0] > best_height
        best_height = torch.where(improved, values[:, 0], best_height)
        candidate = torch.tanh(elite_logits[:, 0])
        best_actions = torch.where(improved.unsqueeze(-1), candidate, best_actions)
        history.append({
            "iteration": iteration + 1,
            "success_count": int((best_height >= 1.0).sum()),
            "mean_best_height": float(best_height.mean()),
            "maximum_height": float(best_height.max()),
        })
    return best_height, best_actions, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-factor", type=float, nargs=4,
                        default=[13.475, 0.06, 0.90, 1.10])
    parser.add_argument("--test-count", type=int, default=8)
    parser.add_argument("--test-seed", type=int, default=20260720)
    parser.add_argument("--blocks", type=int, default=32)
    parser.add_argument("--action-repeat", type=int, default=16)
    parser.add_argument("--population", type=int, default=256)
    parser.add_argument("--elites", type=int, default=24)
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json-out", type=str, default="")
    args = parser.parse_args()

    torch.manual_seed(args.test_seed)
    initial_states = base.reset_down_states(args.test_count)
    height, actions, history = cem_search(
        initial_states, tuple(args.target_factor), args.blocks,
        args.action_repeat, args.population, args.elites, args.iterations,
        args.seed,
    )
    success = height >= 1.0
    output = {
        "target_factor": args.target_factor,
        "initial_state_distribution": "hanging_down_with_small_noise",
        "test_count": args.test_count,
        "physical_horizon": args.blocks * args.action_repeat,
        "blocks": args.blocks,
        "action_repeat": args.action_repeat,
        "population": args.population,
        "elites": args.elites,
        "iterations": args.iterations,
        "feasible_count_found": int(success.sum()),
        "feasible_rate_found": float(success.float().mean()),
        "best_height": height.tolist(),
        "best_action_blocks": actions.tolist(),
        "history": history,
        "note": "CEM success proves reachability; CEM failure would remain inconclusive.",
    }
    text = json.dumps(output, ensure_ascii=False, indent=2)
    print(text)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")


if __name__ == "__main__":
    main()
