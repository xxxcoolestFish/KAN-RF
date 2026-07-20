"""Direct PPO held-out runner without importing executable experiment modules."""

from __future__ import annotations

import torch

from physics_transfer.variants import step
from scripts import stage45_direct_ppo_heldout as base
from scripts.stage41_ppo_cognitive_actor import tip_height


@torch.no_grad()
def evaluate_states(actor, factor, goal, states, steps):
    state = states.clone()
    factor_tensor = torch.tensor(factor, dtype=state.dtype).view(1, 4).expand(state.shape[0], -1)
    maximum = torch.full((state.shape[0],), -float("inf"))
    success = torch.zeros(state.shape[0], dtype=torch.bool)
    for _ in range(steps):
        action, _, _ = actor.sample(state, goal.expand(state.shape[0], -1), deterministic=True)
        state = step(
            state, action, factor_tensor[:, 0], factor_tensor[:, 1],
            factor_tensor[:, 2], factor_tensor[:, 3],
        )
        height = tip_height(state)
        maximum = torch.maximum(maximum, height)
        success |= height >= 1.0
    return {
        "success_count": int(success.sum()),
        "success_rate": float(success.float().mean()),
        "mean_max_height": float(maximum.mean()),
        "max_height": float(maximum.max()),
    }


base.evaluate = evaluate_states
base.main()
