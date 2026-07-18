"""Stage 15 launcher with controlled action exploration for outcome replay."""

from __future__ import annotations

import torch
import torch.nn.functional as F

import scripts.stage15_real_outcome_replay as base
from physics_transfer.multifactor_data import _random_states
from physics_transfer.variants import step
from scripts.stage13_online_task_loss_adaptation import tip_height
from scripts.stage7_single_env_decision_adaptation import operator_query


EXPLORATION_NOISE = 0.10


def run_episode_explore(cognitive, decision, cognitive_optimizer,
                        decision_optimizer, replay, references, factor, args, seed):
    torch.manual_seed(seed)
    state = _random_states(1)
    latent = cognitive.initial_latent(1)
    factor_tensor = torch.tensor(factor).repeat(1, 1)
    trajectory, heights, prediction_errors = [], [], []
    for _ in range(args.rollout_steps):
        with torch.no_grad():
            operator = operator_query(cognitive, state, latent)
        policy_action = decision(state, operator)["action"]
        executed_action = (
            policy_action + EXPLORATION_NOISE * torch.randn_like(policy_action)
        ).clamp(-1.0, 1.0)
        next_state = step(
            state, executed_action.detach(), factor_tensor[:, 0], factor_tensor[:, 1],
            factor_tensor[:, 2], factor_tensor[:, 3],
        )
        current_reward = base.reward(next_state, executed_action.detach()).item()
        prediction = cognitive.predict_next(state, executed_action.detach(), latent)
        cognitive_loss = F.smooth_l1_loss(prediction, next_state)
        cognitive_optimizer.zero_grad(); cognitive_loss.backward()
        torch.nn.utils.clip_grad_norm_(cognitive.parameters(), 5.0)
        cognitive_optimizer.step()
        with torch.no_grad():
            next_latent = cognitive.observe_transition(
                state, executed_action.detach(), next_state, latent
            ).detach()
            trajectory.append({
                "state": state,
                "operator": operator,
                "latent": latent,
                "action": executed_action.detach(),
                "reward": current_reward,
            })
            heights.append(tip_height(next_state).item())
            prediction_errors.append(F.mse_loss(prediction, next_state).item())
            state, latent = next_state.detach(), next_latent

    replay.add_episode(trajectory, args.gamma)
    losses, weights = base.update_from_outcomes(
        cognitive, decision, decision_optimizer, replay, references, args
    )
    return {
        "success": max(heights) >= 1.0,
        "max_height": max(heights),
        "mean_return": sum(item["reward"] for item in trajectory) / len(trajectory),
        "mean_prediction_mse": sum(prediction_errors) / len(prediction_errors),
        "real_loss": sum(item["real_loss"] for item in losses) / max(len(losses), 1),
        "model_loss": sum(item["model_loss"] for item in losses) / max(len(losses), 1),
        "residual_norm": losses[-1]["residual_norm"] if losses else 0.0,
        "mean_weight": sum(weights) / max(len(weights), 1),
    }


base.run_episode = run_episode_explore


if __name__ == "__main__":
    base.main()
