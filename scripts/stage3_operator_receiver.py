"""Decision receiver probe for an operator-valued cognitive interface.

Instead of exposing an arbitrary coefficient coordinate, the decision network
receives the cognitive model's queried physical residual at the current
state/action.  The task branch remains state-only, so counterfactual action
changes must pass through the operator response.
"""

from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F

from physics_transfer.multifactor_data import sample_multifactor_batch
from physics_transfer.separated_decision import SeparatedPhysicsDecision
from scripts.stage2_lowrank_gap_calibration import FACTORS, HELDOUT, _pair
from scripts.stage2_lowrank_semantic import train as train_cognitive
from scripts.stage3_oracle_receiver_probe import conditional_teacher


TRAIN_FACTORS = FACTORS


def get_code(mode, batch, cognitive):
    if mode == "state_only":
        return torch.zeros(batch["state"].shape[0], 6)
    with torch.no_grad():
        out = cognitive(batch["history"], batch["state"], batch["action"])
        if mode == "operator_residual":
            return out["physics_residual"]
        if mode == "operator_next_delta":
            return out["next_state"] - batch["state"]
    raise ValueError(mode)


def train_decision(mode, cognitive, steps, seed):
    torch.manual_seed(seed + 9000)
    model = SeparatedPhysicsDecision(6, 1, 6, hidden_dim=24, n_prototypes=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    for index in range(steps):
        batch = sample_multifactor_batch(64, 8, TRAIN_FACTORS)
        code = get_code(mode, batch, cognitive)
        target = conditional_teacher(batch["state"], batch["factors"])
        output = model(batch["state"], code)
        loss = F.mse_loss(output["action"], target)
        factor_a, factor_b = TRAIN_FACTORS[index % len(TRAIN_FACTORS)], TRAIN_FACTORS[(index + 1) % len(TRAIN_FACTORS)]
        pair = _pair(factor_a, factor_b, batch_size=64)
        fa = torch.tensor(factor_a).float().repeat(64, 1)
        fb = torch.tensor(factor_b).float().repeat(64, 1)
        pair_a = {"history": pair["history_a"], "state": pair["state"], "action": pair["action"], "factors": fa}
        pair_b = {"history": pair["history_b"], "state": pair["state"], "action": pair["action"], "factors": fb}
        code_a, code_b = get_code(mode, pair_a, cognitive), get_code(mode, pair_b, cognitive)
        target_a, target_b = conditional_teacher(pair["state"], fa), conditional_teacher(pair["state"], fb)
        pred_a = model(pair["state"], code_a)["action"]
        pred_b = model(pair["state"], code_b)["action"]
        loss = loss + F.mse_loss(pred_a, target_a) + F.mse_loss(pred_b, target_b)
        loss = loss + 10.0 * F.mse_loss(pred_a - pred_b, target_a - target_b)
        loss = loss + 0.05 * model.separation_loss() + 0.05 * model.physics_basis_orthogonality_loss()
        optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
    return model


def evaluate(mode, model, cognitive):
    model.eval()
    with torch.no_grad():
        batch = sample_multifactor_batch(1024, 8, HELDOUT)
        code = get_code(mode, batch, cognitive)
        target = conditional_teacher(batch["state"], batch["factors"])
        prediction = model(batch["state"], code)["action"]
        shuffled = model(batch["state"], code[torch.randperm(1024)])["action"]
        pair = _pair(HELDOUT[0], HELDOUT[1], batch_size=1024)
        fa = torch.tensor(HELDOUT[0]).float().repeat(1024, 1)
        fb = torch.tensor(HELDOUT[1]).float().repeat(1024, 1)
        pa = {"history": pair["history_a"], "state": pair["state"], "action": pair["action"], "factors": fa}
        pb = {"history": pair["history_b"], "state": pair["state"], "action": pair["action"], "factors": fb}
        pred_a = model(pair["state"], get_code(mode, pa, cognitive))["action"]
        pred_b = model(pair["state"], get_code(mode, pb, cognitive))["action"]
        ta, tb = conditional_teacher(pair["state"], fa), conditional_teacher(pair["state"], fb)
        predicted_gap = (pred_a - pred_b).abs().mean().item()
        true_gap = (ta - tb).abs().mean().item()
        return {"mse": F.mse_loss(prediction, target).item(), "shuffled_mse": F.mse_loss(shuffled, target).item(), "shuffle_delta": F.mse_loss(shuffled, target).item() - F.mse_loss(prediction, target).item(), "predicted_action_gap": predicted_gap, "true_action_gap": true_gap, "action_gap_ratio": predicted_gap / max(true_gap, 1e-8)}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--cognitive-steps", type=int, default=300); parser.add_argument("--decision-steps", type=int, default=300); parser.add_argument("--seed", type=int, default=42); args = parser.parse_args()
    cognitive = train_cognitive(0.1, args.cognitive_steps, args.seed); cognitive.eval()
    results = []
    for mode in ("state_only", "operator_residual", "operator_next_delta"):
        model = train_decision(mode, cognitive, args.decision_steps, args.seed)
        results.append({"mode": mode, "metrics": evaluate(mode, model, cognitive)})
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
