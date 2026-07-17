"""Conditioning probe: scale the cognitive coefficients before reception."""

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


def get_code(batch, cognitive, scale):
    with torch.no_grad():
        return scale * cognitive(batch["history"], batch["state"], batch["action"])["physics_coefficients"]


def train_decision(cognitive, scale, steps, seed):
    torch.manual_seed(seed + 9000)
    model = SeparatedPhysicsDecision(6, 1, 4, hidden_dim=24, n_prototypes=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    for index in range(steps):
        batch = sample_multifactor_batch(64, 8, FACTORS)
        target = conditional_teacher(batch["state"], batch["factors"])
        loss = F.mse_loss(model(batch["state"], get_code(batch, cognitive, scale))["action"], target)
        factor_a, factor_b = FACTORS[index % len(FACTORS)], FACTORS[(index + 1) % len(FACTORS)]
        pair = _pair(factor_a, factor_b, batch_size=64)
        fa, fb = torch.tensor(factor_a).float().repeat(64, 1), torch.tensor(factor_b).float().repeat(64, 1)
        pa = {"history": pair["history_a"], "state": pair["state"], "action": pair["action"], "factors": fa}
        pb = {"history": pair["history_b"], "state": pair["state"], "action": pair["action"], "factors": fb}
        pred_a = model(pair["state"], get_code(pa, cognitive, scale))["action"]
        pred_b = model(pair["state"], get_code(pb, cognitive, scale))["action"]
        ta, tb = conditional_teacher(pair["state"], fa), conditional_teacher(pair["state"], fb)
        loss = loss + F.mse_loss(pred_a, ta) + F.mse_loss(pred_b, tb) + 10.0 * F.mse_loss(pred_a - pred_b, ta - tb)
        loss = loss + 0.05 * model.separation_loss() + 0.05 * model.physics_basis_orthogonality_loss()
        optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
    return model


def evaluate(model, cognitive, scale):
    model.eval()
    with torch.no_grad():
        batch = sample_multifactor_batch(1024, 8, HELDOUT)
        code = get_code(batch, cognitive, scale)
        target = conditional_teacher(batch["state"], batch["factors"])
        pred = model(batch["state"], code)["action"]
        shuffled = model(batch["state"], code[torch.randperm(1024)])["action"]
        pair = _pair(HELDOUT[0], HELDOUT[1], batch_size=1024)
        fa, fb = torch.tensor(HELDOUT[0]).float().repeat(1024, 1), torch.tensor(HELDOUT[1]).float().repeat(1024, 1)
        pa = {"history": pair["history_a"], "state": pair["state"], "action": pair["action"], "factors": fa}
        pb = {"history": pair["history_b"], "state": pair["state"], "action": pair["action"], "factors": fb}
        pred_a = model(pair["state"], get_code(pa, cognitive, scale))["action"]
        pred_b = model(pair["state"], get_code(pb, cognitive, scale))["action"]
        ta, tb = conditional_teacher(pair["state"], fa), conditional_teacher(pair["state"], fb)
        gap, true = (pred_a - pred_b).abs().mean().item(), (ta - tb).abs().mean().item()
        return {"mse": F.mse_loss(pred, target).item(), "shuffle_delta": F.mse_loss(shuffled, target).item() - F.mse_loss(pred, target).item(), "action_gap_ratio": gap / max(true, 1e-8)}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--cognitive-steps", type=int, default=300); parser.add_argument("--decision-steps", type=int, default=300); parser.add_argument("--seed", type=int, default=42); args = parser.parse_args()
    cognitive = train_cognitive(0.1, args.cognitive_steps, args.seed); cognitive.eval()
    results = []
    for scale in (1.0, 5.0, 10.0, 20.0):
        results.append({"scale": scale, "metrics": evaluate(train_decision(cognitive, scale, args.decision_steps, args.seed), cognitive, scale)})
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
