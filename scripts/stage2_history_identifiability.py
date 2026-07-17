"""Audit whether the observed history identifies the hidden context.

This is a diagnostic, not a model component.  If a classifier cannot recover
the finite training contexts from history, a decision teacher that directly
reads those contexts is not a valid end-to-end target for the cognitive model.
"""

from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F

from kanrf._protokan import ProtoKAN
from physics_transfer.multifactor_data import sample_multifactor_batch
from scripts.stage2_lowrank_gap_calibration import FACTORS


def labels_from_factors(factors):
    labels = []
    for value in factors.tolist():
        labels.append(min(range(len(FACTORS)), key=lambda index: sum((value[k] - FACTORS[index][k]) ** 2 for k in range(4))))
    return torch.tensor(labels, dtype=torch.long)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--steps", type=int, default=500); parser.add_argument("--seed", type=int, default=42); args = parser.parse_args()
    torch.manual_seed(args.seed)
    model = ProtoKAN([56, 48, len(FACTORS)], n_prototypes=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    for _ in range(args.steps):
        batch = sample_multifactor_batch(64, 8, FACTORS)
        loss = F.cross_entropy(model(batch["history"]), labels_from_factors(batch["factors"]))
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    test = sample_multifactor_batch(2048, 8, FACTORS)
    with torch.no_grad():
        prediction = model(test["history"]).argmax(dim=1)
    accuracy = (prediction == labels_from_factors(test["factors"])).float().mean().item()
    print(json.dumps({"accuracy": accuracy, "chance": 1.0 / len(FACTORS), "steps": args.steps, "seed": args.seed}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
