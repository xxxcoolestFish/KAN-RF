"""Decision receiver test using semantically regularized cognitive coefficients."""

from __future__ import annotations

import argparse
import json

from scripts.stage2_lowrank_semantic import train as train_semantic
from scripts.stage3_cognitive_receiver import evaluate, train_decision


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cognitive-steps", type=int, default=300)
    parser.add_argument("--decision-steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    cognitive_model = train_semantic(0.1, args.cognitive_steps, args.seed)
    cognitive_model.eval()
    results = []
    for mode in ("state_only", "oracle", "cognitive"):
        decision, transport = train_decision(
            mode, cognitive_model, args.decision_steps, args.seed
        )
        results.append({
            "mode": mode,
            "metrics": evaluate(mode, decision, transport, cognitive_model),
        })
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
