"""Summarize the fixed-hyperparameter Hopper physics-shift prescreen."""

from __future__ import annotations

import json
from pathlib import Path


GRID_FILES = {
    "payload_125": "hopper_cognitive_recovery_grid_payload125_seed1811.json",
    "payload_150": (
        "hopper_cognitive_recovery_grid_payload150_quick_seed1811.json"
    ),
    "friction_070": (
        "hopper_cognitive_recovery_grid_friction070_quick_seed1811.json"
    ),
    "actuator_080": (
        "hopper_cognitive_recovery_grid_actuator080_quick_seed1811.json"
    ),
    "actuator_065": (
        "hopper_cognitive_recovery_grid_actuator065_quick_seed1811.json"
    ),
    "combo_mild": (
        "hopper_cognitive_recovery_grid_combo_mild_quick_seed1811.json"
    ),
}


def main():
    results = Path("results")
    summary = {}
    for target, filename in GRID_FILES.items():
        payload = json.loads(
            (results / filename).read_text(encoding="utf-8"),
        )
        values = {
            budget: float(payload["results"][budget]["mean_return"])
            for budget in ("0", "512", "2048")
        }
        summary[target] = {
            "returns": values,
            "delta_at_512": values["512"] - values["0"],
            "delta_at_2048": values["2048"] - values["0"],
            "relative_at_512": values["512"] / values["0"],
            "relative_at_2048": values["2048"] / values["0"],
            "monotonic": payload["monotonic_recovery"],
            "evaluation_episodes": int(
                payload["config"]["evaluation_episodes"],
            ),
            "source_file": filename,
        }

    medium = {}
    baseline_payload = json.loads(
        (
            results
            / "hopper_distilled_policy_recovery_n512_combo_medium_seed1811.json"
        ).read_text(encoding="utf-8"),
    )
    medium["0"] = float(
        baseline_payload["methods"]["source"]["mean_return"],
    )
    for budget in (512, 2048):
        payload = json.loads(
            (
                results
                / (
                    f"hopper_distilled_policy_recovery_n{budget}_"
                    "combo_medium_seed1811.json"
                )
            ).read_text(encoding="utf-8"),
        )
        medium[str(budget)] = float(
            payload["methods"]["ungated"]["mean_return"],
        )
    summary["combo_medium"] = {
        "returns": medium,
        "delta_at_512": medium["512"] - medium["0"],
        "delta_at_2048": medium["2048"] - medium["0"],
        "relative_at_512": medium["512"] / medium["0"],
        "relative_at_2048": medium["2048"] / medium["0"],
        "monotonic": (
            medium["0"] <= medium["512"] <= medium["2048"]
        ),
        "evaluation_episodes": 3,
        "source_file": "fixed-protocol recovery files",
    }
    positive_512 = [
        target for target, value in summary.items()
        if value["delta_at_512"] > 0.0
    ]
    positive_2048 = [
        target for target, value in summary.items()
        if value["delta_at_2048"] > 0.0
    ]
    output = {
        "experiment": "HopperCognitiveRecoveryPhysicsGridSummary",
        "seed": 1811,
        "fixed_hyperparameters_across_targets": True,
        "screening_protocol": {
            "budgets": [0, 512, 2048],
            "source_actor_frozen": True,
            "target_reward_used_for_policy_update": False,
            "target_physical_parameters_visible": False,
        },
        "targets": summary,
        "positive_targets_at_512": positive_512,
        "positive_targets_at_2048": positive_2048,
        "positive_fraction_at_512": len(positive_512) / len(summary),
        "positive_fraction_at_2048": len(positive_2048) / len(summary),
        "recommended_formal_targets": [
            "actuator_080",
            "friction_070",
            "combo_medium",
            "payload_125",
        ],
    }
    output_path = (
        results / "hopper_cognitive_recovery_physics_grid_summary.json"
    )
    output_path.write_text(
        json.dumps(output, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
