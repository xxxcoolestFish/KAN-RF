"""Checks for matched multi-seed negative-transfer aggregation."""

from scripts.validate_target_negative_transfer_multiseed import (
    CONDITIONS,
    summarize_seed_results,
)


def _condition(successes, count=10):
    return {
        "final": {
            "success_count": successes,
            "evaluation_count": count,
            "success_rate": successes / count,
        },
    }


def test_multiseed_summary_preserves_seedwise_differences():
    per_seed = []
    for source, head, scratch in ((2, 5, 8), (3, 7, 9)):
        values = (source, head, scratch)
        per_seed.append({
            "conditions": {
                name: _condition(value)
                for name, value in zip(CONDITIONS, values)
            },
        })
    summary = summarize_seed_results(per_seed)
    assert summary["source"]["pooled_success_rate"] == 0.25
    assert summary["head_reset"]["training_seed_success_rates"] == [0.5, 0.7]
    deltas = summary["head_reset_minus_source"]["per_training_seed"]
    assert abs(deltas[0] - 0.3) < 1e-12
    assert abs(deltas[1] - 0.4) < 1e-12
    assert summary["scratch"]["pooled_success_count"] == 17
