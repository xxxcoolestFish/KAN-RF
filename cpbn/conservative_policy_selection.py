"""Closed-loop paired-return acceptance for conservative deployment."""

from __future__ import annotations

import numpy as np


def paired_return_lower_bound(
    base_returns,
    candidate_returns,
    *,
    confidence_multiplier: float = 1.0,
):
    base = np.asarray(base_returns, dtype=np.float64)
    candidate = np.asarray(candidate_returns, dtype=np.float64)
    if base.shape != candidate.shape or base.ndim != 1:
        raise ValueError(
            "Base and candidate returns must be matching 1D arrays.",
        )
    if base.size == 0:
        raise ValueError("At least one paired return is required.")
    differences = candidate - base
    standard_error = (
        differences.std(ddof=1) / np.sqrt(differences.size)
        if differences.size > 1
        else 0.0
    )
    mean = float(differences.mean())
    lower_bound = float(
        mean - confidence_multiplier * standard_error,
    )
    return {
        "differences": differences,
        "mean": mean,
        "standard_error": float(standard_error),
        "lower_bound": lower_bound,
        "accepted": lower_bound > 0.0,
    }
