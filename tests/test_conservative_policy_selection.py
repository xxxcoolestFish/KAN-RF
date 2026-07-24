import pytest

from cpbn.conservative_policy_selection import (
    paired_return_lower_bound,
)


def test_accepts_consistently_better_candidate():
    result = paired_return_lower_bound(
        [10.0, 11.0, 12.0],
        [12.0, 13.0, 14.0],
    )
    assert result["accepted"]
    assert result["lower_bound"] == pytest.approx(2.0)


def test_rejects_uncertain_or_harmful_candidate():
    result = paired_return_lower_bound(
        [10.0, 10.0, 10.0],
        [9.0, 12.0, 8.0],
    )
    assert not result["accepted"]
    assert result["lower_bound"] < 0.0


def test_requires_paired_one_dimensional_returns():
    with pytest.raises(ValueError):
        paired_return_lower_bound([1.0], [1.0, 2.0])
