"""Tests for the financial health score."""

from __future__ import annotations

import pytest

from tools.health_score import calculate_health_score


def test_health_score_uses_transparent_weighted_formula() -> None:
    result = calculate_health_score(10_000, 5_000, 2_000, 60_000)

    assert result["score"] == 70
    assert result["savings_points"] == 40
    assert result["debt_points"] == 20
    assert result["expense_points"] == 10
    assert result["weakest_component"] == "expenses"


def test_health_score_is_capped_at_100() -> None:
    result = calculate_health_score(10_000, 0, 3_000, 0)

    assert result["score"] == 100


def test_health_score_rejects_zero_income() -> None:
    with pytest.raises(ValueError, match="monthly_income"):
        calculate_health_score(0, 0, 0, 0)


@pytest.mark.parametrize(
    ("field_values", "field_name"),
    [
        ((10_000, -1, 0, 0), "monthly_expenses"),
        ((10_000, 0, -1, 0), "monthly_savings"),
        ((10_000, 0, 0, -1), "total_debt"),
    ],
)
def test_health_score_rejects_negative_values(
    field_values: tuple[int, int, int, int],
    field_name: str,
) -> None:
    with pytest.raises(ValueError, match=field_name):
        calculate_health_score(*field_values)
