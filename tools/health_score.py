"""Transparent financial health score calculation."""

from __future__ import annotations

import math


def calculate_health_score(
    monthly_income: float,
    monthly_expenses: float,
    monthly_savings: float,
    total_debt: float,
) -> dict[str, float | str]:
    """Return a 0-100 score and normalized component scores.

    Savings rate is capped at 20% for full points. Debt equal to annual income
    receives zero debt points. Expenses equal to income receive zero expense
    points.
    """
    values = {
        "monthly_income": monthly_income,
        "monthly_expenses": monthly_expenses,
        "monthly_savings": monthly_savings,
        "total_debt": total_debt,
    }
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"{name} must be a finite number.")
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be a non-negative finite number.")
    if monthly_income == 0:
        raise ValueError("monthly_income must be greater than zero.")

    savings_rate = monthly_savings / monthly_income
    debt_ratio = total_debt / (monthly_income * 12)
    expense_ratio = monthly_expenses / monthly_income

    savings_points = min(savings_rate / 0.20, 1.0) * 40
    debt_points = max(1.0 - debt_ratio, 0.0) * 40
    expense_points = max(1.0 - expense_ratio, 0.0) * 20
    component_scores = {
        "savings": savings_points,
        "debt": debt_points,
        "expenses": expense_points,
    }
    weakest_component = min(component_scores, key=component_scores.get)

    return {
        "score": round(sum(component_scores.values())),
        "savings_rate": savings_rate,
        "debt_ratio": debt_ratio,
        "expense_ratio": expense_ratio,
        "savings_points": savings_points,
        "debt_points": debt_points,
        "expense_points": expense_points,
        "weakest_component": weakest_component,
    }
