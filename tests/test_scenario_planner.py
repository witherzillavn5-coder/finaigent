"""Tests for the scenario planner calculator."""

from __future__ import annotations

import pytest

from tools.financial_calculators import scenario_planner
from tools.registry import execute_tool, get_tool_names, get_tool_schemas


def test_scenario_planner_returns_three_ordered_scenarios() -> None:
    result = scenario_planner("Buy a car worth 500 million VND", 50_000_000, 5_000_000, 6)

    assert result["target_amount"] == 500_000_000
    assert [row["annual_return_percent"] for row in result["scenarios"]] == [3, 5, 7]
    assert all(row["months_to_goal"] > 0 for row in result["scenarios"])
    assert result["scenarios"][0]["months_to_goal"] > result["scenarios"][2]["months_to_goal"]


def test_scenario_planner_zero_savings_and_contribution_is_unreachable() -> None:
    result = scenario_planner("Goal: 1 billion VND", 0, 0, 5)

    assert [row["months_to_goal"] for row in result["scenarios"]] == [None, None, None]


def test_scenario_planner_zero_contribution_can_grow_existing_savings() -> None:
    result = scenario_planner("Goal 110 million VND", 100_000_000, 0, 5)

    assert all(row["months_to_goal"] is not None for row in result["scenarios"])


def test_scenario_planner_returns_zero_months_when_goal_is_reached() -> None:
    result = scenario_planner("500 million VND car", 500_000_000, 0, 5)

    assert [row["months_to_goal"] for row in result["scenarios"]] == [0, 0, 0]


def test_scenario_planner_rejects_goal_without_numeric_target() -> None:
    with pytest.raises(ValueError, match="numeric target"):
        scenario_planner("Buy a car", 0, 1_000_000, 5)


def test_scenario_planner_is_registered_with_function_schema() -> None:
    assert "scenario_planner" in get_tool_names()
    response = execute_tool(
        "scenario_planner",
        {
            "goal": "500 million VND",
            "current_savings": 0,
            "monthly_contribution": 1_000_000,
            "expected_annual_return_rate": 5,
        },
    )
    assert len(response["result"]["scenarios"]) == 3
    assert any(schema["function"]["name"] == "scenario_planner" for schema in get_tool_schemas())
