"""Register and dispatch financial calculator tools."""

import json
import sys
from pathlib import Path
from typing import Any, Callable  # noqa: UP035

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.financial_calculators import (
    analyze_budget,
    calculate_compound_interest,
    calculate_loan_payment,
    calculate_required_monthly_savings,
    calculate_savings_future_value,
    calculate_savings_goal,
    convert_crypto,
    convert_currency,
    scenario_planner,
)

ToolFunction = Callable[..., dict[str, Any]]

TOOL_FUNCTIONS = {
    "analyze_budget": analyze_budget,
    "calculate_compound_interest": calculate_compound_interest,
    "calculate_loan_payment": calculate_loan_payment,
    "calculate_required_monthly_savings": calculate_required_monthly_savings,
    "calculate_savings_future_value": calculate_savings_future_value,
    "calculate_savings_goal": calculate_savings_goal,
    "convert_crypto": convert_crypto,
    "convert_currency": convert_currency,
    "scenario_planner": scenario_planner,
}


def get_tool_schemas() -> list[dict[str, Any]]:
    """Return OpenAI function-calling schemas for all registered tools."""
    return [
        {
            "type": "function",
            "function": {
                "name": "calculate_savings_future_value",
                "description": (
                    "Calculate future value when saving a fixed amount per month "
                    "for N years at a given annual interest rate. Use this for "
                    "'how much will I have after saving X/month for Y years?'"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "monthly_savings": {
                            "type": "number",
                            "description": "Amount saved per month",
                        },
                        "years": {
                            "type": "number",
                            "description": "Number of years",
                        },
                        "annual_rate_percent": {
                            "type": "number",
                            "description": "Annual interest rate as percent (e.g., 6 for 6%). Defaults to 0.",
                        },
                    },
                    "required": ["monthly_savings", "years"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "calculate_loan_payment",
                "description": "Calculate fixed monthly payments and total interest for a loan.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "principal": {
                            "type": "number",
                            "description": "Loan amount in VND",
                        },
                        "annual_rate_percent": {
                            "type": "number",
                            "description": "Annual interest rate as percentage",
                        },
                        "years": {
                            "type": "number",
                            "description": "Loan term in years",
                        },
                    },
                    "required": ["principal", "annual_rate_percent", "years"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "calculate_required_monthly_savings",
                "description": (
                    "Calculate the monthly savings amount required to reach a "
                    "target amount within N years at a given annual interest rate. "
                    "Use this when user asks 'how much do I need to save per month "
                    "to reach X in Y years?'."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target_amount": {
                            "type": "number",
                            "description": "Target savings amount in VND",
                        },
                        "years": {
                            "type": "number",
                            "description": "Number of years to reach the target",
                        },
                        "annual_rate_percent": {
                            "type": "number",
                            "description": (
                                "Expected annual interest rate as percentage "
                                "(e.g., 5 for 5%). Defaults to 0."
                            ),
                        },
                    },
                    "required": ["target_amount", "years"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "calculate_savings_goal",
                "description": "Estimate how many months are needed to reach a savings target.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target_amount": {
                            "type": "number",
                            "description": "Target savings amount in VND",
                        },
                        "monthly_savings": {
                            "type": "number",
                            "description": "Amount saved per month in VND",
                        },
                        "annual_rate_percent": {
                            "type": "number",
                            "description": "Expected annual return rate",
                            "default": 0,
                        },
                    },
                    "required": ["target_amount", "monthly_savings"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "convert_currency",
                "description": "Convert an amount using the calculator's supported static exchange rates.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "amount": {"type": "number", "description": "Amount to convert"},
                        "from_currency": {
                            "type": "string",
                            "description": "Source currency code (USD, EUR, VND)",
                        },
                        "to_currency": {
                            "type": "string",
                            "description": "Target currency code (USD, EUR, VND)",
                        },
                    },
                    "required": ["amount", "from_currency", "to_currency"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "convert_crypto",
                "description": (
                    "Convert a crypto amount (BTC, ETH, USDT, BNB, SOL) to USD or VND. "
                    "Uses STATIC reference prices (not real-time). "
                    "Use for questions like 'how much is 1 BTC in VND?'."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "amount": {
                            "type": "number",
                            "description": "Amount of crypto to convert",
                        },
                        "from_coin": {
                            "type": "string",
                            "description": "Source coin: BTC, ETH, USDT, BNB, SOL",
                        },
                        "to_currency": {
                            "type": "string",
                            "description": "Target currency: USD or VND. Defaults to USD.",
                        },
                    },
                    "required": ["amount", "from_coin"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "scenario_planner",
                "description": (
                    "Estimate months to reach a numeric savings target mentioned "
                    "in a goal such as 'buy a car worth 500 million VND'. Returns "
                    "conservative 3%, moderate 5%, and aggressive 7% scenarios."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "goal": {
                            "type": "string",
                            "description": "Goal description containing a numeric target and optional million/billion unit",
                        },
                        "current_savings": {
                            "type": "number",
                            "description": "Current savings in VND",
                        },
                        "monthly_contribution": {
                            "type": "number",
                            "description": "Monthly contribution in VND",
                        },
                        "expected_annual_return_rate": {
                            "type": "number",
                            "description": "User's expected annual return rate as a percentage",
                        },
                    },
                    "required": [
                        "goal",
                        "current_savings",
                        "monthly_contribution",
                        "expected_annual_return_rate",
                    ],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "analyze_budget",
                "description": "Analyze monthly expenses against the 50/30/20 budgeting guideline.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "monthly_income": {
                            "type": "number",
                            "description": "Monthly income in VND",
                        },
                        "expenses": {
                            "type": "object",
                            "description": (
                                "Dictionary mapping expense category to amount in VND. "
                                "Example: {'rent': 5000000, 'food': 3000000}"
                            ),
                            "additionalProperties": {"type": "number"},
                        },
                    },
                    "required": ["monthly_income", "expenses"],
                },
            },
        },
    ]


def execute_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Run a registered tool and return its result or an error."""
    function = TOOL_FUNCTIONS.get(name)
    if function is None:
        return {"error": f"Unknown tool: {name}"}

    try:
        result = function(**arguments)  # type: ignore[operator]
    except ValueError as error:
        return {"error": str(error)}
    except Exception as error:
        return {"error": f"Tool execution failed: {error}"}
    return {"result": result}


def get_tool_names() -> list[str]:
    """Return the names of all registered tools."""
    return list(TOOL_FUNCTIONS)


if __name__ == "__main__":
    print("Available tools:", get_tool_names())
    example_result = execute_tool(
        "calculate_compound_interest",
        {"principal": 10_000_000, "annual_rate_percent": 8, "years": 2},
    )
    print(json.dumps(example_result, indent=2, ensure_ascii=False))
