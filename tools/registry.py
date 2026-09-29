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
    calculate_savings_goal,
    convert_currency,
)

ToolFunction = Callable[..., dict[str, Any]]

TOOL_FUNCTIONS: dict[str, ToolFunction] = {
    "calculate_compound_interest": calculate_compound_interest,
    "calculate_loan_payment": calculate_loan_payment,
    "calculate_savings_goal": calculate_savings_goal,
    "convert_currency": convert_currency,
    "analyze_budget": analyze_budget,
}


def get_tool_schemas() -> list[dict[str, Any]]:
    """Return OpenAI function-calling schemas for all registered tools."""
    return [
        {
            "type": "function",
            "function": {
                "name": "calculate_compound_interest",
                "description": "Calculate compound growth and interest earned on a VND principal.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "principal": {
                            "type": "number",
                            "description": "Principal amount in VND",
                        },
                        "annual_rate_percent": {
                            "type": "number",
                            "description": "Annual interest rate as percentage (e.g., 8 for 8%)",
                        },
                        "years": {
                            "type": "number",
                            "description": "Investment duration in years",
                        },
                        "compounds_per_year": {
                            "type": "integer",
                            "description": "Number of compounding periods per year",
                            "default": 12,
                        },
                    },
                    "required": ["principal", "annual_rate_percent", "years"],
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
        result = function(**arguments)
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
