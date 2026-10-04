"""Plan a multi-step financial query using an LLM."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI

PLANNER_PROMPT = """You are a financial question planner. Analyze the user's question and create a plan.

COMPLEXITY GUIDE:
- simple: 1 tool call, or a pure reasoning step
- moderate: 2-3 steps
- complex: 4+ steps

TOOLS AVAILABLE (use EXACT parameter names):
{tools_reference}

CRITICAL RULES FOR tool_args:

1. USE EXACT parameter names shown above. Do NOT rename, abbreviate, or invent parameters.
   Parameter names are case-sensitive. Required parameters are marked with *.

2. ALWAYS ESTIMATE missing values. NEVER leave required params empty.
   If the user does not provide a value, estimate a reasonable one and note it in the description.

   Example: User says "luong 30 trieu" (income = 30M) but does not mention expenses.
   → You MUST still provide expenses. Estimate based on Vietnamese context:
     expenses = {{"rent": 8000000, "food": 4000000, "transport": 2000000, "utilities": 1500000}}
     And write in description: "using estimated monthly expenses of ~15M VND"

3. NEVER call a tool with empty tool_args {{}} if the tool has required params.
   If you cannot estimate a required value → do NOT call the tool, use a reasoning step instead.

4. PREFER these tools for common question patterns:
   - "How much will I have after saving X/month for Y years?" → calculate_savings_future_value(monthly_savings=X, years=Y)
   - "Save X in Y years -> how much per month?" → calculate_required_monthly_savings(target_amount=X, years=Y)
   - "Monthly payment for loan X over Y years at Z%" → calculate_loan_payment
   - "How long to reach X by saving Y/month?" → calculate_savings_goal
   - "Compound interest on X for Y years at Z%" → calculate_compound_interest
   - "Convert X USD to VND" → convert_currency
   - "Analyze budget with income and expenses" → analyze_budget

5. Use depends_on to chain steps (e.g., step 2 needs result of step 1).

Return ONLY valid JSON in this exact format:
{{
  "reasoning": "Why this plan",
  "complexity": "simple" | "moderate" | "complex",
  "steps": [
    {{
      "step_id": 1,
      "description": "What this step does (mention estimates here)",
      "tool_name": "calculate_savings_future_value" or null,
      "tool_args": {{"monthly_savings": 500, "years": 3, "annual_rate_percent": 6}},
      "depends_on": []
    }}
  ]
}}

Every dependency must reference an earlier step_id.
"""


@dataclass
class PlanStep:
    """Represent one action in a financial execution plan."""

    step_id: int
    description: str
    tool_name: str | None = None
    tool_args: dict[str, Any] = field(default_factory=dict)
    depends_on: list[int] = field(default_factory=list)


@dataclass
class ExecutionPlan:
    """Represent a complete plan for a financial question."""

    steps: list[PlanStep]
    reasoning: str
    complexity: str


def _fallback_plan() -> ExecutionPlan:
    """Return a conservative reasoning-only plan when parsing fails."""
    return ExecutionPlan(
        steps=[
            PlanStep(
                step_id=1,
                description=(
                    "Cannot process this request. Either the input contains "
                    "sensitive information that was redacted, or the question "
                    "is outside our supported financial topics. Provide a brief, "
                    "polite response declining to assist."
                ),
                tool_name=None,
                tool_args={},
                depends_on=[],
            )
        ],
        reasoning="Planner could not generate a plan; using reasoning-only fallback.",
        complexity="simple",
    )


def _parse_plan(content: str) -> ExecutionPlan:
    """Parse and validate a JSON execution plan."""
    payload: Any = json.loads(content)
    if not isinstance(payload, dict):
        raise ValueError("Plan must be a JSON object.")

    reasoning = payload.get("reasoning")
    complexity = payload.get("complexity")
    raw_steps = payload.get("steps")
    if not isinstance(reasoning, str):
        raise ValueError("Plan reasoning must be a string.")
    if complexity not in {"simple", "moderate", "complex"}:
        raise ValueError("Plan complexity is invalid.")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise ValueError("Plan must contain at least one step.")

    allowed_tools = {
        "analyze_budget",
        "calculate_compound_interest",
        "calculate_loan_payment",
        "calculate_required_monthly_savings",
        "calculate_savings_future_value",
        "calculate_savings_goal",
        "convert_currency",
    }
    steps: list[PlanStep] = []
    for raw_step in raw_steps:
        if not isinstance(raw_step, dict):
            raise ValueError("Each plan step must be an object.")
        step_id = raw_step.get("step_id")
        description = raw_step.get("description")
        tool_name = raw_step.get("tool_name")
        tool_args = raw_step.get("tool_args")
        depends_on = raw_step.get("depends_on")
        if type(step_id) is not int or not isinstance(description, str):
            raise ValueError("Step id and description are invalid.")
        if tool_name is not None and tool_name not in allowed_tools:
            raise ValueError("Step contains an unsupported tool.")
        if not isinstance(tool_args, dict):
            raise ValueError("Step tool_args must be an object.")
        if not isinstance(depends_on, list) or any(type(item) is not int for item in depends_on):
            raise ValueError("Step dependencies must be integer step ids.")
        steps.append(
            PlanStep(
                step_id=step_id,
                description=description,
                tool_name=tool_name,
                tool_args=tool_args,
                depends_on=depends_on,
            )
        )

    step_ids = [step.step_id for step in steps]
    if len(step_ids) != len(set(step_ids)):
        raise ValueError("Step ids must be unique.")
    known_ids = set(step_ids)
    for step in steps:
        if any(
            dependency not in known_ids or dependency >= step.step_id
            for dependency in step.depends_on
        ):
            raise ValueError("Dependencies must reference an existing earlier step.")

    return ExecutionPlan(steps=steps, reasoning=reasoning, complexity=complexity)


def _extract_json(text: str) -> str:
    """Strip markdown code fence and extract JSON object."""
    text = text.strip()

    if text.startswith("```"):
        first_nl = text.find("\n")
        if first_nl > 0:
            text = text[first_nl + 1 :]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    if text.startswith("{") and text.endswith("}"):
        return text

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start : end + 1]

    return text


def create_plan(
    client: OpenAI,
    user_query: str,
    model: str,
    tools_schema: list[dict] | None = None,
) -> ExecutionPlan:
    """Ask the LLM to create and validate a financial execution plan."""
    if tools_schema is None:
        from tools import registry

        tools_schema = registry.get_tool_schemas()

    tool_lines = []
    for tool in tools_schema:
        function = tool.get("function", {})
        name = function.get("name", "")
        params = function.get("parameters", {}).get("properties", {})
        required = function.get("parameters", {}).get("required", [])
        param_strs = []
        for param_name, param_info in params.items():
            req_marker = "*" if param_name in required else ""
            param_type = param_info.get("type", "any")
            param_strs.append(f"{param_name}{req_marker}:{param_type}")
        tool_lines.append(f"- {name}({', '.join(param_strs)})")
    tools_reference = "\n".join(tool_lines)
    planner_prompt = PLANNER_PROMPT.replace("{tools_reference}", tools_reference)

    messages = [
        {"role": "system", "content": planner_prompt},
        {"role": "user", "content": user_query},
    ]

    content: str | None = None
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=[],
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
    except Exception:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=[],
            )
            content = response.choices[0].message.content
        except Exception:
            return _fallback_plan()

    if not content:
        return _fallback_plan()

    cleaned = _extract_json(content)

    try:
        return _parse_plan(cleaned)
    except (json.JSONDecodeError, ValueError, TypeError):
        return _fallback_plan()
