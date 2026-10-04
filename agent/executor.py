"""Execute an ExecutionPlan: run tools in order, handle dependencies."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from agent.planner import ExecutionPlan, PlanStep
from tools import registry

# Map sai tên tham số → tên đúng cho từng tool
PARAM_ALIASES: dict[str, dict[str, str]] = {
    "calculate_compound_interest": {
        "rate": "annual_rate_percent",
        "interest_rate": "annual_rate_percent",
        "annual_interest_rate": "annual_rate_percent",
        "duration_years": "years",
        "time_years": "years",
        "compounds": "compounds_per_year",
    },
    "calculate_loan_payment": {
        "loan_amount": "principal",
        "amount": "principal",
        "rate": "annual_rate_percent",
        "interest_rate": "annual_rate_percent",
        "annual_interest_rate": "annual_rate_percent",
        "duration_years": "years",
        "term_years": "years",
    },
    "calculate_savings_goal": {
        "target": "target_amount",
        "goal_amount": "target_amount",
        "monthly": "monthly_savings",
        "monthly_amount": "monthly_savings",
        "monthly_contribution": "monthly_savings",
        "rate": "annual_rate_percent",
        "interest_rate": "annual_rate_percent",
        "annual_interest_rate": "annual_rate_percent",
    },
    "convert_currency": {
        "value": "amount",
        "from": "from_currency",
        "to": "to_currency",
    },
    "analyze_budget": {
        "income": "monthly_income",
        "expense": "expenses",
        "expense_list": "expenses",
    },
}


@dataclass
class StepResult:
    """Result of executing one plan step."""

    step_id: int
    description: str
    tool_name: str | None
    tool_args: dict[str, Any]
    tool_output: dict[str, Any] | None
    error: str | None
    success: bool


@dataclass
class ExecutionResult:
    """Result of executing an entire plan."""

    plan: ExecutionPlan
    step_results: list[StepResult] = field(default_factory=list)
    all_succeeded: bool = False
    total_steps: int = 0
    failed_steps: int = 0


def _normalize_args(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Đổi tên tham số sai thành đúng + DROP tham số không hợp lệ."""
    func: Callable[..., Any] | None = registry.TOOL_FUNCTIONS.get(tool_name)
    if func is None:
        return args

    # 1. Áp dụng alias mapping
    aliases = PARAM_ALIASES.get(tool_name, {})
    renamed: dict[str, Any] = {}
    for key, value in args.items():
        new_key = aliases.get(key, key)
        renamed[new_key] = value

    # 2. Filter theo signature thật của tool
    try:
        sig = inspect.signature(func)
        valid_params = set(sig.parameters.keys())
        # Lấy cả positional-or-keyword lẫn keyword-only
        filtered = {k: v for k, v in renamed.items() if k in valid_params}
        return filtered
    except (ValueError, TypeError):
        # Nếu không inspect được (vd: builtin), trả về renamed
        return renamed


def _resolve_dependencies(
    step: PlanStep,
    completed: dict[int, StepResult],
) -> dict[str, Any]:
    """Merge tool_output của các step phụ thuộc vào args của step hiện tại."""
    args = dict(step.tool_args)

    if step.tool_name is None:
        return args

    for dep_id in step.depends_on:
        if dep_id not in completed:
            raise ValueError(f"Step {step.step_id} depends on step {dep_id} which is not completed")
        dep = completed[dep_id]
        if dep.tool_output:
            for key, value in dep.tool_output.items():
                if key not in args:
                    args[key] = value

    return args


def execute_plan(plan: ExecutionPlan) -> ExecutionResult:
    """Execute all steps in a plan in order."""
    result = ExecutionResult(plan=plan)
    completed: dict[int, StepResult] = {}

    sorted_steps = sorted(plan.steps, key=lambda s: s.step_id)

    for step in sorted_steps:
        step_result = _execute_step(step, completed)
        result.step_results.append(step_result)
        completed[step.step_id] = step_result

    result.total_steps = len(result.step_results)
    result.failed_steps = sum(1 for s in result.step_results if not s.success)
    result.all_succeeded = result.failed_steps == 0

    return result


def _execute_step(
    step: PlanStep,
    completed: dict[int, StepResult],
) -> StepResult:
    """Execute a single step."""
    if step.tool_name is None:
        return StepResult(
            step_id=step.step_id,
            description=step.description,
            tool_name=None,
            tool_args=step.tool_args,
            tool_output=None,
            error=None,
            success=True,
        )

    try:
        resolved_args = _resolve_dependencies(step, completed)
        resolved_args = _normalize_args(step.tool_name, resolved_args)

        raw_result = registry.execute_tool(step.tool_name, resolved_args)

        if "error" in raw_result:
            return StepResult(
                step_id=step.step_id,
                description=step.description,
                tool_name=step.tool_name,
                tool_args=resolved_args,
                tool_output=None,
                error=str(raw_result["error"]),
                success=False,
            )

        return StepResult(
            step_id=step.step_id,
            description=step.description,
            tool_name=step.tool_name,
            tool_args=resolved_args,
            tool_output=raw_result.get("result"),
            error=None,
            success=True,
        )

    except Exception as exc:
        return StepResult(
            step_id=step.step_id,
            description=step.description,
            tool_name=step.tool_name,
            tool_args=step.tool_args,
            tool_output=None,
            error=str(exc),
            success=False,
        )


def format_execution_trace(result: ExecutionResult) -> str:
    """Format execution result as markdown trace."""
    lines: list[str] = []
    tool_count = sum(1 for r in result.step_results if r.tool_name is not None)

    lines.append(f"**Execution Trace** ({result.total_steps} steps, {tool_count} tools)")
    lines.append("")

    for sr in result.step_results:
        lines.append(f"**Step {sr.step_id}**: {sr.description}")

        if sr.tool_name is None:
            lines.append("  (reasoning step)")
        else:
            args_str = ", ".join(f"{k}={v}" for k, v in sr.tool_args.items())
            lines.append(f"  → `{sr.tool_name}({args_str})`")

            if sr.success and sr.tool_output:
                out_str = ", ".join(f"{k}={v}" for k, v in list(sr.tool_output.items())[:4])
                lines.append(f"  ✓ Success: {out_str}")
            elif not sr.success:
                lines.append(f"  ✗ Failed: {sr.error}")

        lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    from agent.planner import ExecutionPlan, PlanStep

    demo_plan = ExecutionPlan(
        reasoning="Test plan",
        complexity="moderate",
        steps=[
            PlanStep(
                step_id=1,
                description="Analyze budget",
                tool_name="analyze_budget",
                tool_args={
                    "monthly_income": 30000000,
                    "expenses": {"rent": 8000000, "food": 4000000},
                },
                depends_on=[],
            ),
            PlanStep(
                step_id=2,
                description="Calculate loan payment",
                tool_name="calculate_loan_payment",
                tool_args={
                    "principal": 1000000000,
                    "annual_rate_percent": 8,
                    "years": 5,
                },
                depends_on=[],
            ),
        ],
    )

    result = execute_plan(demo_plan)
    print(format_execution_trace(result))
    print(f"\nTotal: {result.total_steps}, Failed: {result.failed_steps}")
