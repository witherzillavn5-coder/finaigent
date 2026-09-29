"""Financial calculators for common personal finance questions."""

import math
from typing import Any


def _validate_finite(name: str, value: float) -> None:
    """Reject non-numeric or non-finite values."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number.")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number.")


def _format_vnd(amount: float) -> str:
    """Format an amount as Vietnamese dong."""
    return f"{amount:,.0f} VND"


def calculate_compound_interest(
    principal: float,
    annual_rate_percent: float,
    years: float,
    compounds_per_year: int = 12,
) -> dict[str, Any]:
    """Calculate compound growth and earned interest."""
    _validate_finite("principal", principal)
    _validate_finite("annual_rate_percent", annual_rate_percent)
    _validate_finite("years", years)
    if principal <= 0:
        raise ValueError("principal must be greater than zero.")
    if annual_rate_percent < 0:
        raise ValueError("annual_rate_percent cannot be negative.")
    if years <= 0:
        raise ValueError("years must be greater than zero.")
    if isinstance(compounds_per_year, bool) or compounds_per_year <= 0:
        raise ValueError("compounds_per_year must be a positive integer.")

    periodic_rate = annual_rate_percent / 100 / compounds_per_year
    periods = compounds_per_year * years
    try:
        final_amount = principal * math.pow(1 + periodic_rate, periods)
    except OverflowError as error:
        raise ValueError("The calculated final amount is too large.") from error
    if not math.isfinite(final_amount):
        raise ValueError("The calculated final amount is too large.")

    interest_earned = final_amount - principal
    frequency = {
        1: "annually",
        2: "semiannually",
        4: "quarterly",
        12: "monthly",
        365: "daily",
    }.get(compounds_per_year, f"{compounds_per_year} times per year")
    explanation = (
        f"{_format_vnd(principal)} grows to {_format_vnd(final_amount)} after "
        f"{years:g} years at {annual_rate_percent:g}% annual rate, compounding "
        f"{frequency}. Interest earned: {_format_vnd(interest_earned)}."
    )
    return {
        "final_amount": final_amount,
        "interest_earned": interest_earned,
        "explanation": explanation,
    }


def calculate_loan_payment(
    principal: float,
    annual_rate_percent: float,
    years: float,
) -> dict[str, Any]:
    """Calculate fixed monthly loan payments and total interest."""
    _validate_finite("principal", principal)
    _validate_finite("annual_rate_percent", annual_rate_percent)
    _validate_finite("years", years)
    if principal <= 0:
        raise ValueError("principal must be greater than zero.")
    if annual_rate_percent < 0:
        raise ValueError("annual_rate_percent cannot be negative.")
    if years <= 0:
        raise ValueError("years must be greater than zero.")

    payment_count = years * 12
    if not math.isfinite(payment_count):
        raise ValueError("The number of monthly payments is too large.")
    monthly_rate = annual_rate_percent / 12 / 100
    if monthly_rate == 0:
        monthly_payment = principal / payment_count
    else:
        try:
            discount_factor = math.pow(1 + monthly_rate, -payment_count)
        except OverflowError as error:
            raise ValueError("The calculated monthly payment is not finite.") from error
        monthly_payment = principal * monthly_rate / (1 - discount_factor)

    total_payment = monthly_payment * payment_count
    total_interest = total_payment - principal
    if not all(math.isfinite(value) for value in (monthly_payment, total_payment, total_interest)):
        raise ValueError("The calculated loan payments are not finite.")

    explanation = (
        f"A {_format_vnd(principal)} loan over {years:g} years at "
        f"{annual_rate_percent:g}% annual interest requires monthly payments of "
        f"{_format_vnd(monthly_payment)}. Total payment: {_format_vnd(total_payment)}; "
        f"total interest: {_format_vnd(total_interest)}."
    )
    return {
        "monthly_payment": monthly_payment,
        "total_payment": total_payment,
        "total_interest": total_interest,
        "explanation": explanation,
    }


def calculate_savings_goal(
    target_amount: float,
    monthly_savings: float,
    annual_rate_percent: float = 0,
) -> dict[str, Any]:
    """Estimate months needed to reach a savings target."""
    _validate_finite("target_amount", target_amount)
    _validate_finite("monthly_savings", monthly_savings)
    _validate_finite("annual_rate_percent", annual_rate_percent)
    if target_amount <= 0:
        raise ValueError("target_amount must be greater than zero.")
    if monthly_savings <= 0:
        raise ValueError("monthly_savings must be greater than zero.")
    if annual_rate_percent < 0:
        raise ValueError("annual_rate_percent cannot be negative.")

    monthly_rate = annual_rate_percent / 12 / 100
    if monthly_rate == 0:
        months = math.ceil(target_amount / monthly_savings)
    else:
        try:
            periods_needed = math.log1p(
                target_amount * monthly_rate / monthly_savings
            ) / math.log1p(monthly_rate)
            months = math.ceil(periods_needed)
        except (OverflowError, ValueError) as error:
            raise ValueError(
                "The savings goal calculation is outside the supported range."
            ) from error

    if months <= 0:
        raise ValueError("The savings goal calculation did not produce a valid month count.")

    total_contributed = monthly_savings * months
    if monthly_rate == 0:
        accumulated_amount = total_contributed
    else:
        try:
            accumulated_amount = (
                monthly_savings * math.expm1(months * math.log1p(monthly_rate)) / monthly_rate
            )
        except (OverflowError, ValueError) as error:
            raise ValueError(
                "The savings goal calculation is outside the supported range."
            ) from error
    interest_earned = max(0.0, accumulated_amount - total_contributed)
    if not all(math.isfinite(value) for value in (total_contributed, interest_earned)):
        raise ValueError("The savings goal calculation is outside the supported range.")

    years = months / 12
    explanation = (
        f"At {_format_vnd(monthly_savings)} saved monthly and {annual_rate_percent:g}% "
        f"annual interest, you can reach {_format_vnd(target_amount)} in {months} months "
        f"({years:.2f} years). Total contributed: {_format_vnd(total_contributed)}; "
        f"estimated interest earned: {_format_vnd(interest_earned)}."
    )
    return {
        "months": months,
        "years": years,
        "total_contributed": total_contributed,
        "interest_earned": interest_earned,
        "explanation": explanation,
    }


def convert_currency(
    amount: float,
    from_currency: str,
    to_currency: str,
) -> dict[str, Any]:
    """Convert currency using static, non-real-time exchange rates."""
    _validate_finite("amount", amount)
    if amount < 0:
        raise ValueError("amount cannot be negative.")
    if not isinstance(from_currency, str) or not isinstance(to_currency, str):
        raise ValueError("from_currency and to_currency must be currency codes.")

    source = from_currency.strip().upper()
    target = to_currency.strip().upper()
    exchange_rates = {
        ("USD", "VND"): 24500.0,
        ("EUR", "VND"): 26500.0,
        ("VND", "USD"): 1 / 24500,
        ("VND", "EUR"): 1 / 26500,
        ("USD", "EUR"): 0.92,
        ("EUR", "USD"): 1.08,
    }
    pair = (source, target)
    if pair not in exchange_rates:
        supported_pairs = ", ".join(f"{start}->{end}" for start, end in exchange_rates)
        raise ValueError(
            f"Unsupported currency pair {source or from_currency}->{target or to_currency}. "
            f"Supported pairs: {supported_pairs}."
        )

    rate_used = exchange_rates[pair]
    converted_amount = amount * rate_used
    if not math.isfinite(converted_amount):
        raise ValueError("The converted amount is too large.")
    explanation = (
        f"{amount:,.2f} {source} converts to {converted_amount:,.2f} {target} "
        f"at a static rate of {rate_used:g} {target} per {source}."
    )
    return {
        "converted_amount": converted_amount,
        "rate_used": rate_used,
        "from": source,
        "to": target,
        "explanation": explanation,
    }


def analyze_budget(monthly_income: float, expenses: dict[str, float]) -> dict[str, Any]:
    """Compare categorized monthly spending with the 50/30/20 guideline."""
    _validate_finite("monthly_income", monthly_income)
    if monthly_income <= 0:
        raise ValueError("monthly_income must be greater than zero.")
    if not isinstance(expenses, dict):
        raise ValueError("expenses must be a dictionary of category amounts.")

    needs_keywords = ("rent", "utilities", "food", "transport", "insurance", "healthcare")
    wants_keywords = ("entertainment", "dining_out", "shopping", "subscriptions")
    savings_keywords = ("savings", "investment", "emergency_fund")
    group_totals = {"needs": 0.0, "wants": 0.0, "savings": 0.0}
    total_expenses = 0.0

    for category, amount in expenses.items():
        if not isinstance(category, str):
            raise ValueError("Expense category names must be strings.")
        _validate_finite(f"expense '{category}'", amount)
        if amount < 0:
            raise ValueError(f"Expense '{category}' cannot be negative.")

        total_expenses += amount
        normalized_category = category.strip().lower().replace("-", "_").replace(" ", "_")
        if any(keyword in normalized_category for keyword in needs_keywords):
            group_totals["needs"] += amount
        elif any(keyword in normalized_category for keyword in wants_keywords):
            group_totals["wants"] += amount
        elif any(keyword in normalized_category for keyword in savings_keywords):
            group_totals["savings"] += amount

    remaining = monthly_income - total_expenses
    savings_rate_percent = remaining / monthly_income * 100
    needs_percent = group_totals["needs"] / monthly_income * 100
    wants_percent = group_totals["wants"] / monthly_income * 100
    savings_percent = group_totals["savings"] / monthly_income * 100
    percentages = (savings_rate_percent, needs_percent, wants_percent, savings_percent)
    if not all(math.isfinite(value) for value in (total_expenses, remaining, *percentages)):
        raise ValueError("The budget totals are too large to calculate safely.")

    recommendations: list[str] = []
    if remaining < 0:
        recommendations.append(
            f"Expenses exceed income by {_format_vnd(-remaining)}; review spending or income."
        )
    if needs_percent > 50:
        recommendations.append(
            f"Your needs are {needs_percent:.1f}% of income (target: 50%) - consider reducing fixed costs."
        )
    if wants_percent > 30:
        recommendations.append(
            f"Your wants are {wants_percent:.1f}% of income (target: 30%) - review discretionary spending."
        )
    if savings_percent < 20:
        recommendations.append(
            f"Your categorized savings are {savings_percent:.1f}% of income (target: 20%) - "
            "increase savings contributions if feasible."
        )
    if not recommendations:
        recommendations.append("Your categorized budget is within the 50/30/20 targets.")

    explanation = (
        f"Monthly income is {_format_vnd(monthly_income)}. Expenses total "
        f"{_format_vnd(total_expenses)}, leaving {_format_vnd(remaining)}. "
        f"Categorized allocation: needs {needs_percent:.1f}% (target 50%), wants "
        f"{wants_percent:.1f}% (target 30%), and savings {savings_percent:.1f}% "
        "(target 20%)."
    )
    return {
        "total_expenses": total_expenses,
        "remaining": remaining,
        "savings_rate_percent": savings_rate_percent,
        "needs_percent": needs_percent,
        "wants_percent": wants_percent,
        "savings_percent": savings_percent,
        "recommendations": recommendations,
        "explanation": explanation,
    }


if __name__ == "__main__":
    demos = (
        calculate_compound_interest(10_000_000, 8, 2),
        calculate_loan_payment(100_000_000, 6, 5),
        calculate_savings_goal(50_000_000, 2_000_000, 5),
        convert_currency(100, "USD", "VND"),
        analyze_budget(
            20_000_000,
            {"rent": 5_000_000, "food": 3_000_000, "transport": 1_000_000},
        ),
    )
    for demo in demos:
        print(demo["explanation"])
