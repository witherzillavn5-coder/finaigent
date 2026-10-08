import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from tools import financial_calculators as fc
from tools import registry


def test_compound_interest_basic() -> None:
    """Calculate monthly compound growth."""
    result = fc.calculate_compound_interest(10_000_000, 8, 2)
    assert result["final_amount"] == pytest.approx(11_730_000, rel=0.01)


def test_compound_interest_zero_rate() -> None:
    """Preserve principal when the rate is zero."""
    result = fc.calculate_compound_interest(10_000_000, 0, 2)
    assert result["final_amount"] == pytest.approx(10_000_000, rel=0.01)


def test_compound_interest_annual_compound() -> None:
    """Compound once per year when requested."""
    result = fc.calculate_compound_interest(10_000_000, 8, 2, compounds_per_year=1)
    assert result["final_amount"] == pytest.approx(11_664_000, rel=0.01)


def test_compound_interest_negative_principal() -> None:
    """Reject a negative principal."""
    with pytest.raises(ValueError):
        fc.calculate_compound_interest(-10_000_000, 8, 2)


def test_compound_interest_zero_years() -> None:
    """Reject a zero-year investment period."""
    with pytest.raises(ValueError):
        fc.calculate_compound_interest(10_000_000, 8, 0)


def test_loan_basic() -> None:
    """Calculate a fixed monthly loan payment."""
    result = fc.calculate_loan_payment(100_000_000, 6, 5)
    assert result["monthly_payment"] == pytest.approx(1_930_000, rel=0.01)


def test_loan_zero_rate() -> None:
    """Divide a zero-interest loan evenly across its term."""
    result = fc.calculate_loan_payment(100_000_000, 0, 5)
    assert result["monthly_payment"] == pytest.approx(100_000_000 / 60, rel=0.01)


def test_loan_negative_principal() -> None:
    """Reject a negative loan principal."""
    with pytest.raises(ValueError):
        fc.calculate_loan_payment(-100_000_000, 6, 5)


def test_loan_negative_rate() -> None:
    """Reject a negative loan interest rate."""
    with pytest.raises(ValueError):
        fc.calculate_loan_payment(100_000_000, -1, 5)


def test_savings_basic() -> None:
    """Estimate months to reach a savings target with interest."""
    result = fc.calculate_savings_goal(50_000_000, 2_000_000, 5)
    assert result["months"] == pytest.approx(24, rel=0.01)


def test_savings_zero_rate() -> None:
    """Estimate savings duration without interest."""
    result = fc.calculate_savings_goal(50_000_000, 2_000_000, 0)
    assert result["months"] == pytest.approx(25, rel=0.01)


def test_savings_zero_monthly() -> None:
    """Reject a zero monthly savings contribution."""
    with pytest.raises(ValueError):
        fc.calculate_savings_goal(50_000_000, 0)


def test_savings_target_less_than_monthly() -> None:
    """Reach a smaller target in one month."""
    result = fc.calculate_savings_goal(1_000_000, 5_000_000)
    assert result["months"] == 1


def test_convert_usd_to_vnd() -> None:
    """Convert US dollars to Vietnamese dong."""
    result = fc.convert_currency(100, "USD", "VND")
    assert result["converted_amount"] == pytest.approx(2_450_000, rel=0.01)


def test_convert_vnd_to_usd() -> None:
    """Convert Vietnamese dong to US dollars."""
    result = fc.convert_currency(24_500, "VND", "USD")
    assert result["converted_amount"] == pytest.approx(1, rel=0.01)


def test_convert_unsupported_pair() -> None:
    """Reject an unsupported currency pair."""
    with pytest.raises(ValueError):
        fc.convert_currency(100, "JPY", "VND")


def test_convert_same_currency() -> None:
    """Preserve an amount when source and target currencies match."""
    result = fc.convert_currency(100, "USD", "USD")
    assert result["converted_amount"] == pytest.approx(100, rel=0.01)


def test_convert_crypto_to_vnd() -> None:
    """Convert Bitcoin to Vietnamese dong using the static reference price."""
    result = fc.convert_crypto(1, "BTC", "VND")
    assert result["converted_amount"] == pytest.approx(2_327_500_000)
    assert result["is_real_time"] is False


def test_convert_crypto_rejects_unsupported_coin() -> None:
    """Reject coins without a static reference price."""
    with pytest.raises(ValueError, match="Unsupported coin"):
        fc.convert_crypto(1, "DOGE")


def test_budget_basic() -> None:
    """Calculate remaining income after expenses."""
    result = fc.analyze_budget(20_000_000, {"rent": 5_000_000, "food": 3_000_000})
    assert result["remaining"] == pytest.approx(12_000_000, rel=0.01)


def test_budget_overspending() -> None:
    """Report negative remaining income when overspending."""
    result = fc.analyze_budget(10_000_000, {"rent": 8_000_000, "food": 4_000_000})
    assert result["remaining"] < 0


def test_budget_classification() -> None:
    """Calculate percentages for needs and wants categories."""
    result = fc.analyze_budget(
        20_000_000,
        {"rent": 5_000_000, "entertainment": 2_000_000},
    )
    assert result["needs_percent"] == pytest.approx(25, rel=0.01)
    assert result["wants_percent"] == pytest.approx(10, rel=0.01)


def test_budget_zero_income() -> None:
    """Reject a zero monthly income."""
    with pytest.raises(ValueError):
        fc.analyze_budget(0, {"rent": 1_000_000})


def test_registry_has_9_tools() -> None:
    """Register all nine financial calculators."""
    assert len(registry.get_tool_names()) == 9


def test_get_tool_schemas_format() -> None:
    """Return function-calling schemas for each registered tool."""
    schemas = registry.get_tool_schemas()
    assert all(schema["type"] == "function" for schema in schemas)
    assert all(isinstance(schema["function"], dict) for schema in schemas)
    assert any(schema["function"]["name"] == "convert_crypto" for schema in schemas)


def test_execute_tool_valid() -> None:
    """Return a result when executing a valid registered tool."""
    response = registry.execute_tool(
        "calculate_loan_payment",
        {"principal": 100_000_000, "annual_rate_percent": 6, "years": 5},
    )
    assert "result" in response


def test_execute_tool_unknown() -> None:
    """Return an error for an unknown tool name."""
    response = registry.execute_tool("unknown_tool", {})
    assert "error" in response


def test_execute_tool_bad_args() -> None:
    """Return an error when tool arguments are invalid."""
    response = registry.execute_tool("calculate_loan_payment", {"principal": 100_000_000})
    assert "error" in response
