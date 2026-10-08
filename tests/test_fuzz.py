"""Fuzz testing cho FinGuard guardrail.

Dùng Hypothesis để tự động sinh hàng nghìn input ngẫu nhiên,
kiểm tra guardrail không bao giờ crash và luôn trả kết quả hợp lệ.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import guardrail


@pytest.fixture(autouse=True)
def disable_remote_nemo_for_fuzz(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep randomized fuzz cases local; NeMo integration has dedicated tests."""
    monkeypatch.setattr(guardrail.nemo_guard, "_get_rails", lambda: None)


# ============================================================
# CẤU HÌNH CHUNG
# ============================================================
FUZZ_SETTINGS = settings(
    max_examples=500,  # 500 inputs ngẫu nhiên mỗi test
    deadline=None,  # Presidio có thể chậm, bỏ deadline
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
)


# ============================================================
# TEST 1 — normalize_text không bao giờ crash
# ============================================================
@given(st.text())
@FUZZ_SETTINGS
def test_fuzz_normalize_never_crashes(text: str) -> None:
    """normalize_text phải luôn trả string."""
    result = guardrail.normalize_text(text)
    assert isinstance(result, str)


@given(st.text())
@FUZZ_SETTINGS
def test_fuzz_normalize_idempotent(text: str) -> None:
    """Normalize 2 lần = normalize 1 lần."""
    once = guardrail.normalize_text(text)
    twice = guardrail.normalize_text(once)
    assert once == twice


# ============================================================
# TEST 2 — detect_prompt_injection không bao giờ crash
# ============================================================
@given(st.text())
@FUZZ_SETTINGS
def test_fuzz_injection_never_crashes(text: str) -> None:
    """detect_prompt_injection phải luôn trả (bool, float, str)."""
    detected, score, reason = guardrail.detect_prompt_injection(text)
    assert isinstance(detected, bool)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0
    assert isinstance(reason, str)


@given(st.text(min_size=1, max_size=500))
@FUZZ_SETTINGS
def test_fuzz_injection_score_bounded(text: str) -> None:
    """Score luôn trong [0, 1]."""
    _, score, _ = guardrail.detect_prompt_injection(text)
    assert 0.0 <= score <= 1.0


# ============================================================
# TEST 3 — mask_pii không bao giờ crash + không làm mất text
# ============================================================
@given(st.text())
@FUZZ_SETTINGS
def test_fuzz_mask_pii_never_crashes(text: str) -> None:
    """mask_pii phải luôn trả (str, list)."""
    masked, findings = guardrail.mask_pii(text)
    assert isinstance(masked, str)
    assert isinstance(findings, list)


@given(st.text(max_size=200))
@FUZZ_SETTINGS
def test_fuzz_mask_pii_length_bounded(text: str) -> None:
    """Mask không làm text dài hơn 10x (tránh infinite loop)."""
    masked, _ = guardrail.mask_pii(text)
    assert len(masked) <= len(text) * 10 + 100


# ============================================================
# TEST 4 — luhn_check không bao giờ crash
# ============================================================
@given(st.text())
@FUZZ_SETTINGS
def test_fuzz_luhn_never_crashes(text: str) -> None:
    """luhn_check phải luôn trả bool."""
    result = guardrail.luhn_check(text)
    assert isinstance(result, bool)


@given(st.integers(min_value=0, max_value=10**20))
@FUZZ_SETTINGS
def test_fuzz_luhn_with_numbers(num: int) -> None:
    """luhn_check với số ngẫu nhiên không crash."""
    result = guardrail.luhn_check(str(num))
    assert isinstance(result, bool)


# ============================================================
# TEST 5 — check_financial_compliance không bao giờ crash
# ============================================================
@given(st.text())
@FUZZ_SETTINGS
def test_fuzz_compliance_never_crashes(text: str) -> None:
    """check_financial_compliance phải luôn trả tuple 3 phần tử."""
    result = guardrail.check_financial_compliance(text)
    assert isinstance(result, tuple)
    assert len(result) == 3
    violated, score, reason = result
    assert isinstance(violated, bool)
    assert isinstance(score, float)
    assert isinstance(reason, str)


# ============================================================
# TEST 6 — process_input luôn trả GuardrailResult hợp lệ
# ============================================================
@given(st.text())
@FUZZ_SETTINGS
def test_fuzz_process_input_never_crashes(text: str) -> None:
    """process_input phải luôn trả GuardrailResult."""
    result = guardrail.process_input(text)
    assert isinstance(result.allowed, bool)
    assert isinstance(result.layer, str)
    assert isinstance(result.processed_text, str)
    assert isinstance(result.findings, list)


@given(st.text())
@FUZZ_SETTINGS
def test_fuzz_process_input_empty_blocked(text: str) -> None:
    """Text rỗng hoặc chỉ whitespace phải bị block."""
    if not text.strip():
        result = guardrail.process_input(text)
        assert result.allowed is False


# ============================================================
# TEST 7 — process_output luôn trả GuardrailResult
# ============================================================
@given(st.text())
@FUZZ_SETTINGS
def test_fuzz_process_output_never_crashes(text: str) -> None:
    """process_output phải luôn trả GuardrailResult."""
    result = guardrail.process_output(text)
    assert isinstance(result.allowed, bool)
    assert isinstance(result.processed_text, str)


# ============================================================
# TEST 8 — Unicode attacks
# ============================================================
@given(st.text(min_size=1))
@FUZZ_SETTINGS
def test_fuzz_zero_width_injection(text: str) -> None:
    """Input có zero-width chars không crash."""
    payload = text + "\u200b" + text
    result = guardrail.process_input(payload)
    assert isinstance(result.allowed, bool)


@given(st.text(min_size=1))
@FUZZ_SETTINGS
def test_fuzz_fullwidth_text(text: str) -> None:
    """Fullwidth characters không crash."""
    import unicodedata

    payload = unicodedata.normalize("NFKC", text)
    result = guardrail.process_input(payload)
    assert isinstance(result.allowed, bool)


# ============================================================
# TEST 9 — Combined adversarial
# ============================================================
@given(
    st.text(min_size=1, max_size=100),
    st.text(min_size=1, max_size=100),
)
@FUZZ_SETTINGS
def test_fuzz_concatenated_inputs(a: str, b: str) -> None:
    """Ghép 2 input ngẫu nhiên không crash."""
    combined = a + " " + b
    result = guardrail.process_input(combined)
    assert isinstance(result.allowed, bool)


@given(st.lists(st.text(min_size=1, max_size=50), min_size=1, max_size=10))
@FUZZ_SETTINGS
def test_fuzz_list_of_inputs(texts: list) -> None:
    """Process nhiều input liên tiếp không crash."""
    for t in texts:
        result = guardrail.process_input(t)
        assert isinstance(result.allowed, bool)
