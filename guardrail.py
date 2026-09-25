"""Công cụ bảo vệ đầu vào/đầu ra cho FinGuard Agent."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import List, Tuple

ZERO_WIDTH_CHARS: str = "\u200b\u200c\u200d\u2060\ufeff\u180e\u200e\u200f"

HARD_INJECTION_PATTERNS: Tuple[re.Pattern[str], ...] = (
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s*instructions?", re.I),
    re.compile(r"forget\s+(all\s+)?(previous|prior|your)\s+(instructions?|rules?)", re.I),
    re.compile(r"you\s+are\s+now\s+(a|an|in)\b", re.I),
    re.compile(r"\b(jailbreak|dan\s*mode|developer\s*mode)\b", re.I),
    re.compile(r"reveal\s+(your\s+)?(system\s+)?prompt", re.I),
    re.compile(r"override\s+(the\s+)?(safety|guardrail|security|rules?)", re.I),
    re.compile(r"(disable|bypass)\s+(your\s+)?(safety|filter|guardrail|restrictions?)", re.I),
    re.compile(r"bỏ\s+qua\s+(mọi\s+)?(hướng\s+dẫn|quy\s+tắc|chỉ\s+thị)", re.I),
)

SOFT_INJECTION_PATTERNS: Tuple[re.Pattern[str], ...] = (
    re.compile(r"\bacHARD_INJECTION_PATTERNSt\s+as\b", re.I),
    re.compile(r"\brole[\s-]?play\b", re.I),
    re.compile(r"\bhypothetically\b", re.I),
    re.compile(r"without\s+(any\s+)?(restrictions?|limits?|rules?)", re.I),
    re.compile(r"pretend\s+(you\s+)?(have\s+)?no\s+(rules?|limits?|restrictions?)", re.I),
    re.compile(r"from\s+now\s+on\s+you", re.I),
)

COMPLIANCE_RULES: Tuple[Tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"\b(wire\s*transfer|chuyển\s*khoản|chuyen\s*khoan|send\s+money|"
            r"bank\s+transfer|swift\s+transfer)\b",
            re.I,
        ),
        "Yêu cầu chuyển khoản / wire transfer bị chặn.",
    ),
    (
        re.compile(
            r"\b(bypass\s+(auth|authentication|login|2fa|mfa)|vượt\s+qua\s+"
            r"(xác\s+thực|đăng\s+nhập)|skip\s+(auth|2fa|otp))\b",
            re.I,
        ),
        "Yêu cầu bypass xác thực bị chặn.",
    ),
    (
        re.compile(
            r"\b(modify|change|increase|inflate)\s+(account\s+)?(balance|số\s+dư)|"
            r"sửa\s+số\s+dư|thay\s+đổi\s+số\s+dư",
            re.I,
        ),
        "Yêu cầu thay đổi số dư bị chặn.",
    ),
    (
        re.compile(
            r"\b(disable|tắt|turn\s+off)\s+(2fa|mfa|two[\s-]?factor|"
            r"xác\s+thực\s+hai\s+yếu\s+tố)\b",
            re.I,
        ),
        "Yêu cầu tắt 2FA bị chặn.",
    ),
    (
        re.compile(
            r"\b(hack|phishing|fraud|launder|rửa\s+tiền|gian\s+lận|"
            r"lừa\s+đảo|money\s+laundering)\b",
            re.I,
        ),
        "Yêu cầu liên quan hack/gian lận/rửa tiền bị chặn.",
    ),
)

SSN_RE: re.Pattern[str] = re.compile(r"\b(\d{3}-\d{2}-\d{4})\b")
CARD_RE: re.Pattern[str] = re.compile(r"(?:\d[\s-]?){13,19}")
OTP_RE: re.Pattern[str] = re.compile(
    r"(?i)(?:otp|one[\s-]?time(?:\s+pass(?:word|code)?)?|mã\s+otp|"
    r"mã\s+xác\s+thực)\s*[:#-]?\s*(\d{4,8})"
)
CVV_CONTEXT_RE: re.Pattern[str] = re.compile(
    r"(?i)\b(cvv|cvc|cid|security\s+code|mã\s+bảo\s+mật)\b"
)
CVV_NEAR_RE: re.Pattern[str] = re.compile(
    r"(?i)(?:cvv|cvc|cid|security\s+code|mã\s+bảo\s+mật)\s*[:#-]?\s*(\d{3,4})"
)

SYSTEM_LEAK_RE: re.Pattern[str] = re.compile(
    r"(?i)(system\s+prompt|hướng\s+dẫn\s+nội\s+bộ|you\s+are\s+fin\s*guard|"
    r"bạn\s+là\s+fin\s*guard|quy\s+tắc\s+bắt\s+buộc)"
)
RISKY_ADVICE_RE: re.Pattern[str] = re.compile(
    r"(?i)\b(buy\s+now|guaranteed\s+return|lợi\s+nhuận\s+đảm\s+bảo|"
    r"mua\s+ngay|chắc\s+chắn\s+sinh\s+lời)\b"
)


@dataclass
class GuardrailResult:
    """Kết quả một lớp kiểm tra bảo mật."""

    allowed: bool
    layer: str
    reason: str
    risk_score: float
    processed_text: str
    findings: List[str] = field(default_factory=list)


def normalize_text(text: str) -> str:
    """Chuẩn hóa Unicode NFKC, bỏ ký tự zero-width và gộp khoảng trắng."""
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKC", text)
    translator = str.maketrans("", "", ZERO_WIDTH_CHARS)
    cleaned = normalized.translate(translator)
    return re.sub(r"\s+", " ", cleaned).strip()


def luhn_check(number: str) -> bool:
    """Kiểm tra dãy số theo thuật toán Luhn (thẻ tín dụng)."""
    digits = re.sub(r"\D", "", number)
    if not digits or not digits.isdigit():
        return False
    total = 0
    reverse = digits[::-1]
    for i, ch in enumerate(reverse):
        n = ord(ch) - 48
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


def detect_prompt_injection(text: str) -> Tuple[bool, float, str]:
    """Phát hiện prompt injection bằng pattern cứng, mềm và heuristic.

    Trả về (bị_phát_hiện, điểm_rủi_ro, lý_do).
    """
    normalized = normalize_text(text)
    if not normalized:
        return False, 0.0, ""

    for pattern in HARD_INJECTION_PATTERNS:
        if pattern.search(normalized):
            return True, 1.0, "Phát hiện prompt injection (hard pattern)."

    score = 0.0
    matched_soft: List[str] = []
    for pattern in SOFT_INJECTION_PATTERNS:
        if pattern.search(normalized):
            score += 0.35
            matched_soft.append(pattern.pattern)

    lowered = normalized.lower()
    if lowered.count("ignore") >= 2 and "instruction" in lowered:
        score += 0.3
    if "system prompt" in lowered or "system_prompt" in lowered:
        score += 0.4
    if re.search(r"[A-Z]{8,}", normalized) and "jail" in lowered:
        score += 0.2

    score = min(score, 1.0)
    if score >= 0.6:
        reason = "Phát hiện prompt injection (soft/heuristic)."
        return True, score, reason
    if matched_soft:
        return False, score, "Dấu hiệu injection nhẹ, chưa đủ ngưỡng chặn."
    return False, 0.0, ""


def mask_pii(text: str) -> Tuple[str, List[str]]:
    """Che PII: SSN, thẻ (Luhn), OTP, CVV khi có ngữ cảnh."""
    findings: List[str] = []
    result = text

    def _mask_ssn(match: re.Match[str]) -> str:
        findings.append("ssn")
        return "[SSN_REDACTED]"

    result = SSN_RE.sub(_mask_ssn, result)

    def _mask_card(match: re.Match[str]) -> str:
        raw = match.group(0)
        digits = re.sub(r"\D", "", raw)
        if 13 <= len(digits) <= 19 and luhn_check(digits):
            findings.append("credit_card")
            return "[CARD_REDACTED]"
        return raw

    result = CARD_RE.sub(_mask_card, result)

    def _mask_otp(match: re.Match[str]) -> str:
        findings.append("otp")
        return match.group(0).replace(match.group(1), "[OTP_REDACTED]")

    result = OTP_RE.sub(_mask_otp, result)

    if CVV_CONTEXT_RE.search(result):

        def _mask_cvv(match: re.Match[str]) -> str:
            findings.append("cvv")
            return match.group(0).replace(match.group(1), "[CVV_REDACTED]")

        result = CVV_NEAR_RE.sub(_mask_cvv, result)

    return result, findings


def check_financial_compliance(text: str) -> Tuple[bool, float, str]:
    """Chặn yêu cầu tài chính nguy hiểm (chuyển khoản, bypass, gian lận).

    Trả về (vi_phạm, điểm_rủi_ro, lý_do). True nghĩa là vi phạm.
    """
    normalized = normalize_text(text)
    for pattern, reason in COMPLIANCE_RULES:
        if pattern.search(normalized):
            return True, 0.95, reason
    return False, 0.0, ""


def check_output(text: str) -> GuardrailResult:
    """Kiểm tra đầu ra: lộ prompt, lời khuyên rủi ro, PII còn sót."""
    if not text:
        return GuardrailResult(
            allowed=True,
            layer="output",
            reason="Đầu ra trống.",
            risk_score=0.0,
            processed_text="",
            findings=[],
        )

    findings: List[str] = []
    risk = 0.0
    allowed = True
    reasons: List[str] = []
    processed = text

    if SYSTEM_LEAK_RE.search(text):
        allowed = False
        risk = max(risk, 0.9)
        findings.append("system_prompt_leak")
        reasons.append("Phát hiện nguy cơ lộ system prompt.")
        processed = (
            "Nội dung đã bị chặn vì có dấu hiệu tiết lộ hướng dẫn nội bộ."
        )

    if RISKY_ADVICE_RE.search(text):
        allowed = False
        risk = max(risk, 0.85)
        findings.append("risky_investment_advice")
        reasons.append("Phát hiện lời khuyên đầu tư rủi ro.")
        processed = (
            "Nội dung đã bị chặn vì chứa lời khuyên đầu tư không phù hợp "
            "(ví dụ bảo đảm lợi nhuận hoặc kêu gọi mua ngay)."
        )

    masked, pii_findings = mask_pii(processed if allowed else text)
    leftover = pii_findings
    if leftover:
        findings.extend(f"output_pii:{item}" for item in leftover)
        risk = max(risk, 0.5)
        reasons.append("Phát hiện PII còn sót trên đầu ra và đã che.")
        processed = masked

    if allowed and leftover:
        # PII được che nhưng vẫn cho phép hiển thị bản đã mask.
        allowed = True

    reason = " ".join(reasons) if reasons else "Đầu ra đạt kiểm tra."
    return GuardrailResult(
        allowed=allowed,
        layer="output",
        reason=reason,
        risk_score=risk,
        processed_text=processed,
        findings=findings,
    )


def process_input(text: str) -> GuardrailResult:
    """Pipeline: normalize → injection → compliance → mask PII."""
    normalized = normalize_text(text)

    injected, inj_score, inj_reason = detect_prompt_injection(normalized)
    if injected:
        return GuardrailResult(
            allowed=False,
            layer="injection",
            reason=inj_reason,
            risk_score=inj_score,
            processed_text=normalized,
            findings=["prompt_injection"],
        )

    violated, comp_score, comp_reason = check_financial_compliance(normalized)
    if violated:
        return GuardrailResult(
            allowed=False,
            layer="compliance",
            reason=comp_reason,
            risk_score=comp_score,
            processed_text=normalized,
            findings=["financial_compliance"],
        )

    masked, pii_findings = mask_pii(normalized)
    if pii_findings:
        return GuardrailResult(
            allowed=True,
            layer="pii",
            reason="Đã che thông tin cá nhân nhạy cảm.",
            risk_score=max(0.4, inj_score),
            processed_text=masked,
            findings=pii_findings,
        )

    return GuardrailResult(
        allowed=True,
        layer="pass",
        reason="Đầu vào hợp lệ.",
        risk_score=inj_score,
        processed_text=masked,
        findings=[],
    )


def process_output(text: str) -> GuardrailResult:
    """Wrapper kiểm tra đầu ra mô hình."""
    return check_output(text)
