"""FinGuard Security Guardrail Engine.

6 lớp bảo vệ:
- Normalize Unicode
- Detect prompt injection (regex hard + soft)
- PII detection: Presidio (ML) + regex fallback (SSN, OTP, CVV, VN phone)
- Compliance check
- Output validator
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from openai import OpenAI

import config
import nemo_guard

# --- Presidio optional import ---
try:
    from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
    from presidio_analyzer.nlp_engine import NlpEngineProvider

    _PRESIDIO_AVAILABLE = True
except ImportError:
    _PRESIDIO_AVAILABLE = False


# ============================================================
# LỚP 0 — NORMALIZE
# ============================================================

ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060-\u206f\ufeff]")


def normalize_text(text: str) -> str:
    """NFKC + xóa zero-width + gộp khoảng trắng."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = ZERO_WIDTH_RE.sub("", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _strip_vietnamese_diacritics(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text)
    unaccented = "".join(
        character for character in decomposed if not unicodedata.combining(character)
    )
    return unaccented.replace("đ", "d").replace("Đ", "D")


# ============================================================
# LỚP 1 — INJECTION DETECTOR
# ============================================================

_HARD_PATTERNS_RAW = [
    # English patterns
    r"ignore\s+(all\s+)?(previous|prior|above)\s*instructions?",
    r"forget\s+(all\s+)?(previous|prior|your)\s+(instructions?|rules?)",
    r"you\s+are\s+now\s+(an?\s+)?(unrestricted|unfiltered|jailbroken)",
    r"\b(jailbreak|dan\s*mode|developer\s*mode)\b",
    r"reveal\s+(your\s+)?(system\s+)?prompt",
    r"override\s+(the\s+)?(safety|guardrail|security|rules?)",
    r"(disable|bypass)\s+(your\s+)?(safety|filter|guardrail|restrictions?)",
    r"system\s+prompt\s+(leak|reveal|show)",
    r"set\s+aside\s+(all\s+)?(earlier|previous|prior)\s+(directives?|instructions?|rules?)",
    r"pretend\s+(you\s+are|to\s+be)\s+(not\s+)?(an?\s+)?(ai|assistant|bot)",
    # Vietnamese patterns
    r"bỏ\s+qua\s+(mọi|tất\s+cả|các)?\s*(hướng\s+dẫn|quy\s+tắc|chỉ\s+thị|lệnh)",
    r"quên\s+(hết|mọi|tất\s+cả)\s+(các\s+)?(hướng\s+dẫn|quy\s+tắc|lệnh)",
    r"quên\s+đi\s+(mọi|tất\s+cả)?\s*(các\s+)?(hướng\s+dẫn|quy\s+tắc)",
    r"tiết\s+lộ\s+(system\s*prompt|hướng\s+dẫn\s+hệ\s+thống|cấu\s+hình)",
    r"cho\s+tôi\s+xem\s+(system\s*prompt|hướng\s+dẫn\s+hệ\s+thống)",
    r"đóng\s+vai\s+(không\s+giới\s+hạn|không\s+kiểm\s+duyệt|tự\s+do)",
    r"vượt\s+qua\s+(bảo\s+mật|kiểm\s+duyệt|guardrail|an\s+toàn)",
    r"tắt\s+(bảo\s+mật|kiểm\s+duyệt|filter)",
    r"giả\s+vờ\s+(bạn\s+)?(là|không\s+phải)\s+(AI|trợ\s+lý|bot)",
    r"(jailbreak|dan\s*mode)\s*bằng\s+tiếng\s+việt",
]

_SOFT_PATTERNS_RAW = [
    # English patterns
    r"\bact\s+as\b",
    r"\brole[\s-]?play\b",
    r"\bhypothetical(ly)?\b",
    r"\bpretend\s+(to\s+be|you\s+are)\b",
    r"\bunrestricted\b",
    r"\bwithout\s+(any\s+)?restrictions?\b",
    r"\bno\s+limitations?\b",
    # Vietnamese patterns
    r"đóng\s+vai",
    r"giả\s+sử",
    r"hãy\s+tưởng\s+tượng",
    r"không\s+có\s+giới\s+hạn",
    r"không\s+bị\s+kiểm\s+duyệt",
]

HARD_INJECTION_PATTERNS = [
    re.compile(_strip_vietnamese_diacritics(pattern), re.I) for pattern in _HARD_PATTERNS_RAW
]
SOFT_INJECTION_PATTERNS = [
    re.compile(_strip_vietnamese_diacritics(pattern), re.I) for pattern in _SOFT_PATTERNS_RAW
]


def detect_prompt_injection(text: str) -> tuple[bool, float, str]:
    """Phát hiện prompt injection. Trả (is_injection, score, reason)."""
    if not text:
        return False, 0.0, ""

    normalized = _strip_vietnamese_diacritics(normalize_text(text))

    for pattern in HARD_INJECTION_PATTERNS:
        if pattern.search(normalized):
            return True, 1.0, f"hard pattern matched: {pattern.pattern[:60]}"

    soft_hits = sum(1 for p in SOFT_INJECTION_PATTERNS if p.search(normalized))
    score = min(soft_hits * 0.2, 0.9)

    if score >= 0.6:
        return True, score, f"soft score={score:.2f}, hits={soft_hits}"

    return False, score, ""


# ============================================================
# LỚP 2 — PII MASKER (Presidio + regex fallback)
# ============================================================

# Map entity type → (tag, keyword)
ENTITY_MAP: dict[str, tuple[str, str]] = {
    "CREDIT_CARD": ("[CARD_REDACTED]", "credit_card"),
    "US_SSN": ("[SSN_REDACTED]", "ssn"),
    "EMAIL_ADDRESS": ("[EMAIL_REDACTED]", "email"),
    "IBAN_CODE": ("[IBAN_REDACTED]", "iban"),
    "VN_CCCD": ("[CCCD_REDACTED]", "cccd"),
}

# Regex cho OTP / CVV / SĐT VN / SSN fallback
OTP_PATTERN = re.compile(
    r"(?i)(?:mã\s+otp|mã\s+xác\s+thực|otp|one[\s-]?time(?:\s+pass(?:word|code)?)?)"
    r"\s*(?:là|is|[:#-])?\s*\d{4,8}"
)
CVV_PATTERN = re.compile(r"(?i)\b(?:cvv|cvc|security\s*code)\s*[:=#-]?\s*\d{3,4}\b")
CARD_FALLBACK = re.compile(r"\b(?:\d[ -]?){12,18}\d\b")
SSN_FALLBACK = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
VN_PHONE_PATTERN = re.compile(r"(?<!\d)(?:\+?84|0)(?:3|5|7|8|9)\d{8}(?!\d)")


def luhn_check(number: str) -> bool:
    """Kiểm tra số thẻ tín dụng bằng thuật toán Luhn."""
    digits = re.sub(r"\D", "", number)
    if not (13 <= len(digits) <= 19):
        return False
    total = 0
    for i, ch in enumerate(reversed(digits)):
        n = int(ch)
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


# --- Singleton AnalyzerEngine ---
_analyzer: AnalyzerEngine | None = None
_analyzer_failed: bool = False


def _get_analyzer():
    """Lazy-init Presidio. Trả None nếu không load được."""
    global _analyzer, _analyzer_failed
    if _analyzer is not None:
        return _analyzer
    if _analyzer_failed or not _PRESIDIO_AVAILABLE:
        return None
    try:
        configuration = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
        }
        provider = NlpEngineProvider(nlp_configuration=configuration)
        nlp_engine = provider.create_engine()
        analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["en"])

        # Custom recognizer: CCCD VN (12 số)
        vn_cccd = PatternRecognizer(
            supported_entity="VN_CCCD",
            patterns=[Pattern(name="cccd_12", regex=r"\b\d{12}\b", score=0.7)],
        )
        analyzer.registry.add_recognizer(vn_cccd)

        _analyzer = analyzer
        return _analyzer
    except Exception:
        _analyzer_failed = True
        return None


def _mask_with_presidio(text: str) -> tuple[str, list[str]]:
    """Presidio detect + mask. Chỉ mask entity nằm trong ENTITY_MAP."""
    analyzer = _get_analyzer()
    if analyzer is None:
        return text, []

    entities = ["CREDIT_CARD", "US_SSN", "EMAIL_ADDRESS", "IBAN_CODE", "VN_CCCD"]

    try:
        results = analyzer.analyze(
            text=text,
            entities=entities,
            language="en",
            score_threshold=0.6,
        )
    except Exception:
        return text, []

    if not results:
        return text, []

    findings: list[str] = []
    results = sorted(results, key=lambda r: r.start, reverse=True)

    masked = text
    for r in results:
        if r.entity_type not in ENTITY_MAP:
            continue
        matched = masked[r.start : r.end]

        # Credit card: chỉ mask nếu qua Luhn
        if r.entity_type == "CREDIT_CARD" and not luhn_check(matched):
            continue

        tag, keyword = ENTITY_MAP[r.entity_type]
        masked = masked[: r.start] + tag + masked[r.end :]
        findings.append(keyword)

    return masked, findings


def _mask_with_regex(text: str) -> tuple[str, list[str]]:
    """Fallback khi Presidio không khả dụng."""
    findings: list[str] = []
    masked = text

    def ssn_repl(m):
        findings.append("ssn")
        return "[SSN_REDACTED]"

    masked = SSN_FALLBACK.sub(ssn_repl, masked)

    def card_repl(m):
        if luhn_check(m.group(0)):
            findings.append("credit_card")
            return "[CARD_REDACTED]"
        return m.group(0)

    masked = CARD_FALLBACK.sub(card_repl, masked)

    return masked, findings


def mask_pii(text: str) -> tuple[str, list[str]]:
    """Mask PII: Presidio + regex fallback (SSN, OTP, CVV, VN phone)."""
    if not text:
        return "", []

    if _PRESIDIO_AVAILABLE:
        masked, findings = _mask_with_presidio(text)
    else:
        masked, findings = _mask_with_regex(text)

    # Fallback regex cho SSN — Presidio cần context nên không bắt khi đứng riêng
    if "[SSN_REDACTED]" not in masked:

        def ssn_repl(m):
            findings.append("ssn")
            return "[SSN_REDACTED]"

        masked = SSN_FALLBACK.sub(ssn_repl, masked)

    # OTP — Presidio không có recognizer cho cái này
    def otp_repl(m):
        findings.append("otp")
        return "[OTP_REDACTED]"

    masked = OTP_PATTERN.sub(otp_repl, masked)

    # CVV — phải có context "cvv/cvc/security code"
    def cvv_repl(m):
        findings.append("cvv")
        return "[CVV_REDACTED]"

    masked = CVV_PATTERN.sub(cvv_repl, masked)

    # Số điện thoại VN
    def vn_phone_repl(m):
        findings.append("vn_phone")
        return "[PHONE_REDACTED]"

    masked = VN_PHONE_PATTERN.sub(vn_phone_repl, masked)

    return masked, findings


# ============================================================
# LỚP 3 — COMPLIANCE CHECK
# ============================================================

_COMPLIANCE_RAW = [
    (
        r"\b(authorize|execute|perform|make|do)\s+(this|the|a)\s+(wire\s+)?transfer\b",
        "wire transfer",
    ),
    (r"\bwire\s+transfer\b", "wire transfer"),
    (r"\bbypass\s+(my|the|your)?\s*(bank\s+)?(authentication|auth|login|2fa|mfa)\b", "bypass auth"),
    (
        r"\b(modify|change|increase|inflate|edit)\s+(my|the|an?)?\s*(account\s+)?balance\b",
        "modify balance",
    ),
    (r"\bdisable\s+(2fa|mfa|two[\s-]?factor)\b", "disable 2FA"),
    (r"\b(launder|laundering|money\s+laundering)\b", "money laundering"),
    (r"\b(hack|steal|fraud)\b", "hack/fraud"),
]

COMPLIANCE_PATTERNS = [(re.compile(p, re.I), label) for p, label in _COMPLIANCE_RAW]


def check_financial_compliance(text: str) -> tuple[bool, float, str]:
    """Kiểm tra yêu cầu tài chính trái phép."""
    if not text:
        return False, 0.0, ""

    normalized = normalize_text(text)
    for pattern, label in COMPLIANCE_PATTERNS:
        m = pattern.search(normalized)
        if m:
            return True, 1.0, f"compliance violation: {label} ({m.group(0)[:40]})"

    return False, 0.0, ""


# ============================================================
# LỚP 5 — OUTPUT VALIDATOR
# ============================================================

_OUTPUT_LEAK_RAW = [
    r"system\s+prompt",
    r"my\s+instructions?\s+are",
    r"i\s+was\s+told\s+to",
]
_OUTPUT_RISKY_RAW = [
    r"\b(buy|sell|short|long)\s+(now|today|immediately)\b",
    r"\bguaranteed\s+(return|profit|gain)\b",
    r"\b100%\s+(safe|guaranteed|profit)\b",
]

OUTPUT_LEAK_PATTERNS = [re.compile(p, re.I) for p in _OUTPUT_LEAK_RAW]
OUTPUT_RISKY_PATTERNS = [re.compile(p, re.I) for p in _OUTPUT_RISKY_RAW]


# ============================================================
# DATA CLASS
# ============================================================


@dataclass
class GuardrailResult:
    allowed: bool
    layer: str = ""
    reason: str = ""
    risk_score: float = 0.0
    processed_text: str = ""
    findings: list[str] = field(default_factory=list)


# ============================================================
# PIPELINE
# ============================================================


def process_input(
    text: str,
    client: OpenAI | None = None,
) -> GuardrailResult:
    """normalize → injection → compliance → PII."""
    if not text or not text.strip():
        return GuardrailResult(allowed=False, layer="input", reason="empty input")

    normalized = normalize_text(text)

    is_inj, score, reason = detect_prompt_injection(normalized)
    if is_inj:
        return GuardrailResult(
            allowed=False,
            layer="injection",
            reason=reason,
            risk_score=score,
            processed_text=normalized,
            findings=["prompt_injection"],
        )

    # Semantic check (2nd layer) — only if client provided
    if client is not None:
        sem_inj, sem_score, sem_reason = detect_injection_semantic(client, normalized)
        if sem_inj:
            return GuardrailResult(
                allowed=False,
                layer="injection-semantic",
                reason=sem_reason,
                risk_score=sem_score,
                processed_text=normalized,
                findings=["prompt_injection_semantic"],
            )

    # NeMo Guardrails check (NVIDIA)
    nemo_blocked, nemo_reason = nemo_guard.check_with_nemo(normalized)
    if nemo_blocked:
        return GuardrailResult(
            allowed=False,
            layer="injection-nemo",
            reason=nemo_reason,
            risk_score=1.0,
            processed_text=normalized,
            findings=["prompt_injection_nemo"],
        )

    is_comp, comp_score, comp_reason = check_financial_compliance(normalized)
    if is_comp:
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
            processed_text=masked,
            findings=pii_findings,
        )

    return GuardrailResult(
        allowed=True,
        layer="pass",
        processed_text=normalized,
        findings=[],
    )


def check_output(text: str) -> GuardrailResult:
    """Kiểm tra output LLM."""
    if not text:
        return GuardrailResult(allowed=True, layer="output", processed_text="")

    findings: list[str] = []

    for p in OUTPUT_LEAK_PATTERNS:
        if p.search(text):
            findings.append("system_prompt_leak")
            break

    for p in OUTPUT_RISKY_PATTERNS:
        if p.search(text):
            findings.append("risky_investment_advice")
            break

    _, pii = mask_pii(text)
    if pii:
        findings.append("pii_leak")

    if findings:
        return GuardrailResult(
            allowed=False,
            layer="output",
            reason=", ".join(findings),
            risk_score=0.8,
            processed_text=text,
            findings=findings,
        )

    return GuardrailResult(allowed=True, layer="output", processed_text=text, findings=[])


def process_output(text: str) -> GuardrailResult:
    """Wrapper cho check_output."""
    return check_output(text)


def detect_injection_semantic(
    client: OpenAI,
    text: str,
    threshold: float = 0.75,
) -> tuple[bool, float, str]:
    """Detect injection using Groq prompt-guard-2-86m semantic classifier.

    Model returns a single float score (0.0-1.0) as string.
    Score >= threshold -> injection detected.

    Returns (is_injection, score, reason).
    """
    if not text or not text.strip():
        return False, 0.0, ""

    try:
        response = client.chat.completions.create(
            model=config.PROMPT_GUARD_MODEL,
            messages=[{"role": "user", "content": text[:2000]}],
            temperature=0.0,
            max_tokens=20,
        )
        content = (response.choices[0].message.content or "").strip()

        if not content:
            return False, 0.0, ""

        try:
            score = float(content)
        except ValueError:
            try:
                match = re.search(r"0?\.\d+|\d+\.?\d*", content)
                if not match:
                    return False, 0.0, ""
                score = float(match.group(0))
            except ValueError:
                return False, 0.0, ""

        score = max(0.0, min(1.0, score))
        is_inj = score >= threshold

        reason = f"semantic score={score:.3f}" if is_inj else ""
        return is_inj, score, reason

    except Exception:
        # Fail-open on API errors (rate limit, network) — regex layer still active.
        return False, 0.0, ""
