"""NVIDIA NeMo Guardrails integration for FinGuard Agent."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import unicodedata
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nemoguardrails import LLMRails

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).resolve().parent / "nemo_config"
GUARDRAIL_LOG_PATH = Path(__file__).resolve().parent / "guardrail_log.jsonl"
ENV_PATH = Path(__file__).resolve().parent / ".env"

_rails: LLMRails | None = None
_init_failed = False
last_check_degraded: bool | None = None

_INJECTION_PATTERNS = (
    re.compile(r"\bignore\s+(?:all\s+)?(?:previous|prior)\s+instructions?\b"),
    re.compile(r"\bdisregard\s+(?:all\s+)?(?:previous|prior)\s+rules?\b"),
    re.compile(r"\breveal\s+(?:your\s+)?system\s+prompt\b"),
    re.compile(r"\bbo\s+qua\s+(?:(?:moi|tat\s+ca)\s+)?huong\s+dan\s+truoc\s+do\b"),
    re.compile(r"\btiet\s+lo\s+(?:your\s+)?system\s+prompt\b"),
)


def _load_env() -> None:
    """Load .env from the project root before any env var reads."""
    try:
        from dotenv import load_dotenv

        load_dotenv(dotenv_path=ENV_PATH, override=True)
    except ImportError:
        logger.warning(
            "python-dotenv is not installed; environment variables must be set "
            "manually. Run: pip install python-dotenv"
        )
    except OSError:
        logger.warning("Unable to read .env at %s.", ENV_PATH, exc_info=True)

    if not os.environ.get("GROQ_API_KEY"):
        logger.warning("GROQ_API_KEY not found in environment; NeMo classifier will fail open.")


# Load .env immediately at module import time so any module-level env reads
# in nemo_config.actions pick up the correct values.
_load_env()


def _log_blocked_input(text: str, reason: str) -> None:
    """Append a privacy-safe record for a blocked input."""
    try:
        entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "input_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "reason": reason,
            "blocked": True,
        }
        with GUARDRAIL_LOG_PATH.open("a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except (OSError, UnicodeError):
        logger.warning("Unable to append to guardrail audit log.", exc_info=True)


def _normalize_for_injection_check(text: str) -> str:
    """Normalize accents and whitespace for deterministic injection matching."""
    decomposed = unicodedata.normalize("NFD", text.casefold())
    unaccented = "".join(
        character for character in decomposed if not unicodedata.combining(character)
    )
    return re.sub(r"\s+", " ", unaccented.replace("đ", "d")).strip()


def _get_rails() -> LLMRails | None:
    """Lazy init NeMo Guardrails.

    Order matters: load_dotenv must run before importing nemo_config.actions
    so that any module-level os.environ reads in that module see the real key.
    """
    global _rails, _init_failed

    if _rails is not None:
        return _rails
    if _init_failed:
        return None

    try:
        # Re-affirm env load here in case the module was imported from a
        # context that already consumed an older environment.
        _load_env()

        # Import AFTER env is loaded.
        from nemoguardrails import LLMRails, RailsConfig

        from nemo_config.actions import classify_injection_llm

        config = RailsConfig.from_path(str(CONFIG_PATH))
        _rails = LLMRails(config)
        _rails.register_action(classify_injection_llm, name="classify_injection_llm")
        return _rails
    except Exception:
        _init_failed = True
        print("WARNING: NeMo Guardrails initialization failed; using fail-open behavior.")
        logger.exception("NeMo Guardrails initialization failed; failing open.")
        return None


def check_with_nemo(text: str) -> tuple[bool, str]:
    """Check input with NeMo Guardrails.

    Returns (is_blocked, reason).
    Fail-open if NeMo is unavailable.
    """
    global last_check_degraded

    if not text or not text.strip():
        return False, ""

    # Layer 1: deterministic regex fast-path (EN + VI, accent-insensitive).
    normalized = _normalize_for_injection_check(text)
    if any(pattern.search(normalized) for pattern in _INJECTION_PATTERNS):
        reason = "NeMo Guardrails: injection detected"
        _log_blocked_input(text, reason)
        return True, reason

    # Layer 2: LLM rail via NeMo + Groq.
    rails = _get_rails()
    if rails is None:
        last_check_degraded = True
        return False, ""

    try:
        result = rails.generate(messages=[{"role": "user", "content": text}])
        from nemo_config import actions

        last_check_degraded = actions.last_classification_degraded
        response = result.get("content", "") if isinstance(result, dict) else str(result)

        if "can't help with that request" in response.lower():
            reason = "blocked by NeMo LLM rail"
            _log_blocked_input(text, reason)
            return True, reason

        return False, ""
    except Exception:
        last_check_degraded = True
        print("WARNING: NeMo Guardrails input check failed; using fail-open behavior.")
        logger.exception("NeMo Guardrails input check failed; failing open.")
        return False, ""


if __name__ == "__main__":
    print("nemo_guard module loaded.")
    print("Config path:", CONFIG_PATH)
    print("Env path:", ENV_PATH)
    if not CONFIG_PATH.is_dir():
        raise SystemExit(f"NeMo Guardrails config directory not found: {CONFIG_PATH}")

    if not os.environ.get("GROQ_API_KEY"):
        raise SystemExit(
            "GROQ_API_KEY is not set. Add it to D:\\finaigent\\.env "
            "or export it before running this script."
        )

    if _get_rails() is None:
        raise SystemExit("NeMo Guardrails config failed to load; see the logged error above.")

    print("NeMo Guardrails config loaded successfully.")
