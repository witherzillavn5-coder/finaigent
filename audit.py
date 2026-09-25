"""Ghi nhật ký kiểm toán dạng JSONL (không lưu nội dung gốc)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

ALLOWED_EXTRA_KEYS: frozenset[str] = frozenset({"pii_count", "model", "latency_ms"})
LOG_PATH: Path = Path(__file__).resolve().parent / "logs" / "audit.jsonl"


def log_event(
    event_type: str,
    layer: str,
    reason: str,
    risk_score: float,
    extra: Optional[Mapping[str, Any]] = None,
) -> None:
    """Ghi một sự kiện bảo mật vào logs/audit.jsonl.

    Không ghi nội dung gốc người dùng. Chỉ nhận extra keys:
    pii_count, model, latency_ms.
    """
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    safe_extra: Dict[str, Any] = {}
    if extra:
        for key, value in extra.items():
            if key in ALLOWED_EXTRA_KEYS:
                safe_extra[key] = value

    record: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_type": event_type,
        "layer": layer,
        "reason": reason,
        "risk_score": risk_score,
        **safe_extra,
    }
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
