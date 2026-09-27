"""Tamper-proof audit logging với hash chain.

Mỗi entry chứa hash của entry trước, tạo thành chuỗi liên kết.
Sửa bất kỳ entry nào → vỡ chain → phát hiện ngay.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from typing import Any

from config import LOG_DIR

CHAIN_FILE = "audit_chain.jsonl"
GENESIS_HASH = "0" * 64  # hash của entry đầu tiên


def _ensure_dir() -> None:
    os.makedirs(LOG_DIR, exist_ok=True)


def _compute_hash(entry_data: dict[str, Any], prev_hash: str) -> str:
    """Tính SHA256 của entry + prev_hash."""
    payload = json.dumps(entry_data, sort_keys=True, ensure_ascii=False) + prev_hash
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _get_last_hash() -> str:
    """Đọc hash của entry cuối cùng trong chain file."""
    path = os.path.join(LOG_DIR, CHAIN_FILE)
    if not os.path.exists(path):
        return GENESIS_HASH

    last_hash = GENESIS_HASH
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                last_hash = entry.get("hash", GENESIS_HASH)
            except json.JSONDecodeError:
                continue
    return last_hash


def log_event(
    event_type: str,
    layer: str,
    reason: str = "",
    risk_score: float = 0.0,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Ghi 1 entry vào hash chain.

    Trả về entry đã ghi (bao gồm hash).
    """
    _ensure_dir()

    safe_keys = {"pii_count", "model", "latency_ms"}
    extras = {k: v for k, v in (extra or {}).items() if k in safe_keys}

    data = {
        "timestamp": datetime.now(UTC).isoformat(),
        "event_type": event_type,
        "layer": layer,
        "reason": reason[:200],
        "risk_score": round(risk_score, 3),
        **extras,
    }

    prev_hash = _get_last_hash()
    entry_hash = _compute_hash(data, prev_hash)

    entry = {**data, "prev_hash": prev_hash, "hash": entry_hash}

    path = os.path.join(LOG_DIR, CHAIN_FILE)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return entry


def verify_chain() -> tuple[bool, int, str]:
    """Kiểm tra toàn vẹn của hash chain.

    Trả về (is_valid, total_entries, message).
    """
    path = os.path.join(LOG_DIR, CHAIN_FILE)
    if not os.path.exists(path):
        return True, 0, "Chain file does not exist"

    prev_hash = GENESIS_HASH
    total = 0

    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                return False, total, f"Line {line_no}: invalid JSON"

            stored_hash = entry.pop("hash", "")
            stored_prev = entry.pop("prev_hash", "")

            if stored_prev != prev_hash:
                return (
                    False,
                    total,
                    f"Line {line_no}: prev_hash mismatch",
                )

            expected_hash = _compute_hash(entry, prev_hash)
            if expected_hash != stored_hash:
                return (
                    False,
                    total,
                    f"Line {line_no}: hash mismatch (data tampered)",
                )

            prev_hash = stored_hash
            total += 1

    return True, total, "Chain is valid"
