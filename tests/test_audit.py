"""Test tamper-proof audit hash chain."""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import audit
from config import LOG_DIR


def _clear_chain():
    path = os.path.join(LOG_DIR, "audit_chain.jsonl")
    if os.path.exists(path):
        os.remove(path)


def test_log_creates_chain_file():
    _clear_chain()
    audit.log_event("test", "test_layer", "test reason", 0.5)
    path = os.path.join(LOG_DIR, "audit_chain.jsonl")
    assert os.path.exists(path)


def test_chain_valid_after_multiple_entries():
    _clear_chain()
    audit.log_event("blocked", "injection", "test 1", 1.0)
    audit.log_event("allowed", "llm", "test 2", 0.0)
    audit.log_event("pii_redacted", "pii", "test 3", 0.4)

    is_valid, total, msg = audit.verify_chain()
    assert is_valid is True
    assert total == 3


def test_chain_detects_tamper():
    _clear_chain()
    audit.log_event("blocked", "injection", "original", 1.0)
    audit.log_event("allowed", "llm", "second", 0.0)

    # Sửa entry đầu tiên
    path = os.path.join(LOG_DIR, "audit_chain.jsonl")
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()

    entry = json.loads(lines[0])
    entry["reason"] = "TAMPERED"
    lines[0] = json.dumps(entry, ensure_ascii=False) + "\n"

    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    is_valid, total, msg = audit.verify_chain()
    assert is_valid is False


def test_chain_empty_is_valid():
    _clear_chain()
    is_valid, total, msg = audit.verify_chain()
    assert is_valid is True
    assert total == 0
