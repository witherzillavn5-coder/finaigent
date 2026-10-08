"""Tests for output moderation and degraded-service handling."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
from openai import BadRequestError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app


class _Message:
    def __init__(self, content: str) -> None:
        self.content = content


class _Choice:
    def __init__(self, content: str) -> None:
        self.message = _Message(content)


class _Response:
    def __init__(self, content: str) -> None:
        self.choices = [_Choice(content)]


class _Completions:
    def __init__(self, result: str | Exception) -> None:
        self.result = result
        self.kwargs: dict[str, object] = {}

    def create(self, **kwargs):
        self.kwargs = kwargs
        if isinstance(self.result, Exception):
            raise self.result
        return _Response(self.result)


class _Client:
    def __init__(self, result: str | Exception) -> None:
        self.completions = _Completions(result)
        self.chat = type("Chat", (), {"completions": self.completions})()


def _bad_request_error() -> BadRequestError:
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(400, request=request)
    return BadRequestError(
        "Failed to validate JSON",
        response=response,
        body={"error": {"message": "Failed to validate JSON"}},
    )


def _patch_moderation_client(monkeypatch, result: str | Exception) -> _Client:
    client = _Client(result)
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("GROQ_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setattr(app, "OpenAI", lambda **_kwargs: client)
    monkeypatch.setattr(app.st, "session_state", SimpleNamespace())
    return client


def test_moderation_failure_fails_open_by_default(monkeypatch, caplog) -> None:
    _patch_moderation_client(monkeypatch, RuntimeError("moderation backend is unavailable"))

    with caplog.at_level("WARNING"):
        safe, reason = app._moderate_output("Example response")

    assert safe is True
    assert reason == ""
    assert app.st.session_state.moderation_degraded is True
    assert "Output moderation failed" in caplog.text


def test_moderation_infra_error_does_not_block(monkeypatch, caplog) -> None:
    client = _patch_moderation_client(monkeypatch, _bad_request_error())

    with caplog.at_level("WARNING"):
        safe, reason = app._moderate_output("Example response")

    assert safe is True
    assert reason == ""
    assert app.st.session_state.moderation_degraded is True
    assert "Output moderation failed" in caplog.text
    assert "response_format" not in client.completions.kwargs
    assert client.completions.kwargs["messages"][0]["content"].endswith(
        '"SAFE:" or "UNSAFE:" followed by a one-sentence reason.\n'
    )


def test_moderation_unsafe_blocks(monkeypatch) -> None:
    _patch_moderation_client(monkeypatch, "UNSAFE: gambling")

    safe, reason = app._moderate_output("Example response")

    assert safe is False
    assert reason == "gambling"
    assert app.st.session_state.moderation_degraded is False


def test_moderation_safe_allows(monkeypatch) -> None:
    _patch_moderation_client(monkeypatch, "SAFE: general financial education")

    safe, reason = app._moderate_output("Example response")

    assert safe is True
    assert reason == "general financial education"
    assert app.st.session_state.moderation_degraded is False
