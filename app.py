"""Giao diện Streamlit cho FinGuard Agent."""

from __future__ import annotations

import os
import time
from typing import List

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

import audit
import config
import guardrail

load_dotenv()

FALLBACK_REPLY: str = (
    "Hiện không kết nối được dịch vụ mô hình. Vui lòng thử lại sau. "
    "Đây không phải tư vấn tài chính chuyên nghiệp."
)


def _init_state() -> None:
    """Khởi tạo session_state."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "attacks_blocked" not in st.session_state:
        st.session_state.attacks_blocked = 0
    if "pii_redacted" not in st.session_state:
        st.session_state.pii_redacted = 0
    if "request_times" not in st.session_state:
        st.session_state.request_times = []


def _rate_limit_ok() -> bool:
    """Giới hạn 10 tin nhắn mỗi phút theo phiên."""
    now = time.time()
    window = config.RATE_LIMIT_WINDOW_SEC
    times: List[float] = [
        t for t in st.session_state.request_times if now - t < window
    ]
    st.session_state.request_times = times
    return len(times) < config.RATE_LIMIT_MAX_REQUESTS


def _get_client() -> OpenAI | None:
    """Tạo client OpenAI trỏ tới NEBIUS_BASE_URL."""
    api_key = os.getenv("NEBIUS_API_KEY", "").strip()
    if not api_key:
        return None
    return OpenAI(
        api_key=api_key,
        base_url=config.NEBIUS_BASE_URL,
        timeout=config.REQUEST_TIMEOUT,
    )


def _call_llm(client: OpenAI, user_text: str) -> str:
    """Gọi LLM với tối đa MAX_RETRIES lần thử lại."""
    attempts = config.MAX_RETRIES + 1
    last_error: Exception | None = None
    for _ in range(attempts):
        try:
            history = [
                msg
                for msg in st.session_state.messages
                if msg.get("role") in {"user", "assistant"}
            ]
            if history and history[-1].get("role") == "user":
                history = history[:-1]
            response = client.chat.completions.create(
                model=config.MODEL_NAME,
                temperature=config.TEMPERATURE,
                max_tokens=config.MAX_TOKENS,
                messages=[
                    {"role": "system", "content": config.SYSTEM_PROMPT},
                    *history,
                    {"role": "user", "content": user_text},
                ],
            )
            content = response.choices[0].message.content
            return content or FALLBACK_REPLY
        except Exception as exc:  # noqa: BLE001 — fallback khi API lỗi
            last_error = exc
            time.sleep(0.4)
    _ = last_error
    return FALLBACK_REPLY


def main() -> None:
    """Khởi chạy giao diện chat có guardrail."""
    st.set_page_config(page_title="FinGuard Agent", page_icon="🛡️", layout="centered")
    _init_state()

    st.title("🛡️ FinGuard Agent")
    st.info(config.DISCLAIMER)

    with st.sidebar:
        st.subheader("Metrics")
        st.metric("Attacks Blocked", st.session_state.attacks_blocked)
        st.metric("PII Redacted", st.session_state.pii_redacted)
        if st.button("Xóa lịch sử"):
            st.session_state.messages = []
            st.session_state.request_times = []
            st.rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Nhập câu hỏi (không gửi thẻ, CVV, OTP)...")
    if not prompt:
        return

    if len(prompt) > config.MAX_INPUT_LENGTH:
        st.error(f"Tin nhắn vượt quá {config.MAX_INPUT_LENGTH} ký tự.")
        return

    if not _rate_limit_ok():
        st.error("Bạn đã gửi quá 10 tin trong 1 phút. Vui lòng chờ rồi thử lại.")
        return

    st.session_state.request_times.append(time.time())

    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    started = time.perf_counter()
    result = guardrail.process_input(prompt)

    if not result.allowed:
        st.session_state.attacks_blocked += 1
        audit.log_event(
            event_type="blocked",
            layer=result.layer,
            reason=result.reason,
            risk_score=result.risk_score,
            extra={"pii_count": 0},
        )
        error_text = f"⛔ Yêu cầu bị chặn ({result.layer}): {result.reason}"
        with st.chat_message("assistant"):
            st.error(error_text)
        st.session_state.messages.append({"role": "assistant", "content": error_text})
        return

    if result.findings:
        st.session_state.pii_redacted += len(result.findings)
        audit.log_event(
            event_type="pii_redacted",
            layer=result.layer,
            reason=result.reason,
            risk_score=result.risk_score,
            extra={"pii_count": len(result.findings)},
        )
        st.warning("Đã phát hiện và che thông tin nhạy cảm trước khi gửi tới mô hình.")

    client = _get_client()
    if client is None:
        raw_reply = (
            "Thiếu NEBIUS_API_KEY trong môi trường. Không thể gọi mô hình."
        )
        model_name = "none"
    else:
        raw_reply = _call_llm(client, result.processed_text)
        model_name = config.MODEL_NAME

    output = guardrail.process_output(raw_reply)
    latency_ms = int((time.perf_counter() - started) * 1000)
    audit.log_event(
        event_type="output_checked" if output.allowed else "output_blocked",
        layer=output.layer,
        reason=output.reason,
        risk_score=output.risk_score,
        extra={"model": model_name, "latency_ms": latency_ms, "pii_count": 0},
    )

    display = output.processed_text
    if not output.allowed:
        st.session_state.attacks_blocked += 1
        with st.chat_message("assistant"):
            st.error(display)
        st.session_state.messages.append({"role": "assistant", "content": display})
        return

    with st.chat_message("assistant"):
        st.markdown(display)
    st.session_state.messages.append({"role": "assistant", "content": display})


if __name__ == "__main__":
    main()
