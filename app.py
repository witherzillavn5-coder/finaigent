"""Giao diện Streamlit cho FinGuard Agent."""

from __future__ import annotations

import os
import time
from datetime import datetime  # mốc giờ cho stats_history
from pathlib import Path  # đường dẫn file audit log
from typing import List

import pandas as pd  # DataFrame cho line_chart sidebar
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

# Số tin nhắn gần nhất truyền vào LLM (3 cặp user-assistant)
HISTORY_LIMIT: int = 6


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
    if "stats_history" not in st.session_state:
        # Lịch sử điểm dữ liệu cho 2 biểu đồ sidebar
        st.session_state.stats_history = []
    if "blocked_attacks" not in st.session_state:
        st.session_state.blocked_attacks = 0
    # Cờ báo cần rerun sau khi mask PII
    if "_should_rerun" not in st.session_state:
        st.session_state._should_rerun = False


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
    """Gọi LLM với tối đa MAX_RETRIES lần thử lại.

    Multi-turn: chỉ truyền HISTORY_LIMIT tin nhắn gần nhất để tránh
    vượt context window và tiết kiệm token.
    """
    attempts = config.MAX_RETRIES + 1
    last_error: Exception | None = None

    for _ in range(attempts):
        try:
            # Lấy 6 tin nhắn gần nhất (3 cặp user-assistant)
            recent_messages = st.session_state.messages[-HISTORY_LIMIT:]
            history = [
                msg
                for msg in recent_messages
                if msg.get("role") in {"user", "assistant"}
            ]
            # Bỏ tin user cuối vì sẽ thêm lại bên dưới
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


def _record_stats() -> None:
    """Ghi mốc blocked/PII theo thời gian cho biểu đồ sidebar."""
    st.session_state.blocked_attacks = int(st.session_state.attacks_blocked)
    st.session_state.stats_history.append({
        "time": datetime.now().strftime("%H:%M:%S"),
        "blocked": st.session_state.blocked_attacks,
        "pii": st.session_state.pii_redacted,
    })


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

        st.divider()

        # Số cuộc tấn công bị chặn theo thời gian
        st.subheader("📊 Attacks over time")
        if st.session_state.stats_history:
            df = pd.DataFrame(st.session_state.stats_history)
            st.line_chart(df, x="time", y="blocked", height=150)
        else:
            st.caption("Chưa có dữ liệu")

        # Số PII đã che theo thời gian
        st.subheader("📊 PII over time")
        if st.session_state.stats_history:
            df = pd.DataFrame(st.session_state.stats_history)
            st.line_chart(df, x="time", y="pii", height=150)
        else:
            st.caption("Chưa có dữ liệu")

        # Tải file nhật ký kiểm toán JSONL
        st.divider()
        audit_path = Path("logs/audit.jsonl")
        if audit_path.exists() and audit_path.stat().st_size > 0:
            with open(audit_path, "r", encoding="utf-8") as f:
                audit_content = f.read()
            st.download_button(
                label="📥 Tải Audit Log",
                data=audit_content,
                file_name="audit.jsonl",
                mime="application/json",
                use_container_width=True,
            )
            st.caption(f"Log có {len(audit_content.splitlines())} dòng")
        else:
            st.caption("📭 Chưa có log")

        if st.button("Xóa lịch sử"):
            st.session_state.messages = []
            st.session_state.request_times = []
            st.rerun()

    # Hiển thị lịch sử chat
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

    # ============================================================
    # LỚP 1+3 — Input bị chặn (injection hoặc compliance)
    # ============================================================
    if not result.allowed:
        st.session_state.attacks_blocked += 1
        st.session_state.blocked_attacks += 1
        audit.log_event(
            event_type="blocked",
            layer=result.layer,
            reason=result.reason,
            risk_score=result.risk_score,
            extra={"pii_count": 0},
        )
        _record_stats()
        _LABELS = {
            "injection": "Phát hiện dấu hiệu tấn công.",
            "compliance": "Yêu cầu không được phép.",
            "output": "Phản hồi không an toàn.",
        }
        label = _LABELS.get(result.layer, "Yêu cầu bị chặn.")
        error_text = f"⛔ {label}\n\nVui lòng nhập lại câu hỏi tài chính bình thường."
        with st.chat_message("assistant"):
            st.error(error_text)
        st.session_state.messages.append({"role": "assistant", "content": error_text})
        st.rerun()

    # ============================================================
    # LỚP 2 — PII masking
    # ============================================================
    if result.findings:
        st.session_state.pii_redacted += len(result.findings)
        audit.log_event(
            event_type="pii_redacted",
            layer=result.layer,
            reason=result.reason,
            risk_score=result.risk_score,
            extra={"pii_count": len(result.findings)},
        )
        _record_stats()
        st.warning("Đã phát hiện và che thông tin nhạy cảm trước khi gửi tới mô hình.")
        # Đánh dấu để rerun sau khi hiển thị xong
        st.session_state._should_rerun = True

    # ============================================================
    # LỚP 4 — Gọi LLM
    # ============================================================
    client = _get_client()
    if client is None:
        raw_reply = "Thiếu NEBIUS_API_KEY trong môi trường. Không thể gọi mô hình."
        model_name = "none"
    else:
        raw_reply = _call_llm(client, result.processed_text)
        model_name = config.MODEL_NAME

    # ============================================================
    # LỚP 5 — Kiểm tra output LLM
    # ============================================================
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
        st.session_state.blocked_attacks += 1
        _record_stats()
        with st.chat_message("assistant"):
            st.error(display)
        st.session_state.messages.append({"role": "assistant", "content": display})
        st.rerun()

    with st.chat_message("assistant"):
        st.markdown(display)
    st.session_state.messages.append({"role": "assistant", "content": display})

    # Nếu có PII vừa được mask, rerun để sidebar cập nhật ngay
    if st.session_state.pop("_should_rerun", False):
        st.rerun()


if __name__ == "__main__":
    main()