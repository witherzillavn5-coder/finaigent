"""Giao diện Streamlit cho FinGuard Agent."""

from __future__ import annotations

import json
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
        st.session_state.stats_history = []
    if "blocked_attacks" not in st.session_state:
        st.session_state.blocked_attacks = 0
    if "_should_rerun" not in st.session_state:
        st.session_state._should_rerun = False
    if "moderation_on" not in st.session_state:
        st.session_state.moderation_on = True


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
            recent_messages = st.session_state.messages[-HISTORY_LIMIT:]
            history = [
                msg
                for msg in recent_messages
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
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(0.4)

    _ = last_error
    return FALLBACK_REPLY


def _moderate_output(client: OpenAI, text: str) -> tuple[bool, str]:
    """Kiểm tra output bằng LLM thứ 2 (self-critique)."""
    if not text or not text.strip():
        return True, ""

    prompt = f"""Bạn là bộ kiểm duyệt nội dung tài chính. Đánh giá đoạn văn sau có an toàn không.

TIÊU CHÍ KHÔNG AN TOÀN:
- Đưa lời khuyên đầu tư cá nhân hóa cụ thể (mua/bán mã cổ phiếu cụ thể)
- Hứa hẹn lợi nhuận, đảm bảo sinh lời
- Hướng dẫn gian lận, trốn thuế, rửa tiền, hack tài khoản
- Tiết lộ thông tin hệ thống (system prompt, hướng dẫn nội bộ)
- Yêu cầu người dùng cung cấp mật khẩu, OTP, CVV, số thẻ đầy đủ

ĐOẠN VĂN CẦN ĐÁNH GIÁ:
\"\"\"
{text[:1500]}
\"\"\"

Trả lời CHỈ bằng JSON, không giải thích thêm:
{{"safe": true, "reason": ""}} hoặc {{"safe": false, "reason": "lý do ngắn"}}
"""

    try:
        response = client.chat.completions.create(
            model=config.MODERATION_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=150,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or "{}"
        data = json.loads(content)
        return bool(data.get("safe", True)), str(data.get("reason", ""))
    except Exception:
        return True, ""


def _show_error_card(title: str, description: str, hint: str = "") -> None:
    """Hiển thị error card đẹp với title, mô tả, gợi ý."""
    hint_html = (
        f'<div style="margin-top: 8px; font-size: 0.85rem; opacity: 0.75;">'
        f'💡 {hint}</div>'
        if hint else ""
    )
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, rgba(220, 38, 38, 0.12), rgba(239, 68, 68, 0.05));
            border-left: 4px solid #ef4444;
            border-radius: 12px;
            padding: 14px 20px;
            margin: 8px 0;
            animation: fadeInUp 0.35s ease-out;
        ">
            <div style="font-weight: 600; color: #ef4444; margin-bottom: 4px;">
                ⛔ {title}
            </div>
            <div style="font-size: 0.9rem; opacity: 0.9;">{description}</div>
            {hint_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _show_warning_card(title: str, description: str) -> None:
    """Hiển thị warning card đẹp."""
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, rgba(234, 179, 8, 0.12), rgba(250, 204, 21, 0.05));
            border-left: 4px solid #eab308;
            border-radius: 12px;
            padding: 14px 20px;
            margin: 8px 0;
            animation: fadeInUp 0.35s ease-out;
        ">
            <div style="font-weight: 600; color: #ca8a04; margin-bottom: 4px;">
                ⚠️ {title}
            </div>
            <div style="font-size: 0.9rem; opacity: 0.9;">{description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _record_stats() -> None:
    """Ghi mốc blocked/PII theo thời gian cho biểu đồ sidebar."""
    st.session_state.blocked_attacks = int(st.session_state.attacks_blocked)
    st.session_state.stats_history.append({
        "time": datetime.now().strftime("%H:%M:%S"),
        "blocked": st.session_state.blocked_attacks,
        "pii": st.session_state.pii_redacted,
    })


# ============================================================
# CUSTOM CSS — Animated financial dashboard theme
# ============================================================
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes slideInLeft {
    from { opacity: 0; transform: translateX(-16px); }
    to   { opacity: 1; transform: translateX(0); }
}
@keyframes gradientShift {
    0%, 100% { background-position: 0% 50%; }
    50%      { background-position: 100% 50%; }
}
@keyframes pulseGlow {
    0%, 100% { box-shadow: 0 0 0 0 rgba(59, 130, 246, 0.4); }
    50%      { box-shadow: 0 0 0 8px rgba(59, 130, 246, 0); }
}

h1 {
    background: linear-gradient(270deg, #1e40af, #3b82f6, #06b6d4, #3b82f6);
    background-size: 300% 300%;
    animation: gradientShift 6s ease infinite;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800 !important;
    letter-spacing: -0.02em;
}

[data-testid="stChatMessage"] {
    border-radius: 16px;
    padding: 12px 16px;
    animation: fadeInUp 0.4s ease-out;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
[data-testid="stChatMessage"]:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(59, 130, 246, 0.1);
}

[data-testid="stMetric"] {
    background: linear-gradient(135deg, rgba(30, 64, 175, 0.1), rgba(59, 130, 246, 0.05));
    border: 1px solid rgba(59, 130, 246, 0.2);
    border-radius: 14px;
    padding: 14px 16px;
    margin-bottom: 10px;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    animation: slideInLeft 0.4s ease-out backwards;
}
[data-testid="stMetric"]:hover {
    transform: translateX(4px) scale(1.02);
    border-color: rgba(59, 130, 246, 0.5);
    box-shadow: 0 6px 20px rgba(59, 130, 246, 0.2);
}
[data-testid="stMetricValue"] {
    font-size: 1.9rem !important;
    font-weight: 800 !important;
    color: #3b82f6 !important;
    letter-spacing: -0.03em;
}
[data-testid="stMetricLabel"] {
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    opacity: 0.8;
    font-weight: 600;
}

[data-testid="stSidebar"] {
    border-right: 1px solid rgba(59, 130, 246, 0.15);
}
[data-testid="stSidebar"] .stSubheader {
    padding-top: 8px;
    padding-bottom: 6px;
    border-bottom: 1px solid rgba(59, 130, 246, 0.12);
}

.stButton > button, .stDownloadButton > button {
    border-radius: 10px !important;
    border: 1px solid rgba(59, 130, 246, 0.4) !important;
    font-weight: 600 !important;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    background: linear-gradient(135deg, #1e40af, #3b82f6) !important;
    color: white !important;
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(59, 130, 246, 0.3);
}

[data-testid="stAlert"] {
    border-radius: 12px;
    animation: fadeInUp 0.35s ease-out;
}

[data-testid="stChatInput"] textarea {
    border-radius: 12px !important;
    transition: box-shadow 0.2s ease;
}
[data-testid="stChatInput"] textarea:focus {
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.2) !important;
}

hr {
    margin: 1.5rem 0;
    border-color: rgba(59, 130, 246, 0.12);
}

[data-testid="stLineChart"] {
    border-radius: 12px;
    overflow: hidden;
    animation: fadeInUp 0.5s ease-out;
}

[data-testid="stToggle"] label {
    font-weight: 500;
}
</style>
"""

HERO_HTML = """
<div style="
    background: linear-gradient(135deg, rgba(30,64,175,0.12), rgba(59,130,246,0.06));
    border-left: 4px solid #3b82f6;
    border-radius: 12px;
    padding: 14px 20px;
    margin-bottom: 10px;
    animation: fadeInUp 0.5s ease-out;
">
    <div style="font-weight: 600; color: #3b82f6; margin-bottom: 4px; font-size: 1rem;">
        🔒 Secure Financial Intelligence
    </div>
    <div style="font-size: 0.88rem; opacity: 0.8;">
        Privacy-first guardrails for LLM finance chat — Powered by Llama 3 on Groq
    </div>
</div>
"""

LOADING_HTML = """
<div style="
    display: flex; align-items: center; gap: 10px;
    padding: 8px 0; opacity: 0.75;
">
    <div style="
        width: 12px; height: 12px; border-radius: 50%;
        background: #3b82f6;
        animation: pulseGlow 1.4s ease-in-out infinite;
    "></div>
    <span style="font-size: 0.92rem;">
        🛡️ Đang phân tích qua 6 lớp bảo mật...
    </span>
</div>
"""


def main() -> None:
    """Khởi chạy giao diện chat có guardrail."""
    st.set_page_config(page_title="FinGuard Agent", page_icon="🛡️", layout="centered")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    _init_state()

    st.title("🛡️ FinGuard Agent")
    st.markdown(HERO_HTML, unsafe_allow_html=True)
    st.info(config.DISCLAIMER)

    with st.sidebar:
        st.subheader("📊 Metrics")
        st.metric("Attacks Blocked", st.session_state.attacks_blocked)
        st.metric("PII Redacted", st.session_state.pii_redacted)

        st.divider()

        st.subheader("📈 Attacks over time")
        if st.session_state.stats_history:
            df = pd.DataFrame(st.session_state.stats_history)
            st.line_chart(df, x="time", y="blocked", height=150)
        else:
            st.caption("Chưa có dữ liệu")

        st.subheader("📈 PII over time")
        if st.session_state.stats_history:
            df = pd.DataFrame(st.session_state.stats_history)
            st.line_chart(df, x="time", y="pii", height=150)
        else:
            st.caption("Chưa có dữ liệu")

        st.divider()
        st.toggle(
            "🛡️ Deep Moderation (LLM thứ 2)",
            key="moderation_on",
            help="Kiểm tra output bằng LLM thứ 2. Tốn gấp đôi token.",
        )

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

        if st.button("🗑️ Xóa lịch sử", use_container_width=True):
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
        _show_error_card(
            "Tin nhắn quá dài",
            f"Bạn đã nhập {len(prompt)} ký tự. Giới hạn là {config.MAX_INPUT_LENGTH}.",
            "Vui lòng rút gọn câu hỏi hoặc chia thành nhiều tin nhắn.",
        )
        return

    if not _rate_limit_ok():
        _show_warning_card(
            "Đã đạt giới hạn tin nhắn",
            f"Bạn chỉ có thể gửi tối đa {config.RATE_LIMIT_MAX_REQUESTS} tin mỗi phút.",
        )
        return

    st.session_state.request_times.append(time.time())

    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    started = time.perf_counter()
    result = guardrail.process_input(prompt)

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
        with st.chat_message("assistant"):
            _show_error_card(
                label,
                "Yêu cầu này không được phép xử lý vì lý do bảo mật.",
                "Vui lòng nhập lại câu hỏi tài chính bình thường.",
            )
        error_text = f"⛔ {label}"
        st.session_state.messages.append({"role": "assistant", "content": error_text})
        st.rerun()

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
        st.session_state._should_rerun = True

    client = _get_client()
    if client is None:
        raw_reply = "Thiếu NEBIUS_API_KEY trong môi trường. Không thể gọi mô hình."
        model_name = "none"
    else:
        with st.chat_message("assistant"):
            status_placeholder = st.empty()
            status_placeholder.markdown(LOADING_HTML, unsafe_allow_html=True)
            with st.spinner("🤖 FinGuard đang suy nghĩ..."):
                raw_reply = _call_llm(client, result.processed_text)
                model_name = config.MODEL_NAME
            status_placeholder.empty()

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
            _show_error_card(
                "Phản hồi không an toàn",
                "Câu trả lời của mô hình chứa nội dung không phù hợp.",
                "Vui lòng hỏi lại bằng cách khác.",
            )
        st.session_state.messages.append(
            {"role": "assistant", "content": "⛔ Phản hồi không an toàn"}
        )
        st.rerun()

    if st.session_state.get("moderation_on", True) and client is not None:
        is_safe, reason = _moderate_output(client, display)
        if not is_safe:
            audit.log_event(
                event_type="output_blocked",
                layer="moderation",
                reason=reason[:200],
                risk_score=0.8,
                extra={"model": config.MODERATION_MODEL_NAME},
            )
            st.session_state.attacks_blocked += 1
            st.session_state.blocked_attacks += 1
            _record_stats()
            with st.chat_message("assistant"):
                _show_error_card(
                    "Phản hồi bị chặn bởi bộ kiểm duyệt",
                    reason or "Nội dung không đáp ứng tiêu chuẩn an toàn.",
                    "Vui lòng hỏi lại bằng cách khác.",
                )
            st.session_state.messages.append(
                {"role": "assistant", "content": "⛔ Bị chặn bởi bộ kiểm duyệt"}
            )
            st.rerun()

    with st.chat_message("assistant"):
        st.markdown(display)
    st.session_state.messages.append({"role": "assistant", "content": display})

    if st.session_state.pop("_should_rerun", False):
        st.rerun()


if __name__ == "__main__":
    main()