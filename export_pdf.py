"""Export conversation to PDF for FinGuard Agent."""

from __future__ import annotations

import io
from datetime import datetime
from typing import Any

from fpdf import FPDF


class ConversationPDF(FPDF):
    """Custom PDF class with header and footer."""

    def header(self) -> None:
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(30, 64, 175)
        self.cell(0, 10, "FinGuard Agent - Conversation Export", align="C")
        self.ln(8)

        self.set_font("Helvetica", "", 9)
        self.set_text_color(120, 120, 120)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.cell(0, 5, f"Generated: {timestamp}", align="C")
        self.ln(10)

    def footer(self) -> None:
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")


def _sanitize_pdf_text(text: str) -> str:
    """Remove non-latin characters that break default fonts."""
    if not text:
        return ""

    replacements = {
        "->": "->",
        "<-": "<-",
        "[OK]": "[OK]",
        "[FAIL]": "[FAIL]",
        "[BLOCKED]": "[BLOCKED]",
        "[WARN]": "[WARN]",
        "[ALERT]": "[ALERT]",
        "[TOOL]": "[TOOL]",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    try:
        text.encode("latin-1")
        return text
    except UnicodeEncodeError:
        return text.encode("latin-1", errors="replace").decode("latin-1")


def build_conversation_pdf(
    messages: list[dict[str, Any]],
    title: str = "FinGuard Agent Conversation",
) -> bytes:
    """Build a PDF from conversation messages. Returns bytes."""
    pdf = ConversationPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(30, 30, 30)
    pdf.multi_cell(0, 6, _sanitize_pdf_text(title))
    pdf.ln(4)

    pdf.set_draw_color(200, 200, 200)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(6)

    for i, message in enumerate(messages, start=1):
        role = message.get("role", "unknown")

        if role == "user":
            pdf.set_text_color(30, 64, 175)
        else:
            pdf.set_text_color(16, 120, 80)

        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 6, f"Message {i} - {_sanitize_pdf_text(role.upper())}")
        pdf.ln(6)

        tools_used = message.get("tools_used") or []
        if tools_used:
            pdf.set_text_color(100, 100, 100)
            pdf.set_font("Helvetica", "I", 9)
            tools_str = ", ".join(sorted(set(tools_used)))
            pdf.multi_cell(0, 5, f"Tools: {_sanitize_pdf_text(tools_str)}")
            pdf.ln(1)

        docs_used = message.get("doc_names") or []
        if docs_used:
            pdf.set_text_color(100, 100, 100)
            pdf.set_font("Helvetica", "I", 9)
            doc_names = ", ".join(docs_used)
            pdf.multi_cell(0, 5, f"Documents: {_sanitize_pdf_text(doc_names)}")
            pdf.ln(1)

        pdf.set_text_color(30, 30, 30)
        pdf.set_font("Helvetica", "", 10)
        content = _sanitize_pdf_text(message.get("content", ""))
        pdf.multi_cell(0, 5, content)
        pdf.ln(6)

    output = io.BytesIO()
    pdf.output(output)
    return output.getvalue()


if __name__ == "__main__":
    test_messages = [
        {"role": "user", "content": "Vay 500 trieu trong 5 nam lai 8%?"},
        {
            "role": "assistant",
            "content": "Ban can tra 10.14 trieu/thang",
            "tools_used": ["calculate_loan_payment"],
        },
    ]
    pdf_bytes = build_conversation_pdf(test_messages)
    print(f"PDF built: {len(pdf_bytes)} bytes")
    with open("test_conversation.pdf", "wb") as f:
        f.write(pdf_bytes)
    print("Saved to test_conversation.pdf")
