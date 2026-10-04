"""Parse uploaded documents (PDF, TXT) for FinGuard Agent.

Extract text, sanitize, and return metadata.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Any

try:
    from pypdf import PdfReader

    _PYPDF_AVAILABLE = True
except ImportError:
    _PYPDF_AVAILABLE = False


@dataclass
class ParsedDocument:
    """Result of parsing an uploaded file."""

    filename: str
    file_type: str
    text: str
    page_count: int
    char_count: int
    success: bool
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


MAX_PDF_PAGES = 20
MAX_TEXT_CHARS = 50_000


def parse_pdf(file_bytes: bytes, filename: str) -> ParsedDocument:
    """Extract text from PDF bytes."""
    if not _PYPDF_AVAILABLE:
        return ParsedDocument(
            filename=filename,
            file_type="pdf",
            text="",
            page_count=0,
            char_count=0,
            success=False,
            error="pypdf not installed. Run: pip install pypdf",
        )

    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        page_count = len(reader.pages)

        if page_count > MAX_PDF_PAGES:
            return ParsedDocument(
                filename=filename,
                file_type="pdf",
                text="",
                page_count=page_count,
                char_count=0,
                success=False,
                error=f"PDF has {page_count} pages. Maximum {MAX_PDF_PAGES}.",
            )

        text_parts: list[str] = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text_parts.append(page_text)

        full_text = "\n\n".join(text_parts).strip()

        if len(full_text) > MAX_TEXT_CHARS:
            full_text = full_text[:MAX_TEXT_CHARS] + "\n\n[... truncated ...]"

        if not full_text:
            return ParsedDocument(
                filename=filename,
                file_type="pdf",
                text="",
                page_count=page_count,
                char_count=0,
                success=False,
                error="No text found. May be a scanned/image PDF.",
            )

        return ParsedDocument(
            filename=filename,
            file_type="pdf",
            text=full_text,
            page_count=page_count,
            char_count=len(full_text),
            success=True,
        )

    except Exception as exc:
        return ParsedDocument(
            filename=filename,
            file_type="pdf",
            text="",
            page_count=0,
            char_count=0,
            success=False,
            error=f"Failed to parse PDF: {exc}",
        )


def parse_text(file_bytes: bytes, filename: str) -> ParsedDocument:
    """Parse plain text files."""
    try:
        text = file_bytes.decode("utf-8", errors="replace")
        if len(text) > MAX_TEXT_CHARS:
            text = text[:MAX_TEXT_CHARS] + "\n\n[... truncated ...]"
        return ParsedDocument(
            filename=filename,
            file_type="txt",
            text=text,
            page_count=1,
            char_count=len(text),
            success=True,
        )
    except Exception as exc:
        return ParsedDocument(
            filename=filename,
            file_type="txt",
            text="",
            page_count=0,
            char_count=0,
            success=False,
            error=f"Failed to parse text: {exc}",
        )


def parse_document(file_bytes: bytes, filename: str) -> ParsedDocument:
    """Dispatch parser based on file extension."""
    lower = filename.lower()
    if lower.endswith(".pdf"):
        return parse_pdf(file_bytes, filename)
    if lower.endswith(".txt"):
        return parse_text(file_bytes, filename)
    return ParsedDocument(
        filename=filename,
        file_type="unknown",
        text="",
        page_count=0,
        char_count=0,
        success=False,
        error=f"Unsupported file type: {filename}",
    )


if __name__ == "__main__":
    print("document_parser module loaded successfully.")
    print("Supported: .pdf, .txt")
