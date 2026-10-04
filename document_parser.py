"""Parse uploaded documents (PDF, TXT, images) for FinGuard Agent."""

from __future__ import annotations

import base64
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
    """Result of parsing one uploaded file."""

    filename: str
    file_type: str  # "pdf" | "txt" | "image" | "unknown"
    text: str  # for text-based files
    image_b64: str  # for image files (base64, no data: prefix)
    image_mime: str  # e.g., "image/png"
    page_count: int
    char_count: int
    size_bytes: int
    success: bool
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


MAX_PDF_PAGES = 20
MAX_TEXT_CHARS = 50_000
MAX_FILES = 5
MAX_FILE_SIZE_MB = 20

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
MIME_MAP = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
}


def _ext_of(filename: str) -> str:
    idx = filename.rfind(".")
    return filename[idx:].lower() if idx >= 0 else ""


def parse_pdf(file_bytes: bytes, filename: str) -> ParsedDocument:
    if not _PYPDF_AVAILABLE:
        return ParsedDocument(
            filename=filename,
            file_type="pdf",
            text="",
            image_b64="",
            image_mime="",
            page_count=0,
            char_count=0,
            size_bytes=len(file_bytes),
            success=False,
            error="pypdf not installed.",
        )

    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        page_count = len(reader.pages)

        if page_count > MAX_PDF_PAGES:
            return ParsedDocument(
                filename=filename,
                file_type="pdf",
                text="",
                image_b64="",
                image_mime="",
                page_count=page_count,
                char_count=0,
                size_bytes=len(file_bytes),
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
                image_b64="",
                image_mime="",
                page_count=page_count,
                char_count=0,
                size_bytes=len(file_bytes),
                success=False,
                error="No text found. May be a scanned/image PDF.",
            )

        return ParsedDocument(
            filename=filename,
            file_type="pdf",
            text=full_text,
            image_b64="",
            image_mime="",
            page_count=page_count,
            char_count=len(full_text),
            size_bytes=len(file_bytes),
            success=True,
        )
    except Exception as exc:
        return ParsedDocument(
            filename=filename,
            file_type="pdf",
            text="",
            image_b64="",
            image_mime="",
            page_count=0,
            char_count=0,
            size_bytes=len(file_bytes),
            success=False,
            error=f"Failed to parse PDF: {exc}",
        )


def parse_text(file_bytes: bytes, filename: str) -> ParsedDocument:
    try:
        text = file_bytes.decode("utf-8", errors="replace")
        if len(text) > MAX_TEXT_CHARS:
            text = text[:MAX_TEXT_CHARS] + "\n\n[... truncated ...]"
        return ParsedDocument(
            filename=filename,
            file_type="txt",
            text=text,
            image_b64="",
            image_mime="",
            page_count=1,
            char_count=len(text),
            size_bytes=len(file_bytes),
            success=True,
        )
    except Exception as exc:
        return ParsedDocument(
            filename=filename,
            file_type="txt",
            text="",
            image_b64="",
            image_mime="",
            page_count=0,
            char_count=0,
            size_bytes=len(file_bytes),
            success=False,
            error=f"Failed to parse text: {exc}",
        )


def parse_image(file_bytes: bytes, filename: str) -> ParsedDocument:
    """Encode image as base64 for multimodal LLM."""
    try:
        ext = _ext_of(filename)
        mime = MIME_MAP.get(ext, "image/png")
        b64 = base64.b64encode(file_bytes).decode("ascii")

        return ParsedDocument(
            filename=filename,
            file_type="image",
            text="",
            image_b64=b64,
            image_mime=mime,
            page_count=1,
            char_count=0,
            size_bytes=len(file_bytes),
            success=True,
        )
    except Exception as exc:
        return ParsedDocument(
            filename=filename,
            file_type="image",
            text="",
            image_b64="",
            image_mime="",
            page_count=0,
            char_count=0,
            size_bytes=len(file_bytes),
            success=False,
            error=f"Failed to encode image: {exc}",
        )


def parse_document(file_bytes: bytes, filename: str) -> ParsedDocument:
    """Dispatch parser based on file extension."""
    lower = filename.lower()

    if lower.endswith(".pdf"):
        return parse_pdf(file_bytes, filename)
    if lower.endswith(".txt"):
        return parse_text(file_bytes, filename)
    if _ext_of(lower) in IMAGE_EXTENSIONS:
        return parse_image(file_bytes, filename)

    return ParsedDocument(
        filename=filename,
        file_type="unknown",
        text="",
        image_b64="",
        image_mime="",
        page_count=0,
        char_count=0,
        size_bytes=len(file_bytes),
        success=False,
        error=f"Unsupported file type: {filename}",
    )


if __name__ == "__main__":
    print("document_parser loaded.")
    print("Supports: .pdf, .txt, .png, .jpg, .jpeg, .webp, .gif")
