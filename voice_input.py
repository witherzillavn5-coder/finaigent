"""Voice input for FinGuard Agent using Groq Whisper."""

from __future__ import annotations

import io
from dataclasses import dataclass

from openai import OpenAI

WHISPER_MODEL = "whisper-large-v3"
MAX_AUDIO_SIZE_MB = 25
SUPPORTED_LANGUAGES = {"vi", "en"}


@dataclass
class TranscriptionResult:
    """Result of audio transcription."""

    text: str
    language: str
    duration_seconds: float
    success: bool
    error: str | None = None


def transcribe_audio(
    client: OpenAI,
    audio_bytes: bytes,
    filename: str = "audio.wav",
    language: str | None = None,
) -> TranscriptionResult:
    """Transcribe audio using Groq Whisper API."""
    if not audio_bytes:
        return TranscriptionResult(
            text="",
            language="",
            duration_seconds=0.0,
            success=False,
            error="Empty audio",
        )

    size_mb = len(audio_bytes) / (1024 * 1024)
    if size_mb > MAX_AUDIO_SIZE_MB:
        return TranscriptionResult(
            text="",
            language="",
            duration_seconds=0.0,
            success=False,
            error=f"Audio too large ({size_mb:.1f} MB, max {MAX_AUDIO_SIZE_MB} MB)",
        )

    try:
        audio_file = io.BytesIO(audio_bytes)
        # Detect format from magic bytes
        if audio_bytes[:4] == b"RIFF":
            audio_file.name = "audio.wav"
        elif audio_bytes[:4] == b"\x1a\x45\xdf\xa3":
            audio_file.name = "audio.webm"
        else:
            audio_file.name = filename

        kwargs: dict = {
            "model": WHISPER_MODEL,
            "file": audio_file,
            "response_format": "verbose_json",
        }
        if language and language in SUPPORTED_LANGUAGES:
            kwargs["language"] = language

        response = client.audio.transcriptions.create(**kwargs)

        text = getattr(response, "text", "") or ""
        detected_lang = getattr(response, "language", "") or ""
        duration = getattr(response, "duration", 0.0) or 0.0

        if not text.strip():
            return TranscriptionResult(
                text="",
                language=detected_lang,
                duration_seconds=duration,
                success=False,
                error="No speech detected",
            )

        return TranscriptionResult(
            text=text.strip(),
            language=detected_lang,
            duration_seconds=float(duration),
            success=True,
        )

    except Exception as exc:
        return TranscriptionResult(
            text="",
            language="",
            duration_seconds=0.0,
            success=False,
            error=f"Transcription failed: {exc}",
        )


if __name__ == "__main__":
    print("voice_input module loaded.")
    print("Model:", WHISPER_MODEL)
    print("Max size:", MAX_AUDIO_SIZE_MB, "MB")
