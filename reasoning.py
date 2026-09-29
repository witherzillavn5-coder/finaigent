"""Parse and present structured reasoning responses from an LLM."""

import html
import json
import re
from dataclasses import dataclass
from typing import Any


@dataclass
class ReasoningResult:
    """Hold parsed reasoning fields and the original response."""

    understanding: str
    analysis: str
    recommendation: str
    confidence: str
    assumptions: list[str]
    answer: str
    raw_response: str
    parsed_ok: bool


def parse_reasoning(raw_text: str) -> ReasoningResult:
    """Parse an LLM response into a validated reasoning result."""
    empty_result = ReasoningResult(
        understanding="",
        analysis="",
        recommendation="",
        confidence="",
        assumptions=[],
        answer=raw_text,
        raw_response=raw_text,
        parsed_ok=False,
    )

    json_text = re.sub(r"^\s*```(?:json)?\s*", "", raw_text.strip(), flags=re.IGNORECASE)
    json_text = re.sub(r"\s*```\s*$", "", json_text)

    try:
        payload: Any = json.loads(json_text)
    except json.JSONDecodeError:
        return empty_result

    if not isinstance(payload, dict):
        return empty_result

    required_fields = (
        "understanding",
        "analysis",
        "recommendation",
        "confidence",
        "assumptions",
        "answer",
    )
    if any(field not in payload for field in required_fields):
        return empty_result

    understanding = payload["understanding"]
    analysis = payload["analysis"]
    recommendation = payload["recommendation"]
    confidence = payload["confidence"]
    assumptions = payload["assumptions"]
    answer = payload["answer"]

    if not all(
        isinstance(value, str)
        for value in (understanding, analysis, recommendation, confidence, answer)
    ):
        return empty_result
    if confidence not in ("high", "medium", "low"):
        return empty_result
    if not isinstance(assumptions, list) or not all(isinstance(item, str) for item in assumptions):
        return empty_result

    return ReasoningResult(
        understanding=understanding,
        analysis=analysis,
        recommendation=recommendation,
        confidence=confidence,
        assumptions=assumptions,
        answer=answer,
        raw_response=raw_text,
        parsed_ok=True,
    )


def format_reasoning_html(reasoning: ReasoningResult) -> str:
    """Format reasoning fields as safe, collapsible HTML."""
    confidence_colors = {
        "high": "#198754",
        "medium": "#b58100",
        "low": "#c62828",
    }
    confidence = html.escape(reasoning.confidence, quote=True)
    badge_color = confidence_colors.get(reasoning.confidence, "#6c757d")

    sections = (
        ("Understanding", reasoning.understanding),
        ("Analysis", reasoning.analysis),
        ("Recommendation", reasoning.recommendation),
    )
    section_html = "".join(
        "<details><summary>"
        + title
        + '</summary><div class="reasoning-content">'
        + html.escape(content, quote=True).replace("\n", "<br>")
        + "</div></details>"
        for title, content in sections
    )

    assumptions_html = ""
    if reasoning.assumptions:
        assumption_items = "".join(
            "<li>" + html.escape(item, quote=True) + "</li>" for item in reasoning.assumptions
        )
        assumptions_html = (
            "<details><summary>Assumptions</summary><ul>" + assumption_items + "</ul></details>"
        )

    return (
        '<div class="reasoning-panel" style="animation: fadeInUp 0.35s ease-out;">'
        "<style>@keyframes fadeInUp{from{opacity:0;transform:translateY(8px)}"
        "to{opacity:1;transform:translateY(0)}}"
        ".reasoning-panel{font-family: sans-serif;line-height:1.5}"
        ".reasoning-panel details{margin:0.5em 0}"
        ".reasoning-panel summary{cursor:pointer;font-weight:600}"
        ".reasoning-content{padding:0.5em 0.75em}</style>"
        '<span style="display:inline-block;padding:0.15em 0.55em;border-radius:999px;'
        "color:#fff;background-color:"
        + badge_color
        + ';">Confidence: '
        + confidence
        + "</span>"
        + section_html
        + assumptions_html
        + "</div>"
    )
