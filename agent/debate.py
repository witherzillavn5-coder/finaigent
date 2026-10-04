"""Multi-agent debate for complex financial questions.

Three agents:
- Optimist: highlights opportunities and growth potential
- Skeptic: highlights risks, downsides, considerations
- Judge: synthesizes both views into balanced answer
"""

from __future__ import annotations

from dataclasses import dataclass

from openai import OpenAI

DEBATE_MODEL = "openai/gpt-oss-20b"

ANTI_HALLUCINATION_RULES = """
CRITICAL RULES:
- Do NOT invent specific numbers, prices, or statistics.
- If tool results are provided, use those numbers ONLY.
- If no numbers provided, use qualitative language: "significant",
  "substantial", "moderate" instead of fake percentages.
- NEVER cite specific median prices, growth rates, or percentages unless
  they appear in the tool results below.
- No markdown tables, no TAB characters.
"""

OPTIMIST_PROMPT = f"""You are the Optimist financial analyst.

Focus on: opportunities, growth potential, positive scenarios, what could go right.
Maximum 3 sentences. No disclaimers. Do NOT repeat the user's question.
{ANTI_HALLUCINATION_RULES}
"""

SKEPTIC_PROMPT = f"""You are the Skeptic financial analyst.

Focus on: risks, downsides, hidden costs, what could go wrong, worst-case scenarios.
Maximum 3 sentences. No disclaimers. Do NOT repeat the user's question.
{ANTI_HALLUCINATION_RULES}
"""

JUDGE_PROMPT = f"""You are the Judge financial analyst.

Two analysts (Optimist and Skeptic) debated the user's question.
Your task: synthesize their views into a BALANCED, DECISIVE answer.

Rules:
- Acknowledge both views fairly
- End with a clear recommendation
- Maximum 5 sentences
- No disclaimers at start
{ANTI_HALLUCINATION_RULES}
"""


@dataclass
class DebateResult:
    """Result of a 3-agent debate."""

    optimist_view: str
    skeptic_view: str
    judge_view: str
    success: bool = True


def _call_agent(
    client: OpenAI,
    system_prompt: str,
    user_message: str,
    max_tokens: int = 250,
) -> str:
    """Call one debate agent."""
    try:
        response = client.chat.completions.create(
            model=DEBATE_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0.3,
            max_tokens=max_tokens,
            reasoning_effort="low",
        )
        return response.choices[0].message.content or ""
    except Exception:
        return ""


def run_debate(
    client: OpenAI,
    user_query: str,
    context: str = "",
) -> DebateResult:
    """Run 3-agent debate: Optimist vs Skeptic -> Judge."""
    base = f"USER QUESTION: {user_query}"
    if context:
        base += f"\n\nCONTEXT (tool results, use ONLY these numbers):\n{context}"
    else:
        base += "\n\n(No tool results provided. Do NOT invent numbers.)"

    optimist = _call_agent(client, OPTIMIST_PROMPT, base)
    skeptic = _call_agent(client, SKEPTIC_PROMPT, base)

    if not optimist or not skeptic:
        return DebateResult(
            optimist_view=optimist,
            skeptic_view=skeptic,
            judge_view="",
            success=False,
        )

    judge_input = (
        f"{base}\n\n"
        f"OPTIMIST VIEW:\n{optimist}\n\n"
        f"SKEPTIC VIEW:\n{skeptic}\n\n"
        "Now provide your balanced judgment. Do NOT add new numbers."
    )
    judge = _call_agent(client, JUDGE_PROMPT, judge_input, max_tokens=350)

    return DebateResult(
        optimist_view=optimist,
        skeptic_view=skeptic,
        judge_view=judge,
        success=bool(judge),
    )


def format_debate_trace(result: DebateResult) -> str:
    """Format debate result as markdown for UI display."""
    if not result.success:
        return ""

    lines = [
        "**Multi-Agent Debate**",
        "",
        f"**Optimist:** {result.optimist_view}",
        "",
        f"**Skeptic:** {result.skeptic_view}",
        "",
        f"**Judge (final):** {result.judge_view}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    print("debate module loaded.")
    print("Agents: Optimist vs Skeptic -> Judge")
    print("Model:", DEBATE_MODEL)
