"""Persistent user memory for FinGuard Agent.

Stores user profile and past conversation metadata in a local JSON file.
No external database needed — pure file I/O.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DATA_DIR = "data"
PROFILE_FILE = "user_profile.json"
HISTORY_FILE = "conversation_history.jsonl"


@dataclass
class UserProfile:
    """User's financial profile for personalized responses."""

    name: str = ""
    age: int = 0
    monthly_income: float = 0.0
    monthly_expenses: float = 0.0
    savings_goal: str = ""
    risk_tolerance: str = ""  # "low" | "medium" | "high"
    currency: str = "VND"
    notes: str = ""
    updated_at: str = ""

    def is_empty(self) -> bool:
        """Check if profile has no meaningful data."""
        return not any(
            [
                self.name,
                self.age,
                self.monthly_income,
                self.monthly_expenses,
                self.savings_goal,
                self.risk_tolerance,
                self.notes,
            ]
        )

    def to_context_string(self) -> str:
        """Format profile as context for LLM."""
        if self.is_empty():
            return ""

        parts = ["USER PROFILE (remembered from previous sessions):"]
        if self.name:
            parts.append(f"- Name: {self.name}")
        if self.age:
            parts.append(f"- Age: {self.age}")
        if self.monthly_income:
            parts.append(f"- Monthly income: {self.monthly_income:,.0f} {self.currency}")
        if self.monthly_expenses:
            parts.append(f"- Monthly expenses: {self.monthly_expenses:,.0f} {self.currency}")
        if self.savings_goal:
            parts.append(f"- Savings goal: {self.savings_goal}")
        if self.risk_tolerance:
            parts.append(f"- Risk tolerance: {self.risk_tolerance}")
        if self.notes:
            parts.append(f"- Notes: {self.notes}")

        parts.append(
            "Use this information to personalize responses. "
            "Do NOT repeat it back unless relevant to the question."
        )
        return "\n".join(parts)


@dataclass
class ConversationSummary:
    """Summary of one past conversation."""

    session_id: str
    started_at: str
    message_count: int
    first_user_message: str
    tools_used: list[str] = field(default_factory=list)


def _ensure_data_dir() -> None:
    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)


def _profile_path() -> Path:
    return Path(DATA_DIR) / PROFILE_FILE


def _history_path() -> Path:
    return Path(DATA_DIR) / HISTORY_FILE


def load_profile() -> UserProfile:
    """Load user profile from disk. Returns empty profile if not found."""
    path = _profile_path()
    if not path.exists():
        return UserProfile()

    try:
        with open(path, encoding="utf-8") as f:
            data: dict[str, Any] = json.load(f)

        return UserProfile(
            name=str(data.get("name", "")),
            age=int(data.get("age", 0) or 0),
            monthly_income=float(data.get("monthly_income", 0.0) or 0.0),
            monthly_expenses=float(data.get("monthly_expenses", 0.0) or 0.0),
            savings_goal=str(data.get("savings_goal", "")),
            risk_tolerance=str(data.get("risk_tolerance", "")),
            currency=str(data.get("currency", "VND")),
            notes=str(data.get("notes", "")),
            updated_at=str(data.get("updated_at", "")),
        )
    except (json.JSONDecodeError, ValueError, OSError):
        return UserProfile()


def save_profile(profile: UserProfile) -> None:
    """Save user profile to disk."""
    _ensure_data_dir()
    profile.updated_at = datetime.now(UTC).isoformat()

    try:
        with open(_profile_path(), "w", encoding="utf-8") as f:
            json.dump(asdict(profile), f, ensure_ascii=False, indent=2)
    except OSError:
        pass


def clear_profile() -> None:
    """Delete user profile file."""
    path = _profile_path()
    if path.exists():
        try:
            os.remove(path)
        except OSError:
            pass


def append_conversation_summary(summary: ConversationSummary) -> None:
    """Append a conversation summary to history file (JSONL)."""
    _ensure_data_dir()
    try:
        with open(_history_path(), "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(summary), ensure_ascii=False) + "\n")
    except OSError:
        pass


def load_recent_summaries(limit: int = 5) -> list[ConversationSummary]:
    """Load most recent conversation summaries."""
    path = _history_path()
    if not path.exists():
        return []

    try:
        with open(path, encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
    except OSError:
        return []

    summaries: list[ConversationSummary] = []
    for line in lines[-limit:]:
        try:
            data = json.loads(line)
            summaries.append(
                ConversationSummary(
                    session_id=str(data.get("session_id", "")),
                    started_at=str(data.get("started_at", "")),
                    message_count=int(data.get("message_count", 0)),
                    first_user_message=str(data.get("first_user_message", "")),
                    tools_used=list(data.get("tools_used", [])),
                )
            )
        except (json.JSONDecodeError, ValueError, TypeError):
            continue

    return summaries


def clear_history() -> None:
    """Delete conversation history file."""
    path = _history_path()
    if path.exists():
        try:
            os.remove(path)
        except OSError:
            pass


if __name__ == "__main__":
    print("memory module loaded.")

    # Test profile
    p = UserProfile(
        name="Test User",
        age=30,
        monthly_income=30000000,
        monthly_expenses=15000000,
        savings_goal="Buy house 2 billion VND in 5 years",
        risk_tolerance="medium",
    )
    print("\nProfile context:")
    print(p.to_context_string())

    save_profile(p)
    loaded = load_profile()
    print(f"\nLoaded name: {loaded.name}")

    # Test summary
    s = ConversationSummary(
        session_id="test-123",
        started_at=datetime.now(UTC).isoformat(),
        message_count=5,
        first_user_message="Vay 500 triệu",
        tools_used=["calculate_loan_payment"],
    )
    append_conversation_summary(s)
    recent = load_recent_summaries(limit=3)
    print(f"\nRecent summaries: {len(recent)}")
