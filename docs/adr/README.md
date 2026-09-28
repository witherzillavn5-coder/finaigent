# Architecture Decision Records

This directory contains Architecture Decision Records (ADRs) for FinGuard Agent.

Each ADR documents a significant architectural decision, the context that drove it,
and the consequences (trade-offs) of the choice.

## Index

| ADR | Title | Status |
|-----|-------|--------|
| [001](001-presidio-pii.md) | Use Presidio for PII Detection | Accepted |
| [002](002-hash-chain-audit.md) | Hash Chain for Tamper-Proof Audit Logs | Accepted |
| [003](003-redis-rate-limit.md) | Redis for Server-Side Rate Limiting | Accepted |
| [004](004-groq-llm.md) | Groq for LLM Inference | Accepted |
| [005](005-streamlit-ui.md) | Streamlit for Chat UI | Accepted |
| [006](006-precommit-ruff.md) | Pre-commit with Ruff and Mypy | Accepted |

## Format

Each ADR follows this structure:

- **Status**: Proposed / Accepted / Deprecated / Superseded
- **Context**: What problem are we solving?
- **Decision**: What did we choose?
- **Consequences**: What are the trade-offs?
- **Alternatives Considered**: What else did we evaluate?
