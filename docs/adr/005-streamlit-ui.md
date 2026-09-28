# ADR-005: Streamlit for Chat UI

## Status
Accepted

## Context

FinGuard needs a chat UI to demo. Options:
- React + FastAPI: full control, but 2-3 days to build
- Gradio: ML-focused, limited UI customization
- Streamlit: Python-native, 1-2 hours to build
- Next.js + Vercel AI SDK: modern but heavy setup

The hackathon timeline is 4 weeks with 1.5h/day = ~62 hours total.

## Decision

Use Streamlit for the chat UI.

Rationale:
- Python-only: no separate frontend codebase
- Built-in chat components (st.chat_message, st.chat_input)
- Custom CSS injection for branded look
- Session state for chat history
- One-line deploy

## Consequences

Benefits:
- Fast to build - chat UI in 2 hours vs 2 days for React
- Single language - no context switch Python to JS
- Built-in widgets - metrics, charts, file download, toggle
- Custom CSS - we injected ~80 lines to match financial branding
- Rerun model - simple mental model for state management

Drawbacks:
- Rerun on every interaction - sidebar counters lag by 1 message
- Limited animation - CSS animations only, no JS
- Not mobile-first - responsive but not optimized
- Single-process - no built-in load balancing

Mitigation:
- st.rerun() after critical state changes
- Custom CSS animations (fadeInUp, pulseGlow)
- Deploy behind nginx for multi-user production

## Alternatives Considered

- React + FastAPI: Too much time for hackathon
- Gradio: UI too restrictive; chat UX poor
- Next.js + Vercel: JS required; overkill
- Streamlit: Chosen - best time-to-demo ratio
