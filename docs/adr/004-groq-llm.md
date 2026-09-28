# ADR-004: Groq for LLM Inference

## Status
Accepted

## Context

FinGuard requires fast LLM inference for real-time chat. Options:
- OpenAI GPT-4: $0.03/1K input tokens, 2-5s latency
- Anthropic Claude: similar pricing, 2-4s latency
- Groq: free tier, 0.3-1s latency (LPU hardware)
- Self-hosted Llama: free, but requires GPU instance (~$100/month)
- Nebius AI Studio: hackathon sponsor, but requires credit card

The hackathon budget is $0 and demo must respond in <3 seconds.

## Decision

Use Groq as the LLM inference provider with Llama 3 (via OpenAI-compatible API).

Groq provides:
- Free tier with generous limits (14,400 requests/day for Llama 3.3)
- Ultra-fast inference via custom LPU hardware
- OpenAI-compatible endpoint (drop-in replacement)
- Multiple model options

## Consequences

Benefits:
- Free - critical for hackathon budget
- Fast - 0.3-1s latency vs 2-5s for GPT-4
- OpenAI-compatible - no code rewrite, just base_url swap
- Multiple models - can use 70B for chat, 20B for moderation
- Global edge - low latency from anywhere

Drawbacks:
- Rate limits - 30 req/min on free tier (mitigated by our own rate limiter)
- No SLA - uptime not guaranteed
- Model availability - Groq occasionally removes models
- No streaming in free tier - SSE support limited

Mitigation:
- Client-side rate limiter (10 req/min) prevents hitting Groq limit
- Fallback message if API fails after 2 retries
- Model name in config (easy to swap if Groq deprecates)
- Also compatible with OpenAI/Anthropic via base_url change

## Alternatives Considered

- OpenAI GPT-4: Too expensive for hackathon
- Anthropic Claude: Same cost issue
- Nebius AI Studio: Requires credit card
- Self-hosted Llama: No GPU instance available
- Groq: Chosen - free, fast, OpenAI-compatible
