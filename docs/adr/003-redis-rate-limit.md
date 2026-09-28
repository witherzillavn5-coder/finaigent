# ADR-003: Redis for Server-Side Rate Limiting

## Status
Accepted

## Context

FinGuard must limit requests per user to prevent accidental spam, cost amplification (each request triggers paid LLM inference), and DoS attacks.

Initial implementation stored timestamps in st.session_state (Streamlit session). Problems:
- Per-session only - user opens new tab means reset
- In-memory - restart app means all limits cleared
- Not scalable - 2 app instances means 2 independent limits

## Decision

Use Redis as a shared rate-limit backend:
- Sorted set per user: finaigent:rl:<user_id>
- Each request adds a timestamp
- Old timestamps (>60s) automatically removed via ZREMRANGEBYSCORE
- Expiry set on key to avoid leaks
- Fallback to in-memory dict if Redis is unavailable

## Consequences

Benefits:
- Shared across all app instances (horizontal scaling)
- Survives app restarts
- Atomic operations via Redis pipeline (no race conditions)
- TTL-based cleanup (no manual pruning)
- Fallback ensures app still works if Redis crashes

Drawbacks:
- Extra infrastructure (Redis container ~5 MB)
- Network hop per request (~0.5 ms)
- Requires Docker or managed Redis in production

Mitigation:
- 2-second timeout on Redis operations
- Automatic fallback to in-memory if Redis down
- Docker Compose file makes local setup trivial

## Alternatives Considered

- In-memory only: Not shared across instances; lost on restart
- Postgres: Overkill for rate limiting; slower for counters
- Memcached: No sorted-set operations; harder atomic time-window
- Redis: Chosen - perfect fit for time-window counters
