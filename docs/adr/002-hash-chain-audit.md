# ADR-002: Hash Chain for Tamper-Proof Audit Logs

## Status
Accepted

## Context

FinGuard logs every security event (blocked attacks, PII redactions, API errors) to a JSONL file for post-incident analysis. Initial implementation used plain JSONL: each line was an independent event.

Problem: anyone with file access can modify a log entry without detection. In a financial context, this is unacceptable — audit logs must be tamper-evident.

## Decision

Implement a SHA256 hash chain for audit logs, similar to a blockchain:
- Each entry stores prev_hash (hash of previous entry)
- Each entry's hash = SHA256(entry_data + prev_hash)
- First entry uses prev_hash = "000...0" (genesis)

Modifying any entry breaks the chain for all subsequent entries, making tampering detectable via verify_chain().

## Consequences

Benefits:
- Tamper-evident: any modification is detectable
- No external dependency (uses Python's hashlib)
- Low overhead: SHA256 on small JSON is ~0.001 ms
- Verifiable offline

Drawbacks:
- Cannot modify old entries even for legitimate reasons (append-only)
- Log rotation requires rebuilding the chain
- Verification is O(n) — slower for large logs

Mitigation:
- Chain verification is on-demand (button in UI), not on every log write
- For production scale, consider Merkle tree instead of linear chain

## Alternatives Considered

- Plain JSONL: No tamper detection
- Digital signatures (RSA): Slower; requires key management
- Write-only storage: Requires infrastructure; not portable
- Hash chain: Chosen - simple, portable, fast enough
- Merkle tree: Overkill for hackathon scale
