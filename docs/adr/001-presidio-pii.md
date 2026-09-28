# ADR-001: Use Presidio for PII Detection

## Status
Accepted

## Context

FinGuard Agent must detect and redact Personally Identifiable Information (PII)
before sending user input to a cloud LLM. Initial implementation used only regex
patterns for credit cards, SSN, OTP, and CVV.

Problems with regex-only approach:
- Misses context-dependent PII (e.g., "my email is john at example dot com")
- Cannot detect entity types outside predefined patterns (names, addresses, IBAN)
- Requires maintaining separate regex for every country/format
- High false-positive rate (e.g., `\d{3,4}` matches years, prices, room numbers)

## Decision

Integrate **Microsoft Presidio** — an open-source, ML-based PII detection framework
built on spaCy NLP models.

Presidio runs locally (no cloud API), supports multi-language, and is extensible
via custom PatternRecognizers.

## Consequences

**Benefits:**
- Detects 20+ PII entity types out of the box (email, IBAN, person names, locations)
- Context-aware: distinguishes "SSN 123-45-6789" from "order #123456789"
- Extensible: we added a custom recognizer for Vietnamese CCCD (12-digit ID)
- Battle-tested by Microsoft, used in production at scale

**Drawbacks:**
- Adds ~18 ms latency per request (measured in `benchmarks/run_bench.py`)
- Requires spaCy model download (~12 MB)
- Increases Docker image size by ~200 MB
- Adds 3 dependencies: `presidio-analyzer`, `presidio-anonymizer`, `spacy`

**Mitigation:**
- Regex fallback for SSN, OTP, CVV (Presidio cannot detect these when standing alone)
- Luhn check on credit cards to filter false positives from Presidio
- Configurable: can disable Presidio via `_PRESIDIO_AVAILABLE` flag if latency is critical

## Alternatives Considered

| Option | Why rejected |
|--------|--------------|
| **Regex only** | Cannot detect context-dependent PII; high false positives |
| **AWS Comprehend** | Requires cloud API, adds cost, sends PII to third party |
| **Google DLP API** | Same as AWS; also needs GCP account |
| **Custom NER model** | 2–3 weeks to train; out of hackathon scope |
| **Presidio + regex** | **Chosen** — best balance of accuracy, latency, cost |
