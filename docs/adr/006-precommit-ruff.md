# ADR-006: Pre-commit with Ruff and Mypy

## Status
Accepted

## Context

Code quality matters for hackathon judging, but manual review misses:
- Style inconsistencies (import order, trailing whitespace)
- Unused imports, dead code
- Type errors
- Secrets accidentally committed

Doing this manually every commit is slow and error-prone.

## Decision

Use pre-commit framework with:
- ruff - fast Python linter + formatter
- mypy - static type checker
- built-in hooks - trailing-whitespace, end-of-file-fixer, detect-private-key, check-yaml, check-merge-conflict

Hooks run automatically on git commit. If a hook fails, commit is blocked until fixed.

## Consequences

Benefits:
- Consistency - every commit passes same checks
- Fast - ruff lints 1000 files in <100 ms
- Catches bugs early - mypy caught our redis type stub issue
- Prevents secrets - detect-private-key hook blocks .env commits
- Auto-fixes - ruff format rewrites files in place

Drawbacks:
- Slower commits - 5-15s per commit
- Learning curve - developers must understand hook failures
- False confidence - passing mypy doesn't mean code works
- Hook drift - pre-commit versions need periodic updates

Mitigation:
- Hooks cached after first run (fast subsequent commits)
- CI runs same hooks (catches bypasses)
- pre-commit run --all-files for one-shot cleanup

## Alternatives Considered

- flake8 + black + isort: Slower; 3 tools vs 1 (ruff)
- Manual review: Doesn't scale; humans miss details
- CI-only checks: Too late - bugs already pushed
- pre-commit + ruff + mypy: Chosen - best speed + coverage
