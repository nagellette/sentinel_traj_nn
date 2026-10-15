---
name: verification
description: Check a project's code changes, tests, documentation consistency, and readiness for Claude Code session handoff.
---

# Verification

1. Read test/build commands in `vault/projects/PROJECT.md` and project configuration. Avoid assuming tools exist.
2. Inspect `git diff` and changed files; identify likely regressions and sensitive data leaks.
3. Run relevant tests/lints and `python3 scripts/check_context.py` when available.
4. Record exact commands, successes, failures and skipped checks.
5. Suggest fixes or clearly mark blockers; do not claim successful validation without evidence.
6. Propagate verified status into `vault/CURRENT_STATE.md` at a meaningful checkpoint.
