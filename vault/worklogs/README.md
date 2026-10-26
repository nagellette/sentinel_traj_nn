# Worklogs and session handoffs

Write `YYYY-MM-DD-HHMM-short-topic.md` (local time with timezone noted inside), or add a suffix to avoid collisions. Do not store raw chat logs or secrets. Keep the latest handoff referenced from `../CURRENT_STATE.md`.

Each handoff should include:
- Timestamp and objective
- Current branch and commit; clean/dirty status
- Completed changes with exact paths
- Verification commands and outcomes
- Decisions and assumptions
- Uncommitted work and blockers
- Next 1–3 executable steps
- Recovery / restart instructions

Use `.agents/skills/session-handoff/SKILL.md`. Historical handoffs are evidence, not a replacement for current repository state.
