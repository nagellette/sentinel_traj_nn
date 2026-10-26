---
name: session-handoff
description: Produce durable Git-tracked checkpoints and recovery instructions so a new Codex session can pick up work without the previous chat.
---

# Session handoff

Perform these actions when asked to checkpoint/hand off, before planned interruption, and when closing substantial work if allowed by the task.

1. Read `vault/CURRENT_STATE.md`, `vault/TODOS.md`; inspect real `git status --short`, branch and `git log -1 --oneline` (if available).
2. Verify the claimed implementation against current files; run appropriate safe validations if practical. Mark anything unverified explicitly.
3. Update `vault/CURRENT_STATE.md` with timestamp (include timezone), completed/in-progress work, risks, next exact step, branch/hash, and latest worklog path.
4. Update task statuses in `vault/TODOS.md`; reconcile active plans and decisions if a milestone changed them.
5. Create `vault/worklogs/YYYY-MM-DD-HHMM-topic.md` using the structure in `vault/worklogs/README.md`. Summarize *facts and concrete paths*, not internal reasoning or chat history.
6. Run `python3 scripts/check_context.py`. Recheck Git status and explicitly report uncommitted files, including the new handoff.
7. Do not commit or push without user authorization. Explain if handoff exists locally only; a new machine must receive a commit through Git.

Never invent test results, dates, commits, or deployment outcomes.
