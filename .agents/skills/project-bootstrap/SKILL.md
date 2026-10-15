---
name: project-bootstrap
description: Initialize a new project cloned from the Agent Seed; turn requirements into a durable charter, tasks, and verification workflow.
---

# Project bootstrap

1. Read root `AGENTS.md`, `vault/INDEX.md`, `vault/projects/PROJECT.md`, `vault/CURRENT_STATE.md`, and `vault/TODOS.md`.
2. Inspect repository files and Git status. Determine whether project has already been initialized; do not overwrite existing context.
3. Elicit only missing *essential* requirements when not in user request: goals, users, acceptance criteria, constraints, preferred stack.
4. Write project-specific details into `vault/projects/PROJECT.md`; replace placeholders and set the real commands after confirming them.
5. Establish 2–5 small, testable milestones in `vault/TODOS.md`; start a plan in `vault/projects/PLANS.md` if needed.
6. Update `vault/CURRENT_STATE.md` with a concrete next action; fix index links if needed.
7. Run `python3 scripts/check_context.py`; report remaining open questions.
