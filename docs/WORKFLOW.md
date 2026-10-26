# Operating procedure

## Startup, every new Codex session

1. Work from repository root so root `AGENTS.md` is discoverable.
2. `AGENTS.md` requires selective reading of index, charter, state, tasks, and newest worklog.
3. Run `git status --short`, `git branch --show-current`, `git log -1 --oneline` (when commits exist).
4. Check whether changes from a previous session are uncommitted or partially applied; reconcile them with the last checkpoint.
5. Execute user request, keeping task scope explicit.

Suggested first prompt: `Read AGENTS.md and resume from the repository's durable state. Summarize next steps, then continue T-001.`

## During work

Checkpoint after a validated milestone, before destructive work, and before long context-heavy explorations. Update the **single-page** `CURRENT_STATE.md`, task status and, when meaningful, a dated worklog. Avoid creating a separate log for each minor edit.

## End / transfer

Ask: `Use the session-handoff skill to checkpoint this work for a brand-new Codex session.`

- Update state/tasks, any key decisions and a new worklog.
- Verify file paths, test evidence, open errors and actual Git status.
- Run `python3 scripts/check_context.py`.
- Commit/push after explicit authorization or using your existing Git workflow.

## Abrupt termination / recovery

There is **no guaranteed shutdown callback**. A killed process cannot record its last reasoning. New session compares saved context against actual code, diff, and test results, and reports uncertain or incomplete work. Improve recovery by checkpointing during work and version-controlling state files.

## Multi-device use

A fresh clone gets only **committed/pushed** context, not uncommitted local modifications. Pull latest changes, resolve conflicts normally, and be cautious of concurrent sessions editing `CURRENT_STATE.md` or `TODOS.md`. Prefer feature branches for parallel agents; consolidate handoffs in the integration branch.

## Security

Documentation travels with the repository and inherits its access settings. Treat it as publishable within the project audience; no secrets or confidential raw discussions. Do not put generated runtime files or credentials under tracked knowledge folders.
