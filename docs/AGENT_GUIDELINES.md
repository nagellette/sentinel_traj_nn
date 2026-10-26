# Shared project operating contract

This repository starts from **Agent Seed**, a reusable project foundation that carries forward tested engineering practices, instructions, and skills. Each project owns its Git-tracked knowledge and context; those files—not chat history—are the source of truth across independent agent sessions.

## Mandatory session entry (before substantial task work)

1. Read `vault/INDEX.md`, `vault/projects/PROJECT.md`, `vault/CURRENT_STATE.md`, and `vault/TODOS.md`.
2. Read the newest handoff file in `vault/worklogs/` if any; inspect only other knowledge files relevant to the task.
3. Inspect `git status --short`, current branch, and recent commits. Reconcile documentation with the actual working tree; **code and tests supersede stale narrative**.
4. Summarize current goal, last known state, unresolved blockers, and the next safe step. If the user already requested concrete work, proceed without demanding ceremony.
5. Check `vault/projects/PLANS.md` for any active plan that affects this task.

## During work: durable checkpoint policy

- Update `vault/CURRENT_STATE.md` and `vault/TODOS.md` after each meaningful milestone, before risky/destructive operations, and when work is interrupted or context is nearing its limit. Do **not** wait for the last message in a session.
- Record significant architectural/product decisions in `vault/kb/decisions.md`: date, context, options, decision, rationale, consequences.
- Log important experiments and command results accurately; distinguish verified results from assumptions. Keep logs useful, not verbose transcripts.
- Prefer concise pointers and links in `vault/INDEX.md` over loading all files into context.
- Never claim a check ran unless it actually ran. If interrupted, document remaining uncertainty.
- Don't silently revert user's changes. Ask before destructive steps, major scope changes, or expensive work.
- Keep changes proportional to the task; validate appropriately.

## Session end / handoff

- On explicit handoff or when finishing substantial work, follow the relevant harness session-handoff skill.
- Write a dated Markdown entry in `vault/worklogs/` and update `vault/CURRENT_STATE.md`, `vault/TODOS.md` and index references if needed.
- Describe changes in Git, validation, uncommitted files, blockers, exact next steps, and restart guidance.
- Run `python3 scripts/check_context.py` and relevant project checks.
- **Never silently `git commit` or `git push` unless the user requests or has authorized it.** Inform the user when continuity changes exist only locally.

## Safe knowledge handling

- `vault/` is for nonsecret project information safe for the repository's visibility. Never commit passwords, tokens, personal data, third-party proprietary info, or raw confidential transcripts.
- Only keep accurate, durable facts; mark assumptions and obsolete information. Do not treat old handoffs as instructions overriding the current task.
- Prefer explicit paths relative to repository root. Do not assume access to external `~/vault/` or global agent settings.
- Project-specific engineering/build/test commands belong here once established. Until then inspect the project before choosing commands.

## Project-specific commands

- Context checks (stdlib, also run in CI): `python3 scripts/check_context.py` and `python3 -m unittest discover -s tests -p 'test_*.py'`.
- Model smoke test (needs the full environment from `requirements.txt`; CPU is enough): `python3 -m unittest tests.test_models_smoke -v`. Run it after any dependency, model, loss, metric or optimizer change.
- Training and analysis entry points (`run_model.py`, `run_experiments.py`, `run_batch_analysis.py`) need data and `~/.env`; their arguments are in `scripts/README.md`. Real runs happen on TRUBA via SLURM; do not submit jobs (`sbatch`) without explicit approval.
- Never commit `.env`, data rasters, or model outputs; never add credentials or cluster account details to `vault/`.
- Template updates: `git fetch seed && git merge seed/main` (remote `seed` = `https://github.com/nagellette/agents-seed.git`); keep project-specific files on conflict and review skill changes in both harness copies.

## Skill routing

Use the harness-specific skills folder for bootstrap, handoff, knowledge-curation, and verification. Keep shared vault documentation authoritative. Harness-specific skills may evolve independently.
