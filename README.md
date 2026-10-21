# Agent Seed

A reusable, Git-first project seed for Codex and Claude Code. New projects inherit a foundation of agent instructions, reusable skills, engineering practices, and structured knowledge. As each project evolves, its decisions and progress stay durable across sessions, while transferable lessons can be contributed back to the seed for future projects.

## Inspiration and attribution

This template is inspired by Eugene Yan’s article **[How to Work and Compound with AI](https://eugeneyan.com/writing/working-with-ai/)**. It adapts ideas such as persistent context, reusable instructions and skills, verification, and continuous improvement into a **general-purpose project template for Codex and Claude Code**.

It is an independent, community-style adaptation—not an official template by Eugene Yan or OpenAI, and not a one-to-one reproduction of the article’s Claude-oriented setup. The repository-local `vault/` and session-handoff conventions are deliberate changes to make new projects portable and resumable across agent sessions.

## Start a new project

**Option A: GitHub template** (after publishing this repository on GitHub): select **Use this template**, create your own repository, then clone it.

**Option B: Clone locally and make a fresh history:**

```bash
git clone <TEMPLATE_GIT_URL> my-project
cd my-project
rm -rf .git
git init
git add .
git commit -m "chore: initialize project from Agent Seed"
```

Or extract the provided ZIP into a new project directory and initialize Git. Do not clone the template into a project that needs separate history without reinitializing Git.

## First-run workflow

1. Edit `vault/projects/PROJECT.md` with project goals, scope, and constraints.
2. Edit `vault/INDEX.md` to reference any new source of truth.
3. Run `python3 scripts/check_context.py` (only Python standard library required).
4. Open the repository root in Codex CLI (`codex`) or Claude Code (`claude`).
5. Say **"Bootstrap this project using the project-bootstrap skill. Ask for missing project requirements, then update the project context."**

Codex loads `AGENTS.md` and discovers `.agents/skills/`; Claude Code loads `CLAUDE.md` and discovers `.claude/skills/`. The two skills directories contain **independent copies**, not symlinks. Shared context and operating principles live in `vault/` and `docs/AGENT_GUIDELINES.md`.

## Session continuity

**On entry:** The agent reads `vault/INDEX.md`, `vault/CURRENT_STATE.md`, `vault/TODOS.md`, and the latest handoff under `vault/worklogs/`. It checks Git state before continuing.

**During work:** It updates `CURRENT_STATE.md` and `TODOS.md` at meaningful milestones, captures durable decisions in `vault/kb/decisions.md`, and writes dated worklogs for substantial tasks.

**At handoff:** Say **"Create a session handoff"** (or request the `session-handoff` skill). It records the current branch/commit, completed work, validation results, blockers, next actions, and any uncommitted changes. Run the context validator and commit/push approved work. The handoff files are intentionally tracked in Git.

**On restart:** Launch a *new* Codex or Claude Code session in the **same checked-out repository**. You need not resume the old chat; the committed context survives. If using another machine, fetch/pull the latest Git changes first. If the previous run ended abruptly, the next session reconstructs from the most recent checkpoint **and** the actual Git tree/diff, and calls out uncertainties.

> Continuity is best-effort, not magical: an abrupt termination before a checkpoint can lose ephemeral reasoning; uncommitted work on another machine does not travel with Git. Never store secrets, credentials, private transcripts, or confidential source data in tracked Markdown.

## Key files

| Path | Purpose |
| --- | --- |
| `AGENTS.md` | Codex entry point |
| `CLAUDE.md` | Claude Code entry point |
| `docs/AGENT_GUIDELINES.md` | Shared operating contract |
| `.codex/config.toml` / `.claude/settings.json` | Harness-specific configuration |
| `.claude/skills/` | Claude Code skill definitions (independently maintained) |
| `vault/INDEX.md` | Selective guide to context files |
| `vault/CURRENT_STATE.md` | Single-page latest project status, updated frequently |
| `vault/TODOS.md` | Prioritized actionable task list |
| `vault/projects/PROJECT.md` | Goals, constraints, success criteria |
| `vault/kb/decisions.md` | Durable decisions with evidence and trade-offs |
| `vault/worklogs/` | Chronological checkpoints and session handoffs |
| `.agents/skills/` | Codex workflow skill definitions |
| `scripts/check_context.py` | Mechanical context-integrity checks |
| `docs/WORKFLOW.md` | Detailed startup, checkpoint, handoff, recovery guidance |

## Customization

Replace the example placeholders, keep worklogs short, move lasting insights to `kb/`, and prune stale instructions. For large plans use `vault/projects/PLANS.md`. Introduce tooling-specific test commands into `AGENTS.md` once the tech stack is known. Any external `~/vault/` is optional; this repo-local vault is the portable source of truth.

## Evolving Agent Seed across projects

Each new project inherits the seed but maintains its own project-specific code, decisions, tasks, and logs. When a project reveals a **generalizable, verified practice**, propose it as a change to the upstream `agent-seed` template rather than automatically transferring project-specific material. Review and test those improvements before future projects inherit them. Never propagate confidential or proprietary content.

The upstream seed evolves independently; existing projects do not receive its updates automatically. Treat edits to `AGENTS.md`, skills, and shared conventions as code changes that require review.

## Harness independence and skill maintenance

Agent Seed is a reusable, general-purpose starting point. Both harnesses operate on the **same tracked project state** in `vault/`, but each has its own entry point and skills directory. There are intentionally **no symlinks** between `.agents/skills/` and `.claude/skills/`.

When improving a workflow relevant to both agents, **review and update both copies** of that skill. They are not automatically synchronized; you may also deliberately customize one copy for a specific harness. Keep universal rules in `docs/AGENT_GUIDELINES.md` rather than duplicating them in entry points. The `.codex/config.toml` and `.claude/settings.json` files are minimal placeholders to avoid imposing potentially unsafe permissions or settings.

When useful knowledge emerges from a project, review it for general applicability and contribute the sanitized lesson to this master template **explicitly**. Do not automatically push individual projects' context, secrets, or transcripts to the master template.
