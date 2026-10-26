# Knowledge index

Read this index first. Load only the documents needed for the current task. Update it when adding durable knowledge sources.

| Location | What it contains | When to read |
| --- | --- | --- |
| `projects/PROJECT.md` | Project mission, scope, constraints, acceptance criteria | Every new session |
| `CURRENT_STATE.md` | Current position, active branch, known risks, next action | Every new session and checkpoint |
| `TODOS.md` | Prioritized tasks, statuses, next steps | Every new session |
| `projects/PLANS.md` | Active and archived execution plans | Multi-step or high-risk work |
| `kb/decisions.md` | Accepted/reversed decisions and why | Before changing design |
| `kb/architecture.md` | System design and technical invariants | Before architectural code changes |
| `kb/conventions.md` | Coding and verification expectations | When contributing code |
| `notes/README.md` | Exploratory notes and sources | Research-oriented work |
| `worklogs/` | Dated evidence-rich checkpoints and handoffs | Latest handoff at startup; older only if relevant |

**Truth ordering:** explicit new user instructions > current code/configuration/test results > maintained current-state docs > historical worklogs. Flag conflicts instead of guessing.
