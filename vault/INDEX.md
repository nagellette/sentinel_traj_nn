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
| `../scripts/README.md` | Model names, run arguments, experiment/batch table formats | Before running or adding models |
| `../truba_docs/` | TRUBA environment setup and SLURM commands | Environment or cluster work |
| `../model_config_files/` | Training (`model_config_*`) and input raster (`input_files_*`) configs | Before changing run parameters |
| `../Data.md` | Link to the external dataset | Data questions |
| `../docs/AGENT_SEED.md` | Upstream Agent Seed README and attribution | Updating the template (`git merge seed/main`) |

**Truth ordering:** explicit new user instructions > current code/configuration/test results > maintained current-state docs > historical worklogs. Flag conflicts instead of guessing.
