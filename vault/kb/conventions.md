# Engineering conventions

## Coding
- Match existing language conventions and project style.
- Prefer simple, testable and observable changes over abstractions without a need.
- Explain risky migrations and breaking changes before implementing them.

## Project-specific
- New model: add a method to `ModelRepository` and a matching `elif self.model_name == ...` branch in `__init__`; document the name in `scripts/README.md`.
- Use public `tf.keras` / `tensorflow.keras` APIs only; never import from `tensorflow.python.*`.
- Run parameters go in `model_config_files/*.json`, not in code; machine paths go in `~/.env` (template: `.env_template`).
- Keep everything on Keras 2 until T-005 (see ADR-2026-001).
- Results tables live in `experiments/` and `batch_analysis/` with a date in the file name.

## Validation
- Report command, outcome, and relevant failures.
- Add regression tests for bug fixes when feasible.
- After dependency or model changes run `python3 -m unittest tests.test_models_smoke -v` in the full environment.
- Do not invent passing test outcomes.

## Documentation
- Update enduring docs only for factual changes.
- Use brief evidence-based worklogs instead of raw chat transcripts.
- Reference code and commit hashes when helpful.

## Version control
- Inspect the diff before handoff.
- Avoid auto-committing or auto-pushing without authorization.
- Commit context updates alongside related code when approved.
