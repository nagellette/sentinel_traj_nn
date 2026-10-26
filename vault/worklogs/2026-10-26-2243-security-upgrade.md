# 2026-10-26 22:43 CET — Security dependency upgrade and Agent Seed adoption

## Objective
Resolve all open GitHub Dependabot alerts without a GPU/data environment to test on, then adopt the Agent Seed template with its history.

## Git
- `security_dependency_upgrade` (from `master` @ `105f842`): `a2a2267` dependency upgrade + code fixes, `b3fbdbc` README disclaimer.
- `agents_seed_integration` (from `b3fbdbc`): merge of `nagellette/agents-seed` main @ `1c4540f` (`--allow-unrelated-histories`), then project fill-in commit.
- Local remote `seed` → `https://github.com/nagellette/agents-seed.git`. Nothing pushed.

## Changes
- `requirements.txt`: GDAL 3.13.1, pandas 2.2.3, Pillow 13.0.0, python-dotenv 1.2.2, scikit-image 0.24.0, `tensorflow[and-cuda]` 2.15.1, tensorflow-addons 0.23.0, numpy 1.26.4, matplotlib 3.9.2; `tensorflow-gpu` removed.
- `utils/custom_metrics.py`, `models/model_repository.py`: `tensorflow.python.keras` → `tensorflow.keras`.
- `utils/construct_optimizer.py`: `tf.keras.optimizers.legacy.SGD/Adam`, `lr=` → `learning_rate=`.
- `truba_docs/TRUBA_ENV_INST.md`: Python 3.11, `gdal=3.13.1 numpy=1.26.4`, env `tensorflow_2.15`, CUDA from pip.
- `README.md`: security upgrade disclaimer pointing to `105f842`.
- Agent Seed files added; template `LICENSE` and `src/` not adopted; template README → `docs/AGENT_SEED.md`; `.gitignore` merged.
- `tests/test_models_smoke.py`: CPU smoke test (skips without TensorFlow).
- Untracked `.DS_Store` and `output/.DS_Store` (`git rm --cached`).

## Verification
- `pip install --dry-run --only-binary=:all: --platform manylinux2014_x86_64 ... --python-version 3.11 -r requirements.txt` (GDAL excluded): resolves.
- anaconda.org metadata: GDAL 3.13.1 linux-64 py311 builds depend on `numpy >=1.23,<3`.
- Smoke test in a Python 3.11 venv on macOS arm64, CPU: 71/71 pass.
- `python3 scripts/check_context.py`: OK. `python3 -m unittest discover -s tests -p 'test_*.py'`: OK (smoke test skipped without TensorFlow).

## Decisions
ADR-2026-001, ADR-2026-002, ADR-2026-003 in `vault/kb/decisions.md`.

## Not verified / blockers
GPU on TRUBA, GDAL with real rasters, loading TF 2.5 `.h5` models.

## Next steps
1. T-004 push and PRs. 2. T-001 TRUBA env + smoke test. 3. T-002/T-003 real run and old-model reload.

## Recovery
Check `git log --graph --oneline -12` and `git status`; compare with this log and `vault/CURRENT_STATE.md`.
