# Current state

Last checkpoint: **2026-10-26 22:43 CET**
Project phase: **maintenance — security dependency upgrade**
Branch / last known commit: **`agents_seed_integration`**; security upgrade merged to `master` via PR #169 (`e36c954`)
Latest worklog: `vault/worklogs/2026-10-26-2243-security-upgrade.md`

## One-paragraph status
Research code for road extraction from Sentinel-2 + GPS trajectories. Dependencies were upgraded to fix the GitHub security alerts (TF 2.5 → 2.15.1, Python 3.11, Pillow 13, GDAL 3.13.1, ...) with small compatibility fixes, and Agent Seed was merged in with its history. The upgraded stack has only been verified on CPU with random data.

## Most recent verified results
- 2026-10-26: `pip install --dry-run` for linux x86_64 / Python 3.11 resolves `requirements.txt` (without GDAL) including CUDA 12.2 / cuDNN 8.9 wheels.
- 2026-10-26: conda-forge has GDAL 3.13.1 linux-64 py311 builds requiring `numpy >=1.23,<3`.
- 2026-10-26: CPU smoke test on macOS arm64 (TF 2.15.1, tfa 0.23.0, numpy 1.26.4) — 71/71 cases pass: all 19 models, 5 fusion types, 7 losses × 2 optimizers, metrics as in `run_analysis.py`, SavedModel + `.h5` save/reload.
- 2026-10-26: `python3 scripts/check_context.py` and `python3 -m unittest discover -s tests -p 'test_*.py'` pass (see worklog).

## In progress
- PR for `agents_seed_integration` → `master` opened; confirm Dependabot alerts close after the rescan of `master` (T-004).

## Risks, unknowns, blockers
- Not tested: GPU on TRUBA (driver must be >= 525 for CUDA 12.2), GDAL with real rasters, loading models saved under TF 2.5.
- TF 2.15 and tensorflow-addons are end-of-life (ADR-2026-001).
- Pre-upgrade working version for reproducing old results: commit `105f842`.

## Next actions (ordered)
1. Merge the Agent Seed PR (merge commit, keeps template history) and confirm the alerts closed (T-004).
2. Build the TRUBA environment and run the smoke test there (T-001).
3. Short real training run and old-model reload (T-002, T-003).

## Recovery notes
Start by reviewing Git status/diff and this file. If a session ended unexpectedly, validate the working tree before trusting the last narrative checkpoint.
