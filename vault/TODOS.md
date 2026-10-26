# Tasks

Use checkboxes and stable identifiers. Move completed items to Done periodically; include acceptance criteria where possible.

## Now
- [ ] T-001 — Create the TRUBA conda env per `truba_docs/TRUBA_ENV_INST.md`. Done when `nvidia-smi` shows driver >= 525, `python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"` lists a GPU, `from osgeo import gdal` works, and `python3 -m unittest tests.test_models_smoke -v` passes.

## Next
- [ ] T-002 — One short real training run on TRUBA (e.g. `unet` and one `*_traj_type2` model, few epochs). Done when it finishes with `*_log_completed.csv` and both model files saved.
- [ ] T-003 — Reload a model trained under TF 2.5 (`.h5`) with `run_analysis.py`. Done when predictions and metrics are produced; otherwise document the failure and the workaround (use commit `105f842`).

## Later
- [ ] T-005 — Plan a Keras 3 migration: replace `tensorflow-addons` (focal/contrastive loss, F1) and `keras-unet-collection` transformer blocks, move off legacy optimizers and extension-less `model.save()`.

## Blocked
- None.

## Done
- [x] T-000 — Install Agent Seed (merged with history, 2026-10-26).
- [x] T-S01 — Upgrade dependencies to resolve security alerts (`a2a2267`, README disclaimer `b3fbdbc`), 2026-10-26.
- [x] T-004 — PRs merged with merge commits (security upgrade #169, Agent Seed #170; README references #171). Dependabot: 0 open, 492 fixed, 2026-10-26.
