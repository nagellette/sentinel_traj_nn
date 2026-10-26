# Project charter

Status: **initialized 2026-10-26** (repository existed before Agent Seed was adopted; charter reconstructed from code and docs)

## Project name
sentinel_traj_nn — road extraction from Sentinel-2 imagery and GPS trajectories with deep learning

## Problem / motivation
Extract road networks as binary segmentation masks from Sentinel-2 multispectral (MSI) imagery, optionally fused with GPS trajectory density rasters, and compare model families and fusion strategies across study areas (Istanbul, Montreal).

## Outcomes / acceptance criteria
- [x] Train and evaluate segmentation models on satellite-only and satellite + trajectory inputs (historical experiments under `experiments/`, `batch_analysis/`).
- [ ] Keep the code runnable on the upgraded, security-patched dependency stack (see `vault/TODOS.md` T-001…T-004).

## Scope
**In:** model definitions (`models/model_repository.py`), data pipeline (`utils/`), training/evaluation entry points (`run_*.py`), TRUBA SLURM job templates (`scripts/`), experiment and analysis tables/notebooks.

**Out:** raw data in the repository (data is hosted externally, see `Data.md`), production deployment.

## Constraints and non-goals
- Training runs on the TRUBA HPC cluster (SLURM, GPU partitions such as `akya-cuda`); local runs are for debugging.
- Stay on Keras 2 (`tf.keras`) until a planned migration: `tensorflow-addons` and `keras-unet-collection` do not support Keras 3 (ADR-2026-001).
- Paths are machine-specific and come from a `~/.env` file, never from tracked files.

## Stakeholders / users
Repository owner (research project).

## References
- Publication: Gengec, Tari, Bagci (2024), *Earth Science Informatics* 17(2):1013–1029, doi:10.1007/s12145-023-01201-6 (preprint arXiv:2312.17040).
- PhD thesis: YÖK National Thesis Center link in `README.md`.
- Dataset: Hugging Face `negengec/sentinel_trajectory_ist_mont` (CC-BY-4.0), see `Data.md`.

## Stack and commands
- Runtime / versions: Python 3.11, TensorFlow 2.15.1 (Keras 2.15), CUDA 12.2 / cuDNN 8.9 via `tensorflow[and-cuda]`, GDAL 3.13.1 (conda-forge), numpy 1.26.4. Full pins in `requirements.txt`.
- Setup: `truba_docs/TRUBA_ENV_INST.md` (conda for Python + GDAL, then `pip install -r requirements.txt`); copy `.env_template` to `~/.env` and set paths.
- Test:
  - Context/template checks (stdlib only, run in CI): `python3 scripts/check_context.py` and `python3 -m unittest discover -s tests -p 'test_*.py'`
  - Model smoke test (needs the full environment; skipped when TensorFlow is missing): `python3 -m unittest tests.test_models_smoke -v`
- Lint / format: none configured.
- Run:
  - Train: `python run_model.py <model_config.json> <input_files.json | "[list]"> <model_name> <run_tag>` (see `scripts/README.md` for model names and the `test` tag for local runs)
  - Submit an experiment table to SLURM: `python run_experiments.py ./experiments/<table>.csv ./scripts/run_remote_template.sh`
  - Batch evaluation: `python3 run_batch_analysis.py ./batch_analysis/<list>.csv ./scripts/run_analysis_template.sh`

## External systems and secrets
- TRUBA cluster account and VPN (`truba_docs/README.md`); no credentials in the repo.
- `~/.env` keys: `input_root`, `output_root`, `code_root`, `model_outputs_path` (values are machine-specific, see `.env_template`).
- Dataset: external shared folder linked in `Data.md`.

## Assumptions / open questions
- [ ] Upgraded stack is verified on CPU with random data only; GPU, GDAL and real-data runs are pending (T-001…T-003).
- [ ] Whether models saved as `.h5` under TF 2.5 still load in TF 2.15 (T-003).
