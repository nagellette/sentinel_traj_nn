# Architecture and product decision log

Write entries for consequential choices, not routine implementation details. Never silently rewrite history: mark old decisions superseded and link the replacement.

## Decision template

### ADR-YYYY-NNN — Title
- **Date:** YYYY-MM-DD
- **Status:** Proposed | Accepted | Superseded
- **Context:** Problem and requirements.
- **Options:** Alternatives considered.
- **Decision:** Chosen approach.
- **Rationale / evidence:** Why, with sources/tests when available.
- **Consequences:** Trade-offs, migration or revisit triggers.
- **Links:** Related issues, code, plans, or superseding decision.

## Recorded decisions

### ADR-2026-001 — Upgrade to TensorFlow 2.15.1 (last Keras 2 release)
- **Date:** 2026-10-26
- **Status:** Accepted
- **Context:** ~300 open Dependabot alerts in `requirements.txt` (tensorflow, tensorflow-gpu, Pillow, GDAL, numpy, python-dotenv). TensorFlow fixes require >= 2.12.1.
- **Options:** (a) TF 2.15.1 with Keras 2; (b) latest TF with Keras 3.
- **Decision:** (a). Drop `tensorflow-gpu`, use `tensorflow[and-cuda]==2.15.1`; Python 3.11 (Pillow 13 needs >= 3.10, TF 2.15 supports <= 3.11).
- **Rationale / evidence:** Keras 3 breaks `tensorflow-addons` (focal/contrastive losses, F1), `keras-unet-collection` (TransUNet/Swin-UNet) and extension-less `model.save()`. CPU smoke test passed 71/71 cases on 2026-10-26 (`tests/test_models_smoke.py`).
- **Consequences:** TF 2.15 and tensorflow-addons are end-of-life; new advisories against them will not be fixed. Revisit with T-005.
- **Links:** commit `a2a2267`, `vault/worklogs/2026-10-26-2243-security-upgrade.md`.

### ADR-2026-002 — Use legacy Keras optimizers
- **Date:** 2026-10-26
- **Status:** Accepted
- **Context:** Since TF 2.11 the new `tf.keras.optimizers.SGD` rejects `decay`, which the configs use.
- **Decision:** `ConstructOptimizer` returns `tf.keras.optimizers.legacy.SGD` / `legacy.Adam` with `learning_rate=`.
- **Rationale / evidence:** Keeps TF 2.5 training behaviour comparable with earlier results.
- **Consequences:** Legacy optimizers do not exist in Keras 3; part of T-005.
- **Links:** `utils/construct_optimizer.py`.

### ADR-2026-003 — Pin numpy 1.26.4 in the conda GDAL install
- **Date:** 2026-10-26
- **Status:** Accepted
- **Context:** conda-forge GDAL 3.13.1 builds are compiled against numpy 2 but declare `numpy >=1.23,<3`; TF 2.15 needs numpy < 2.
- **Decision:** `conda install -c conda-forge gdal=3.13.1 numpy=1.26.4` in `truba_docs/TRUBA_ENV_INST.md`.
- **Rationale / evidence:** Prevents conda from installing numpy 2 that pip would then downgrade. Build metadata checked on anaconda.org on 2026-10-26.
- **Consequences:** Not yet verified on TRUBA (T-001).
