# Architecture

Status: **documented 2026-10-26** from the existing code

## System overview
- `run_model.py` — training entry point: reads configs, builds generators and the model, trains, saves the model and logs, optionally predicts the test set.
- `models/model_repository.py` — `ModelRepository(model_name, dim, input_channels, batch_size, ...)` builds and compiles one of 19 models: `unet`, `unetlight`, `srcnn_unet`, `resunet`, `resunetlight`, `dlinknet`, `transunet`, `swin_unet`, `test_model`, and late-fusion variants `{unet,resunet,dlinknet,transunet,swin_unet}_traj_type{1,2}`. TransUNet/Swin-UNet use `keras_unet_collection.base`.
- `utils/` — `InputReader` (JSON configs), `TrainTestValidateSplitter` (patch grid and split), `RasterDataGenerator` (`tf.keras.utils.Sequence`), `ConstructLossFunction`, `ConstructOptimizer`, `get_fusion_layer`, `custom_losses.DiceLoss`, `custom_metrics`, `custom_callbacks.TimeKeeper`.
- `run_experiments.py` / `run_batch_analysis.py` — fill SLURM templates in `scripts/` from CSV tables and submit them.
- `run_analysis.py`, `run_analysis_rgb.py`, `analysis/`, `batch_analysis/` — evaluate saved models and aggregate results.

## Data flow and interfaces
1. `model_config_files/model_config_*.json` — training parameters (batch size, `image_dim`, optimizer, loss, epochs, split, augmentation, overlap, `fusion_type`, ...).
2. `model_config_files/input_files_*.json` — raster list per area: `[file, normalization type, min, max]` entries plus `label_file`, under `working_path` relative to `input_root`.
3. GDAL opens the GeoTIFFs → `TrainTestValidateSplitter` builds patch coordinates (with rotation augmentation) → `RasterDataGenerator` yields `(batch, H, W, channels)` inputs and one-hot labels.
4. Model outputs `(batch, H, W, 2)` sigmoid probabilities.
5. Outputs per run under `output_root`: `*_log.csv` (renamed `*_log_completed.csv` on success), `timekeeper.csv`, SavedModel dir `*_model`, `*_model.h5`, optional test predictions and `test_evaluation.csv`.

## Invariants and failure modes
- Labels have 2 one-hot channels (`to_categorical`, `num_classes=2`); losses and metrics assume this shape.
- `*_traj_type*` models split the input and treat the **last channel** as the trajectory raster.
- `DiceLoss` and `MeanIoU` loop over `batch_size`; the fixed batch size in `Input(batch_size=...)` must match the generator.
- `fusion_type` ∈ `average | maximum | multiply | concat | multiheadattention` (late-fusion models only).
- Optimizers come from `tf.keras.optimizers.legacy` (ADR-2026-002).

## Environments and deployment
- TRUBA: conda env `tensorflow_2.15` per `truba_docs/TRUBA_ENV_INST.md`; jobs from `scripts/run_remote_template.sh` (SLURM, 1 GPU).
- Local: same requirements; macOS has no CUDA, so use plain `tensorflow==2.15.1`.

## Open architectural questions
- Migration path to Keras 3 (replace `tensorflow-addons` losses/F1 metric and `keras-unet-collection` transformer blocks) — T-005.
