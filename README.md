# Road Extraction with Sentinel-2 Imagery & GPS Trajectory with Deep Learning

This repository contains the code of a PhD study on automatic road network extraction from **low-resolution satellite imagery fused with GPS trajectory data**. Roads are predicted as a binary segmentation mask from 10 m Sentinel-2 bands (B02, B03, B04 and B08), combined with trajectory rasters derived from GPS data (trajectory count/frequency, average and maximum speed). Experiments cover two study areas, **Istanbul** and **Montreal**, separately and together.

- **Fusion strategies:** *early fusion*, where trajectory rasters are added as extra input channels, and *late fusion* (`*_traj_type1` / `*_traj_type2` models), where satellite and trajectory data go through separate streams that are merged with average, maximum, multiply, concatenation or multi-head attention layers.
- **Models:** U-Net, ResUNet and D-LinkNet (plus light variants and SRCNN + U-Net), and the transformer-based TransUNet and Swin-UNet.
- **Loss functions:** binary cross-entropy, Dice, Dice + binary cross-entropy, focal and mean squared error.
- **Evaluation:** region-based metrics (IoU, precision, recall, F1, accuracy) and boundary-based IoU.

The published results show that ResUNet performs better than U-Net and D-LinkNet for this task, and that the approach outperforms the benchmark study that uses low-resolution Sentinel-2 data.

## Publication, thesis and data

- **Publication:** Gengec, N. E., Tari, E., & Bagci, U. (2024). AI powered road network prediction with fused low-resolution satellite imagery and GPS trajectory. *Earth Science Informatics*, 17(2), 1013–1029. [https://doi.org/10.1007/s12145-023-01201-6](https://link.springer.com/article/10.1007/s12145-023-01201-6) (open-access preprint: [arXiv:2312.17040](https://arxiv.org/abs/2312.17040))
- **PhD thesis:** this work is also the author's PhD thesis, which can be downloaded from the [YÖK National Thesis Center](https://tez.yok.gov.tr/UlusalTezMerkezi/TezGoster?key=1pwTzRXnomYf6jwqVORfUdLnWxqaR5BcTq8lSEZL_REHkBhgmFrD3IbGp33uHMPG).
- **Dataset:** Sentinel-2 and trajectory data for Istanbul and Montreal on Hugging Face: [negengec/sentinel_trajectory_ist_mont](https://huggingface.co/datasets/negengec/sentinel_trajectory_ist_mont) (CC-BY-4.0). See also [Data.md](./Data.md).

If you use this code or data, please cite:

```bibtex
@article{gengec2024road,
  title   = {AI powered road network prediction with fused low-resolution satellite imagery and GPS trajectory},
  author  = {Gengec, Necip Enes and Tari, Ergin and Bagci, Ulas},
  journal = {Earth Science Informatics},
  volume  = {17},
  number  = {2},
  pages   = {1013--1029},
  year    = {2024},
  doi     = {10.1007/s12145-023-01201-6}
}
```

## Repository guide

- Link to TRUBA VPN connection setup and major SLURM functions are [here](truba_docs/README.md). 
- Link to runtime scripts definitions and options [here](./scripts/README.md). 
- Use [.env_template](./.env_template) to configure input/output paths.
- Truba python environment installation instructions are [here](./truba_docs/TRUBA_ENV_INST.md)
- [Data](./Data.md)

## Working with agents

This repository uses the [Agent Seed](https://github.com/nagellette/agents-seed) template (see [docs/AGENT_SEED.md](./docs/AGENT_SEED.md)).
- Entry points: [AGENTS.md](./AGENTS.md) (Codex) and [CLAUDE.md](./CLAUDE.md) (Claude Code), with shared rules in [docs/AGENT_GUIDELINES.md](./docs/AGENT_GUIDELINES.md).
- Project state, tasks and decisions: [vault/](./vault/INDEX.md).
- Checks: `python3 scripts/check_context.py`, `python3 -m unittest discover -s tests -p 'test_*.py'`, and the model smoke test `python3 -m unittest tests.test_models_smoke -v` (needs the full environment).

> ⚠️ **Security upgrade disclaimer**
>
> The dependencies of this repository were upgraded to resolve the GitHub security (Dependabot) alerts. Main changes:
> - TensorFlow 2.5 → 2.15.1 (`tensorflow-gpu` removed), Python 3.8 → 3.11, CUDA 11.2 → 12.2 (installed via `tensorflow[and-cuda]`)
> - GDAL 3.3.2 → 3.13.1, Pillow 9.1.0 → 13.0.0, numpy 1.19.5 → 1.26.4, python-dotenv 0.20.0 → 1.2.2, along with compatible versions of pandas, scikit-image, matplotlib and tensorflow-addons
> - Small code changes to stay compatible: `tensorflow.python.keras` imports replaced with `tensorflow.keras`, and the SGD/Adam optimizers switched to `tf.keras.optimizers.legacy`
>
> These changes were made only for security reasons. They were checked with a CPU-only test on random data, but they were **not** tested on GPU, with real data, with GDAL, or with models saved under the previous version. **The upgraded code is not guaranteed to work without issues.**
>
> The latest working version from before the upgrade (which still contains the known vulnerabilities) is commit [`105f842`](https://github.com/nagellette/sentinel_traj_nn/tree/105f84207a6ec5534abd0e73ff6177e2c6c0e716) (2023-11-07). Use it to reproduce earlier results, together with its [TRUBA environment instructions](https://github.com/nagellette/sentinel_traj_nn/blob/105f84207a6ec5534abd0e73ff6177e2c6c0e716/truba_docs/TRUBA_ENV_INST.md).
