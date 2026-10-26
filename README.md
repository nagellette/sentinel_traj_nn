#### Main Project for Road extraction with Sentinel & Trajectory with Deep Learning

> [!WARNING]
> **Security upgrade disclaimer**
>
> The dependencies of this repository were upgraded to resolve the GitHub security (Dependabot) alerts. Main changes:
> - TensorFlow 2.5 → 2.15.1 (`tensorflow-gpu` removed), Python 3.8 → 3.11, CUDA 11.2 → 12.2 (installed via `tensorflow[and-cuda]`)
> - GDAL 3.3.2 → 3.13.1, Pillow 9.1.0 → 13.0.0, numpy 1.19.5 → 1.26.4, python-dotenv 0.20.0 → 1.2.2, along with compatible versions of pandas, scikit-image, matplotlib and tensorflow-addons
> - Small code changes to stay compatible: `tensorflow.python.keras` imports replaced with `tensorflow.keras`, and the SGD/Adam optimizers switched to `tf.keras.optimizers.legacy`
>
> These changes were made only for security reasons. They were checked with a CPU-only test on random data, but they were **not** tested on GPU, with real data, with GDAL, or with models saved under the previous version. **The upgraded code is not guaranteed to work without issues.**
>
> The latest working version from before the upgrade (which still contains the known vulnerabilities) is commit [`105f842`](https://github.com/nagellette/sentinel_traj_nn/tree/105f84207a6ec5534abd0e73ff6177e2c6c0e716) (2023-11-07). Use it to reproduce earlier results, together with its [TRUBA environment instructions](https://github.com/nagellette/sentinel_traj_nn/blob/105f84207a6ec5534abd0e73ff6177e2c6c0e716/truba_docs/TRUBA_ENV_INST.md).

- Link to TRUBA VPN connection setup and major SLURM functions are [here](truba_docs/README.md). 
- Link to runtime scripts definitions and options [here](./scripts/README.md). 
- Use [.env_template](./.env_template) to configure input/output paths.
- Truba python environment installation instructions are [here](./truba_docs/TRUBA_ENV_INST.md)
- [Data](./Data.md)

#### Working with agents

This repository uses the [Agent Seed](https://github.com/nagellette/agents-seed) template (see [docs/AGENT_SEED.md](./docs/AGENT_SEED.md)).
- Entry points: [AGENTS.md](./AGENTS.md) (Codex) and [CLAUDE.md](./CLAUDE.md) (Claude Code), with shared rules in [docs/AGENT_GUIDELINES.md](./docs/AGENT_GUIDELINES.md).
- Project state, tasks and decisions: [vault/](./vault/INDEX.md).
- Checks: `python3 scripts/check_context.py`, `python3 -m unittest discover -s tests -p 'test_*.py'`, and the model smoke test `python3 -m unittest tests.test_models_smoke -v` (needs the full environment).

 