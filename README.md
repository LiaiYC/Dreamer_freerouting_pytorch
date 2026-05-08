# DreamerV3 x Freerouting (PyTorch/JAX)

This project integrates the DreamerV3 reinforcement learning framework with the Freerouting PCB autorouter. The goal is to learn a generalizable set of routing parameters that improves routing quality across different PCB (`.dsn`) boards.

Instead of directly predicting each trace, this project formulates Freerouting parameter tuning as a reinforcement learning decision problem.

> **Patent Pending Notice**  
> Parts of this project, including neural network training workflows, model configurations, and key hyperparameters, are related to pending patent applications. These implementation details are currently withheld and will be released after the patent process is completed.

## Highlights

- Uses board-level features from `DSN` files as observations for learning routing decisions.
- Calls Freerouting CLI, then parses `.ses` and DRC JSON outputs.
- Supports two modes:
  - `jax`: full DreamerV3 training workflow (partially withheld).
  - `pytorch`: quick benchmark and RL probe (partially withheld).
- Includes preprocessing scripts to build manifest files (`.jsonl`) for faster experiments.

## Project Structure

```text
Dreamer_freerouting_pytorch/
|-- dreamerv3/                 # DreamerV3 code and integration scripts
|   |-- dreamerv3/             # agent / configs / main
|   |-- embodied/              # env and training framework
|   |   `-- envs/freerouting.py # Freerouting RL environment
|   `-- scripts/               # training, benchmark, data-prep scripts
|-- freerouting/               # Freerouting source code (Java)
|-- DSN/                       # PCB board data (.dsn)
|-- logdir/                    # training and experiment outputs
|-- .upstream_dreamerv3/       # upstream DreamerV3 reference snapshot
`-- chapter_*.md/.tex          # research and design documents
```

## Method Summary

### 1) Observation

`embodied/envs/freerouting.py` converts each board into an 8-dimensional feature vector (with `log1p` scaling):

- file size
- number of nets / components / pins
- number of layers / keepouts
- board width and height (mm)

### 2) Action

> This section is patent-sensitive. Only high-level information is provided.

The policy outputs a routing control vector. The environment maps it into valid Freerouting settings to influence behavior such as exploration intensity, routing preference, and convergence strategy. Exact parameter names, mapping ranges, and constraints are currently withheld.

### 3) Reward

> This section is patent-sensitive. Only high-level information is provided.

The reward is a multi-objective quality signal combining routability, design-rule compliance, and routing cost. Exact weights, reward/penalty composition, and exception handling logic are currently withheld.

## DSN Data Usage (Training / Testing)

The `.dsn` files under `DSN/` are board-level data sources used for:

- Training Data
  - To expose the model to diverse board layouts and routing conditions during training.
- Testing Data
  - To evaluate generalization on held-out or unseen boards.

Recommended practice: split `DSN/` into fixed train/test subsets (for example, 80/20), and keep seed and board lists fixed for reproducibility and fair comparison.

## Requirements

- Python `3.11+`
- Java Runtime (recommended: 21)
- Freerouting `.jar` (release build is acceptable)

## Installation

From the project root:

```bash
cd dreamerv3
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# macOS/Linux
# source .venv/bin/activate
```

### Dependencies for Full JAX Training

```bash
pip install -U -r requirements.txt
```

### Dependencies for PyTorch Benchmark / Probe

```bash
pip install -U -r requirements-pytorch.txt
```

## Quick Start

### A. Build a Data Manifest (Optional but Recommended)

```bash
python scripts/prepare_freerouting_data.py \
  --input-dir ../DSN \
  --output data/freerouting/boards.jsonl \
  --patterns "*.dsn,*.DSN" \
  --shuffle --seed 0
```

### B. PyTorch Quick Benchmark

> Withheld due to pending patent application (to be released later).

### C. GPU / Output Verification

> Withheld due to pending patent application (to be released later).

### D. PyTorch RL Probe (Real Freerouting Interaction)

> Withheld due to pending patent application (to be released later).

### E. JAX DreamerV3 Full Training

> Withheld due to pending patent application (to be released later).

## Key Configurations (`dreamerv3/configs.yaml`)

Details related to neural-network training strategy and key hyperparameters are withheld due to pending patent application.

Currently public content focuses on data preprocessing and environment integration.

## Planned Post-Patent Disclosure Scope

After patent procedures are completed, the following items will be progressively released:

1. Full Action parameter definitions
2. Action normalization and inverse-mapping formulas (including boundary/clipping rules)
3. Full Reward equation and per-term weights
4. Reward shaping strategy and failure-case handling rules
5. Reproducible training command sets (JAX / PyTorch)
6. Key hyperparameters and ablation settings
7. Representative experiment results and comparison scripts

## Outputs and Monitoring

- Default outputs are saved under `logdir/...`
- Metrics include:
  - Task quality: `log/length_mm`, `log/vias`, `log/violations*`, `log/unconnected`, `log/completion`
  - Learning signals: `train-loss-*`, `report-*`
  - System usage: CPU/RAM/GPU

Scope viewer:

```bash
pip install -U scope
python -m scope.viewer --basedir ~/logdir --port 8000
```

## FAQ

- `Missing --jar or FREEROUTING_JAR`
  - A Freerouting jar is required for JAX training or RL probe.
- `No DSN boards found...`
  - Check `--data-dir` / `--manifest` paths and ensure `.dsn/.DSN` extensions.
- `--device gpu` falls back to CPU
  - Use `verify_torch_benchmark.py` to check `resolved_device` and `cuda_available`.

## Notes

- `logdir/` and large `.ses/.json` outputs are usually not suitable for direct Git commits.
- `freerouting/` and `.upstream_dreamerv3/` are upstream/third-party code; integration logic is mainly in `dreamerv3/embodied/envs/freerouting.py` and `dreamerv3/scripts/`.

## Acknowledgements

- DreamerV3: https://github.com/danijar/dreamerv3
- Freerouting: https://github.com/freerouting/freerouting
