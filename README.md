# Jailbreak Attack and Defense Experiments

An experimental Python framework for preparing jailbreak prompts, running them
against local large language models, applying defenses, and evaluating model
responses on an HPC/PBS environment.

The maintained implementation lives in `prompt_attacker/`. The main entry point
is `prompt_attacker/run_orchestrator.py`, which reads
`prompt_attacker/config_orchestrator.yaml`, generates PBS job scripts, and
submits them with `qsub` unless `dry_run` is enabled.

## Table of Contents

- [About](#about)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Models](#models)
- [Configuration](#configuration)
- [Basic Usage](#basic-usage)
- [Project Structure](#project-structure)
- [Outputs](#outputs)
- [Responsible Use](#responsible-use)
- [Contributing](#contributing)
- [Security](#security)
- [Citation](#citation)
- [License](#license)

## About

This repository is intended for controlled LLM safety experiments. It supports:

- prepared jailbreak attack datasets stored as JSON files;
- single-model and batch-model attack execution;
- Ollama and vLLM inference backends;
- baseline and custom defense workflows;
- PBS job generation for MetaCentrum-style clusters;
- evaluation scripts for generated responses.

The current recommended workflow uses prepared JSON attack files from
`prompt_attacker/dataset/` and model directories under
`prompt_attacker/models/`.

## Prerequisites

You need:

- access to a PBS cluster with `qsub`;
- Python 3.12 is recommended; the current dependency set was tested with
  Python 3.12;
- `mamba` or `conda`;
- CUDA-capable GPU nodes for vLLM runs;
- local model files for vLLM, or an Ollama installation and pulled Ollama
  models;

On MetaCentrum, the expected module setup is:

```bash
module add mambaforge
```

For interactive vLLM debugging, request a GPU allocation first:

```bash
qsub -I -l walltime=8:0:0 -q default@pbs-m1.metacentrum.cz -l select=1:ncpus=1:ngpus=1:mem=200gb:gpu_mem=60gb:scratch_local=400gb
```

The generated PBS templates already request compatible GPU nodes with
`gpu_cap=cuda80`.

## Setup

Create a fresh environment from the repository root:

```bash
module add mambaforge
mamba create -n jailbreak-exp python=3.12
mamba activate jailbreak-exp
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cd prompt_attacker
```

`requirements.txt` lists the direct project dependencies instead of a full
environment dump. It includes the orchestration tools, vLLM/Ollama backends,
PyTorch/Transformers, and the attack/defense utilities used by this codebase.

For vLLM, make sure the installed PyTorch build supports the GPU architecture
allocated by PBS. If a job fails with a CUDA architecture error, use the
provided PBS templates or request a compatible GPU capability.


## Models

Models are not part of the repository. They must be downloaded or prepared
before running attacks.

### vLLM backend

Use this when `use_ollama: false` in `config_orchestrator.yaml`.

Place Hugging Face-compatible model folders under:

```text
prompt_attacker/models/
```

The folder name should match the model name used in config. For example:

```text
prompt_attacker/models/falcon3:3b
prompt_attacker/models/gemma3:12b
prompt_attacker/models/llama2:7b
```

Example download pattern:

```bash
cd <repo>/prompt_attacker
mkdir -p models
huggingface-cli download <hf-org>/<hf-model> --local-dir models/<model-name-used-in-config>
```

Then set:

```yaml
use_ollama: false
models_dir: "models"
target_model: "falcon3:3b"
```

For `--attack-single`, `target_model` selects one model. For `--attack-batch`,
all model directories in `models_dir` are used.

### Ollama backend

Use this when `use_ollama: true`.

Pull the model before running interactive jobs:

```bash
ollama pull gemma3:12b
```

For generated PBS jobs, the Ollama template starts the local Ollama server and
pulls the configured model automatically for workflows that use that template.
For manual interactive runs, start or verify Ollama yourself:

```bash
ollama serve
```

Then set:

```yaml
use_ollama: true
target_model: "gemma3:12b"
ollama_bin: "ollama"
```

## Configuration

The main configuration file is:

```text
prompt_attacker/config_orchestrator.yaml
```

Important keys:

```yaml
models_dir: "models"
local_model_path: "models"
results_dir: "results"
dataset_to_attack_path: "dataset"

use_ollama: false
dry_run: true
module_command: "module add mambaforge"
conda_env: "jailbreak-exp"
ollama_bin: "ollama"

target_model: "falcon3:3b"
single_attack: "_1_cypher"
```

Key meanings:

- `models_dir`: directory containing one subdirectory per local model.
- `local_model_path`: base model path used by single-model vLLM runs.
- `results_dir`: output directory for generated jobs and JSON results.
- `dataset_to_attack_path`: directory with prepared attack JSON files.
- `use_ollama`: `false` for vLLM/local model folders, `true` for Ollama.
- `dry_run`: `true` creates job scripts only; `false` also submits with `qsub`.
- `module_command`: shell command used by generated PBS jobs before activation.
- `conda_env`: mamba/conda environment activated inside generated PBS jobs.
- `ollama_bin`: Ollama executable used by generated Ollama/evaluation jobs.
- `target_model`: model used by `--attack-single`.
- `single_attack`: attack JSON stem used by `--attack-single`, for example
  `_1_cypher`; use `all`, `*`, or an empty value to run all prepared attacks.

Advanced users can replace `module_command` and `conda_env` with a full custom
setup block:

```yaml
job_env_setup: |
  module add mambaforge
  mamba activate jailbreak-exp
```

Before large runs, keep `dry_run: true`, inspect generated job scripts, then set
`dry_run: false` and rerun the same command.

## Basic Usage

All commands below start from:

```bash
cd <repo>/prompt_attacker
```

List available prepared attacks:

```bash
python3 run_orchestrator.py --config config_orchestrator.yaml --list-attacks
```

Create a dry-run job for one model and one attack:

```bash
python3 run_orchestrator.py --config config_orchestrator.yaml --attack-single
```

Override the selected attack without editing config:

```bash
python3 run_orchestrator.py --config config_orchestrator.yaml --attack-single --single-attack _1_cypher
```

Submit the same job for real:

```yaml
dry_run: false
```

```bash
python3 run_orchestrator.py --config config_orchestrator.yaml --attack-single
```

Run all prepared attacks for all model folders in `models_dir`:

```bash
python3 run_orchestrator.py --config config_orchestrator.yaml --attack-batch
```

Run a small action directly in the current terminal instead of creating PBS
jobs:

```bash
python3 run_orchestrator.py --config config_orchestrator.yaml --attack-single --interactive
```

For `--interactive` with `use_ollama: false`, run inside a GPU allocation. The
orchestrator checks whether the CUDA driver is visible before starting vLLM.

Run defenses:

```bash
python3 run_orchestrator.py --config config_orchestrator.yaml --defense ea
python3 run_orchestrator.py --config config_orchestrator.yaml --defense rallm
python3 run_orchestrator.py --config config_orchestrator.yaml --defense llamaguard
python3 run_orchestrator.py --config config_orchestrator.yaml --defense safeguard
```

Train and apply the rule-tree defense:

```bash
python3 run_orchestrator.py --config config_orchestrator.yaml --defense-train
python3 run_orchestrator.py --config config_orchestrator.yaml --defense-apply-rules
python3 run_orchestrator.py --config config_orchestrator.yaml --defense-train-apply
```

Generate evaluation jobs:

```bash
python3 run_orchestrator.py --config config_orchestrator.yaml --evaluate
```

Show all CLI options:

```bash
python3 run_orchestrator.py --help
```

## Project Structure

```text
requirements.txt          Curated Python dependencies for the project
prompt_attacker/
  attacks/                 Attack implementations and shared LLM wrapper
  defense/                 Baseline defenses and rule-tree defense utilities
  evaluate/                Evaluation scripts and selected examples
  scripts/                 Small runners used inside generated PBS jobs
  dataset/                 Prepared attack JSON files
  models/                  Local model folders; not committed
  results/                 Generated jobs and outputs; not committed
  run_orchestrator.py      Main orchestration CLI
  config_orchestrator.yaml Main configuration file
  steps.txt                Practical local notes for running the orchestrator
```

The detailed implementation README is maintained in
`prompt_attacker/README.md`.

## Outputs

Generated PBS scripts are written under `results_dir/jobs/` for single-model
runs. Batch runs create per-model result directories:

```text
results_dir/<model>/<attack>.json
results_dir/<model>/jobs/job_onlyattackbatch_<attack>.sh
```

For `--attack-single`, outputs go directly to:

```text
results_dir/<attack>.json
```

If `dry_run: true`, only job scripts are created. If `dry_run: false`, the
orchestrator also calls `qsub`.

## Responsible Use

This repository is intended for research and defensive evaluation of model
safety behavior. Run experiments only on models, datasets, and infrastructure
that you are authorized to use. Treat harmful generations as sensitive research
outputs and avoid publishing raw outputs unless there is a clear research need
and appropriate safeguards.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for
details.
