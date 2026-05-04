# Jailbreak Attack and Defense Orchestrator

This directory contains the implementation used to generate, run, defend, and
evaluate jailbreak attack datasets against local LLMs. The main entry point is
`run_orchestrator.py`, which creates PBS job scripts for MetaCentrum and submits
them with `qsub`.

The project supports two inference backends:

- `use_ollama: true` uses a local Ollama server and model name such as
  `gemma3:12b`.
- `use_ollama: false` uses a local model directory through vLLM.

For the current environment, vLLM jobs should run on GPUs compatible with the
installed PyTorch build. The PBS templates request `gpu_cap=cuda80` to avoid
Blackwell `sm_120` GPUs, which are not supported by the current PyTorch
installation.

## Main Files

- `run_orchestrator.py`  
  Creates PBS jobs for attacks, defenses, defense training, and evaluation.

- `config_orchestrator.yaml`  
  Central configuration file with model paths, dataset paths, backend selection,
  target model, selected attack, and defense parameters.

- `scripts/job_templates.py`  
  PBS shell script templates. This is where GPU memory, walltime, conda
  activation, `PYTHONPATH`, Ollama startup, and `gpu_cap` are configured.

- `scripts/only_attack.py`  
  Runs one prepared attack JSON file sequentially.

- `scripts/only_attack_batch.py`  
  Runs one prepared attack JSON file in batches. This is preferred for larger
  datasets and vLLM inference.

- `scripts/only_defense_batch.py`  
  Applies one selected defense to one prepared attack JSON file.

- `defense/`  
  Contains baseline defenses, the lightweight EA defense, rule training, rule
  application, and vocabulary utilities.

- `evaluate/`  
  Contains evaluation scripts and selected examples used by defense training.

- `dataset/oponent_show/`  
  Small prepared attack dataset intended for demonstration and opponent review.

- `results/`  
  Output directory for generated jobs and model responses.

## Environment

Typical setup on MetaCentrum:

```bash
module add mambaforge
mamba activate /storage/brno2/home/xkaska01/.conda/envs/diplomka
cd /storage/brno2/home/xkaska01/master/my_implementation
```

The generated PBS scripts perform the same environment setup automatically.

## Configuration

Important keys in `config_orchestrator.yaml`:

- `models_dir`  
  Directory containing model folders. Batch modes iterate over these folders.

- `results_dir`  
  Base output directory for generated job scripts and JSON responses.

- `dataset_to_attack_path`  
  Directory with prepared attack JSON files, for example `_1_cypher.json`.

- `dataset_to_train_attack_path`  
  Legacy dataset for generating attack JSON files. Normal orchestrator attack
  runs use prepared JSON files from `dataset_to_attack_path`.

- `use_ollama`  
  Selects backend. `false` means local vLLM; `true` means Ollama.

- `dry_run`  
  If `true`, job scripts are created but not submitted. If `false`, the
  orchestrator calls `qsub`.

- `target_model`  
  Single model used by `--attack-single` and the default JSON attack mode.

- `single_attack`  
  Attack JSON selected by `--attack-single`, for example `_1_cypher`.

- `vllm_use_v1`  
  If `false`, generated vLLM commands include `VLLM_USE_V1=0`.

## Common Commands

List prepared attacks:

```bash
python3 run_orchestrator.py --config config_orchestrator.yaml --list-attacks
```

Create/submit batch attack jobs for all model folders in `models_dir`:

```bash
python3 run_orchestrator.py --config config_orchestrator.yaml --attack-batch
```

Create/submit a single attack job for `target_model` and `single_attack`:

```bash
python3 run_orchestrator.py --config config_orchestrator.yaml --attack-single
```

The no-action default is equivalent to the single-model JSON attack path, so
this also uses `dataset_to_attack_path` and `single_attack`:

```bash
python3 run_orchestrator.py --config config_orchestrator.yaml --interactive
```

Create/submit defense jobs:

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

## Recommended Test Flow

Before submitting many jobs, set:

```yaml
dry_run: true
target_model: "falcon3:3b"
single_attack: "_1_cypher"
```

Then run:

```bash
python3 run_orchestrator.py --config config_orchestrator.yaml --attack-single
```

Check the generated job script in `results_dir/jobs/`. If it looks correct, set:

```yaml
dry_run: false
```

and run the command again.

## Output Layout

For `--attack-single`, outputs go directly to:

```text
results_dir/<attack>.json
```

For `--attack-batch`, outputs are grouped by model:

```text
results_dir/<model>/<attack>.json
results_dir/<model>/jobs/job_onlyattackbatch_<attack>.sh
```

Defense jobs write JSON files with the same base filename and add a `defense`
field to each item.

## CUDA and GPU Notes

If a job fails with:

```text
NVIDIA RTX PRO 6000 Blackwell ... sm_120 is not compatible with the current PyTorch installation
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

then the job received a GPU that is too new for the installed PyTorch/vLLM
build. The current PBS templates request:

```bash
gpu_cap=cuda80
```

to avoid those nodes. If the cluster syntax changes, update
`scripts/job_templates.py`.

Useful diagnostics inside a GPU job:

```bash
nvidia-smi
python3 -c "import torch; print(torch.cuda.get_device_name(0)); print(torch.cuda.get_device_capability(0)); print(torch.cuda.get_arch_list())"
```

The current PyTorch build must list the GPU architecture that the job received.

## Known Limitations

- `--run-pipeline` expects `scripts/run_pipeline.py`, but the historical
  `run_pipeline.py` was removed during the orchestrator migration.
- Some model names contain characters such as `:` and `-`. They are valid as
  model folder names, but external libraries may print warnings when they try to
  interpret them as Python module names.
- Large vLLM models require compatible GPU memory and a supported CUDA
  architecture.
