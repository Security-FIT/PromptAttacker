# Jailbreak Attack and Defense Experiments

This repository contains an experimental framework for generating jailbreak
attack prompts, running them against local large language models, applying
defenses, and evaluating the resulting responses.

The maintained implementation lives in `prompt_attacker/`. For detailed usage
instructions, see [prompt_attacker/README.md](prompt_attacker/README.md).

## Repository Layout

- `prompt_attacker/attacks/`  
  Individual jailbreak attack implementations.

- `prompt_attacker/defense/`  
  Baseline defenses, prompt-rewrite defenses, and rule-tree utilities.

- `prompt_attacker/evaluate/`  
  Evaluation scripts and selected examples for defense training.

- `prompt_attacker/scripts/`  
  Small runners used by the orchestrator inside PBS jobs.

- `prompt_attacker/run_orchestrator.py`  
  Main orchestration CLI. It creates PBS job scripts and submits them with
  `qsub` unless `dry_run` is enabled.

- `prompt_attacker/config_orchestrator.yaml`  
  Main configuration file for paths, model selection, backend selection,
  attacks, defenses, and evaluation.

## Quick Start

From the cluster environment:

```bash
module add mambaforge
mamba activate /storage/brno2/home/xkaska01/.conda/envs/diplomka
cd /storage/brno2/home/xkaska01/master/prompt_attacker
```

List prepared attack JSON files:

```bash
python3 run_orchestrator.py --config config_orchestrator.yaml --list-attacks
```

Create or submit batch attack jobs:

```bash
python3 run_orchestrator.py --config config_orchestrator.yaml --attack-batch
```

Create or submit one selected attack for `target_model`:

```bash
python3 run_orchestrator.py --config config_orchestrator.yaml --attack-single
```

Run defenses:

```bash
python3 run_orchestrator.py --config config_orchestrator.yaml --defense ea
python3 run_orchestrator.py --config config_orchestrator.yaml --defense rallm
python3 run_orchestrator.py --config config_orchestrator.yaml --defense llamaguard
python3 run_orchestrator.py --config config_orchestrator.yaml --defense safeguard
```

## Recommended Safe Test

Before submitting many PBS jobs, set this in
`prompt_attacker/config_orchestrator.yaml`:

```yaml
dry_run: true
target_model: "falcon3:3b"
single_attack: "_1_cypher"
```

Then run:

```bash
python3 run_orchestrator.py --config config_orchestrator.yaml --attack-single
```

Inspect the generated job script under `results_dir/jobs/`. If it is correct,
set `dry_run: false` and run the command again.

## Backends

The project supports two inference backends:

- `use_ollama: true`  
  Use an Ollama model through the local Ollama HTTP API.

- `use_ollama: false`  
  Use a local model directory through vLLM.

For the current environment, vLLM jobs request `gpu_cap=cuda80` in
`prompt_attacker/scripts/job_templates.py`. This avoids Blackwell `sm_120`
GPUs that are not supported by the installed PyTorch build.

## Documentation

The detailed project documentation is maintained in:

```text
prompt_attacker/README.md
```

The main Python entry points include Doxygen-style docstrings with `@brief`,
`@param`, and `@return` tags so generated API documentation can be added later.
