#!/usr/bin/env python3
"""Generate and submit PBS jobs for attack, defense, and evaluation workflows.

This is the main entry point for the implementation. The script reads
`config_orchestrator.yaml`, discovers local model directories, creates concrete
commands for the selected workflow, renders PBS scripts via
`scripts/job_templates.py`, and submits them with `qsub` unless `dry_run` is
enabled.

Typical usage:

    python3 run_orchestrator.py --config config_orchestrator.yaml --attack-batch
    python3 run_orchestrator.py --config config_orchestrator.yaml --attack-single
    python3 run_orchestrator.py --config config_orchestrator.yaml --defense ea

Batch modes discover models from `models_dir`. Single-model modes use
`target_model`. For vLLM jobs the generated command can be prefixed with
environment variables such as `VLLM_USE_V1=0`; GPU selection itself is handled in
the PBS templates.
"""

import os
import re
import shlex
import textwrap
import subprocess
import argparse
import time
import ctypes
from pathlib import Path
import yaml
from importlib.machinery import SourceFileLoader

# --- Attack modules (copied from run.py) ---
ATTACK_MODULES = [
    "attacks._1_Cypher.main",
    "attacks._2_Flip.main",
    "attacks._4_SQL_StructTransform.main",
    "attacks._5_suffix.main",
    "attacks._6_Sequential.main",
    "attacks._7_CitationBreak.main",
    "attacks._8_Bijection.main",
    "attacks._9_Dialog_completition.main",
    "attacks._10_Random_Search.main",
    "attacks._11_Pair.main",
    "attacks._12_Tap.main",
    "attacks._13_GPT4cypher.main",
    "attacks._14_MultiLang.main",
    "attacks._15_Rewrite.main",
    "attacks._16_Ica.main",
    "attacks._17_Overload.main",
    "attacks._19_Deepinception.main",
    # "attacks._20_Base.main",
    "attacks._21_Art_Prompt.main",
    "attacks._22_Renellm.main",
    "attacks._24_Autodan.main",
    "attacks._25_past.main",
    "attacks._26_Chameleon.main",
]


def substitute_model_in_string(template: str, model: str) -> str:
    """@brief Replace a `{model}` placeholder in a string.

    @param template String that may contain the `{model}` placeholder.
    @param model Model name to substitute into the template.
    @return String with `{model}` replaced, or the original value if it is None.
    """
    if template is None:
        return template
    return template.replace("{model}", model)


def build_model_specific_config(cfg: dict, model: str):
    """@brief Build model-specific paths for batch workflows.

    Batch modes use the discovered model name as `target_model`. If the base
    results directory does not contain `{model}`, results are grouped under
    `results_dir/<model>` to avoid overwriting outputs from other models.

    @param cfg Loaded orchestrator configuration.
    @param model Model folder name discovered in `models_dir`.
    @return Tuple `(local_model_path, results_dir, target_model)`.
    """
    models_dir = str(cfg.get("models_dir", cfg.get("local_model_path", cfg.get("victim_llm", ""))))
    local_model_t = str(cfg.get("local_model_path", cfg.get("victim_llm", models_dir)))
    results_dir_t = str(cfg.get("results_dir", ""))

    if "{model}" in local_model_t:
        local_model = substitute_model_in_string(local_model_t, model)
    else:
        model_dir = os.path.join(models_dir, model)
        local_model = model_dir if os.path.isdir(model_dir) else local_model_t

    if "{model}" in results_dir_t:
        results_dir = substitute_model_in_string(results_dir_t, model)
    else:
        results_dir = os.path.join(results_dir_t, model)

    return local_model, results_dir, model


def build_single_model_path(cfg: dict, target_model: str):
    """@brief Resolve the concrete model path for single-model workflows.

    @param cfg Loaded orchestrator configuration.
    @param target_model Model selected by `target_model`.
    @return Local model path used by vLLM or the configured fallback path.
    """
    models_dir = str(cfg.get("models_dir", cfg.get("local_model_path", cfg.get("victim_llm", ""))))
    local_model_t = str(cfg.get("local_model_path", cfg.get("victim_llm", models_dir)))

    if "{model}" in local_model_t:
        return substitute_model_in_string(local_model_t, target_model)

    model_dir = os.path.join(models_dir, str(target_model))
    if os.path.isdir(model_dir):
        return model_dir

    return local_model_t

# --- Utilities ---

def load_config(path: str) -> dict:
    """@brief Load an orchestrator YAML configuration file.

    @param path Path to a YAML configuration file.
    @return Parsed configuration dictionary.
    """
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def ensure_directory(path: str):
    """@brief Create a directory if it does not exist.

    @param path Directory path to create.
    """
    os.makedirs(path, exist_ok=True)


def discover_models(models_dir: str):
    """@brief Discover model names from a local models directory.

    @param models_dir Directory containing one subdirectory per local model.
    @return Sorted list of discovered model directory names.
    @throws FileNotFoundError If `models_dir` does not exist.
    """
    if not models_dir or not os.path.isdir(models_dir):
        raise FileNotFoundError(f"Models directory not found: {models_dir}")
    models = []
    for entry in os.scandir(models_dir):
        if entry.is_dir() and not entry.name.startswith(".") and entry.name != "__pycache__":
            models.append(entry.name)
    return sorted(models)


def normalize_selector(value: str):
    """@brief Normalize an attack selector for robust matching.

    @param value User-provided attack name, stem, or suffix.
    @return Lowercase alphanumeric-only selector.
    """
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def select_attack_inputs(dataset_dir: str, selected_attack=None):
    """@brief Select prepared attack JSON files for `--attack-single`.

    `selected_attack` can be a filename, stem, or loose suffix such as
    `_1_cypher` / `cypher`. If empty or `all`, all JSON files are returned.

    @param dataset_dir Directory containing prepared attack JSON files.
    @param selected_attack Optional filename, stem, suffix, `all`, or `*`.
    @return List of selected JSON file paths.
    @throws ValueError If the selector does not match any available attack.
    """
    inputs = [
        os.path.join(dataset_dir, f)
        for f in sorted(os.listdir(dataset_dir))
        if f.lower().endswith(".json")
    ]
    if not selected_attack or str(selected_attack).lower() in {"all", "*"}:
        return inputs

    selector = normalize_selector(os.path.splitext(os.path.basename(str(selected_attack)))[0])
    selected = []
    for path in inputs:
        stem = os.path.splitext(os.path.basename(path))[0]
        norm_stem = normalize_selector(stem)
        if norm_stem == selector or norm_stem.endswith(selector):
            selected.append(path)

    if not selected:
        available = ", ".join(os.path.splitext(os.path.basename(p))[0] for p in inputs)
        raise ValueError(f"Unknown single_attack '{selected_attack}'. Available attacks: {available}")
    return selected


def print_attack_overview(dataset_dir: str):
    """@brief Print available prepared attack JSON files and generator modules.

    @param dataset_dir Directory containing prepared attack JSON files.
    """
    print("Prepared attack JSON files in dataset_to_attack_path:")
    for path in select_attack_inputs(dataset_dir):
        print(f"  - {os.path.splitext(os.path.basename(path))[0]}")
    print("\nAttack generator modules in ATTACK_MODULES:")
    for module in ATTACK_MODULES:
        print(f"  - {module}")


def resolve_path(base_dir: str, value):
    """@brief Resolve a config path relative to the implementation directory.

    @param base_dir Absolute path to `my_implementation`.
    @param value Path value from configuration.
    @return Absolute path, relative path joined to `base_dir`, or None.
    """
    if value is None:
        return None
    path = str(value)
    if os.path.isabs(path):
        return path
    return os.path.join(base_dir, path)


def env_value(value):
    """@brief Convert config values to shell-friendly environment strings.

    @param value Configuration value.
    @return String representation suitable for environment variables, or None.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def quote_cmd(cmd_list):
    """@brief Shell-quote a command represented as a list.

    @param cmd_list Command and arguments.
    @return Safely quoted command string.
    """
    return " ".join(shlex.quote(str(x)) for x in cmd_list)


def env_prefix(env_overrides):
    """@brief Build a shell environment prefix from key-value pairs.

    @param env_overrides Environment variable overrides.
    @return Prefix such as `KEY=value OTHER=value`.
    """
    parts = []
    for key, value in env_overrides.items():
        value = env_value(value)
        if value is not None:
            parts.append(f"{key}={shlex.quote(value)}")
    return " ".join(parts)


def with_env_command(cmd_list, env_overrides):
    """@brief Combine environment overrides with a shell-quoted command.

    @param cmd_list Command and arguments.
    @param env_overrides Environment variable overrides.
    @return Command string prefixed with environment assignments.
    """
    prefix = env_prefix(env_overrides)
    cmd = quote_cmd(cmd_list)
    return f"{prefix} {cmd}" if prefix else cmd


def merge_env_overrides(*overrides):
    """@brief Merge multiple environment override dictionaries.

    Later dictionaries override earlier values.

    @param overrides Environment dictionaries.
    @return Merged environment override dictionary.
    """
    merged = {}
    for env in overrides:
        if env:
            merged.update({k: v for k, v in env.items() if v is not None})
    return merged


def template_multiline_command(cmd: str):
    """@brief Indent multiline commands for insertion into PBS templates.

    @param cmd Command string that may contain newlines.
    @return Command string with continuation lines indented.
    """
    return cmd.replace("\n", "\n        ")


def is_true(value) -> bool:
    """@brief Interpret common truthy configuration values.

    @param value Value to interpret.
    @return True if the value is one of `1`, `true`, `yes`, `y`, or `on`.
    """
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def normalize_cuda_visible_devices(value):
    """@brief Convert CUDA UUID selectors to numeric indexes.

    Some schedulers expose GPUs as UUIDs such as `GPU-...`. Older vLLM versions
    expect numeric indices and may fail while parsing UUID values.

    @param value Current `CUDA_VISIBLE_DEVICES` value.
    @return Tuple `(normalized_value, changed)`.
    """
    if not value:
        return value, False

    devices = [part.strip() for part in str(value).split(",") if part.strip()]
    if not devices or not any(device.startswith("GPU-") for device in devices):
        return value, False

    normalized = ",".join(str(index) for index, _device in enumerate(devices))
    return normalized, True


def vllm_env_overrides(cfg: dict, use_ollama):
    """@brief Build environment overrides for vLLM runs.

    @param cfg Loaded orchestrator configuration.
    @param use_ollama Backend flag. Truthy values disable vLLM overrides.
    @return Environment overrides for generated commands or interactive runs.
    """
    if is_true(use_ollama):
        return {}

    env = {}
    if not is_true(cfg.get("vllm_use_v1", False)):
        env["VLLM_USE_V1"] = "0"

    if is_true(cfg.get("normalize_cuda_visible_devices", True)):
        current_cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        normalized, changed = normalize_cuda_visible_devices(current_cuda_devices)
        if changed:
            env["CUDA_VISIBLE_DEVICES"] = normalized

    return env


def cuda_driver_available() -> bool:
    """@brief Check whether the CUDA driver library is visible.

    @return True if `libcuda.so.1` can be loaded in the current shell.
    """
    try:
        ctypes.CDLL("libcuda.so.1")
        return True
    except OSError:
        return False


def require_vllm_cuda_for_interactive(use_ollama, action_name: str) -> bool:
    """@brief Validate CUDA availability before an interactive vLLM run.

    @param use_ollama Backend flag. Ollama runs do not require this check.
    @param action_name Human-readable action name for error messages.
    @return True if the action can continue, otherwise False.
    """
    if is_true(use_ollama) or cuda_driver_available():
        return True

    print(f"[VLLM][FATAL] {action_name} is configured with use_ollama=false, so it needs vLLM + CUDA.", flush=True)
    print("[VLLM][FATAL] CUDA driver library libcuda.so.1 is not visible in this shell.", flush=True)
    print("[VLLM][FATAL] This usually means you are on a login/CPU node, not inside a GPU allocation.", flush=True)
    print("[VLLM][FIX] Start an interactive GPU job first, then run the orchestrator there:", flush=True)
    print("  qsub -I -l walltime=8:0:0 -q default@pbs-m1.metacentrum.cz -l select=1:ncpus=1:ngpus=1:mem=200gb:gpu_mem=60gb:scratch_local=400gb", flush=True)
    print("[VLLM][FIX] Or run without --interactive so the PBS job script gets a GPU allocation.", flush=True)
    return False


def format_elapsed(seconds: float) -> str:
    """@brief Format elapsed seconds as a compact human-readable string.

    @param seconds Duration in seconds.
    @return Formatted duration such as `42s`, `3m 12s`, or `1h 4m 2s`.
    """
    seconds = int(seconds)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def run_interactive(cmd_list, env_overrides=None, label=None, heartbeat_seconds=30):
    """@brief Run a command directly and print progress heartbeats.

    This mode is intended only for small debugging runs. Production runs should
    be submitted through PBS jobs.

    @param cmd_list Command and arguments to execute.
    @param env_overrides Optional environment variable overrides.
    @param label Optional label printed before the command starts.
    @param heartbeat_seconds Interval for "still running" progress messages.
    @throws subprocess.CalledProcessError If the command exits with a non-zero status.
    """
    env = os.environ.copy()
    if env_overrides:
        env.update({k: env_value(v) for k, v in env_overrides.items() if env_value(v) is not None})
    if label:
        print(label, flush=True)
    print(f"[INTERACTIVE] Command: {quote_cmd(cmd_list)}", flush=True)
    if env_overrides:
        shown = {k: env_value(v) for k, v in env_overrides.items() if k in {"CUDA_VISIBLE_DEVICES", "VLLM_USE_V1"}}
        if shown:
            print(f"[INTERACTIVE] Env: {shown}", flush=True)
    print(f"[INTERACTIVE] Started: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

    start = time.monotonic()
    proc = subprocess.Popen(cmd_list, env=env)
    heartbeat_seconds = int(heartbeat_seconds or 0)
    try:
        if heartbeat_seconds <= 0:
            return_code = proc.wait()
        else:
            while True:
                try:
                    return_code = proc.wait(timeout=heartbeat_seconds)
                    break
                except subprocess.TimeoutExpired:
                    elapsed = format_elapsed(time.monotonic() - start)
                    print(f"[INTERACTIVE] Still running after {elapsed}...", flush=True)
    except KeyboardInterrupt:
        print("[INTERACTIVE] Interrupted, terminating child process...", flush=True)
        proc.terminate()
        raise

    elapsed = format_elapsed(time.monotonic() - start)
    if return_code != 0:
        print(f"[INTERACTIVE] Failed after {elapsed} with exit code {return_code}", flush=True)
        raise subprocess.CalledProcessError(return_code, cmd_list)
    print(f"[INTERACTIVE] Finished after {elapsed}", flush=True)


# --- Orchestration logic ---

def write_and_submit_job(script_path: str, content: str, dry_run: bool):
    """@brief Write a PBS job script and optionally submit it with `qsub`.

    If `submit` is True and `dry_run` is False the script will be submitted.

    @param script_path Path where the generated shell script should be written.
    @param content Complete shell script content.
    @param dry_run If True, only write the script and skip `qsub`.
    """
    with open(script_path, 'w', encoding='utf-8') as fh:
        fh.write(content)
    os.chmod(script_path, 0o755)
    print(f"Created {script_path}")
    if not dry_run:
        try:
            res = subprocess.run(["qsub", script_path], check=True, capture_output=True, text=True)
            print(f"  -> qsub: {res.stdout.strip()}")
        except subprocess.CalledProcessError as e:
            print(f"  !! qsub error: {e.stderr.strip()}")


def main():
    """@brief Parse CLI arguments and execute the selected orchestration mode."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_orchestrator.yaml")
    parser.add_argument("--evaluate", action="store_true", help="Create eval job scripts for models in config")
    parser.add_argument("--attack-single", action="store_true", help="Create only-attack job scripts for the single model specified in `target_model` in config")
    parser.add_argument("--attack-batch", action="store_true", help="Create only-attack batch job scripts and iterate over model folders in local_model_path/models_dir")
    parser.add_argument("--single-attack", default=None, help="Attack JSON stem/file to use with --attack-single, e.g. _1_cypher. Overrides config single_attack.")
    parser.add_argument("--list-attacks", action="store_true", help="List available prepared attack JSON files and ATTACK_MODULES")
    parser.add_argument("--defense", choices=["rallm", "llamaguard", "safeguard", "ea"], help="Create defense job scripts for specified defense type") 
    parser.add_argument("--defense-train", action="store_true", help="Create/run job for defense/def.py tree-rule training")
    parser.add_argument("--defense-apply-rules", action="store_true", help="Create/run job for defense/apply_rules.py on a dataset directory")
    parser.add_argument("--defense-create-vocab", action="store_true", help="Create/run job for defense/create_vocabulary.py")
    parser.add_argument("--defense-train-apply", action="store_true", help="Create/run one job that trains a defense rule and then applies it")
    parser.add_argument("--run-pipeline", action="store_true", help="Run run_pipeline.py using the config dataset and write outputs to --pipeline-out")
    parser.add_argument("--pipeline-out", type=str, default=None, help="Directory to write pipeline outputs (overrides config.results_dir)")
    parser.add_argument("--interactive", action="store_true", help="Run selected action interactively instead of creating job scripts")
    args = parser.parse_args()

    cfg_path = Path(__file__).parent / args.config
    if not cfg_path.exists():
        print(f"Config not found: {cfg_path}")
        return

    cfg = load_config(str(cfg_path))
    # base paths and dynamic loading of job_templates from scripts/
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.environ["PYTHONPATH"] = base_dir + os.pathsep + os.environ.get("PYTHONPATH", "")
    scripts_dir = os.path.join(base_dir, 'scripts')
    jt_path = os.path.join(scripts_dir, 'job_templates.py')
    if os.path.exists(jt_path):
        job_templates = SourceFileLoader('job_templates', jt_path).load_module()
        job_template = job_templates.job_template
        batch_template = job_templates.batch_template
        results_eval_template = job_templates.results_eval_template
    else:
        # fallback to legacy import if scripts/ not present
        from job_templates import job_template, batch_template, results_eval_template
    # prefer new, more intuitive keys but fall back to legacy names for compatibility
    local_model_path = cfg.get('local_model_path', cfg.get('victim_llm'))
    models_dir = cfg.get('models_dir', local_model_path)
    results_dir = cfg.get('results_dir')
    dataset = cfg.get('dataset_to_train_attack_path')
    dataset_to_attack_dir = cfg.get('dataset_to_attack_path')
    use_ollama = str(cfg.get('use_ollama', True)).lower()
    target_model = cfg.get('target_model', cfg.get('ollama_model'))
    single_model_path = build_single_model_path(cfg, target_model)
    batch_models = discover_models(models_dir)
    single_attack = args.single_attack if args.single_attack is not None else cfg.get('single_attack')
    dry_run = bool(cfg.get('dry_run', False))
    interactive_heartbeat_seconds = int(cfg.get('interactive_heartbeat_seconds', 30))
    runtime_vllm_env = vllm_env_overrides(cfg, use_ollama)
    defense_runtime_vllm_env = vllm_env_overrides(cfg, False)

    jobs_dir = os.path.join(results_dir, 'jobs')
    ensure_directory(jobs_dir)
    ensure_directory(results_dir)

    if args.list_attacks:
        print_attack_overview(dataset_to_attack_dir)
        return

    def defense_train_settings():
        """@brief Build command settings for rule-tree defense training.

        @return Tuple `(train_script, output_rule, env_overrides)`.
        """
        train_script = resolve_path(base_dir, cfg.get('defense_train_script', 'defense/def.py'))
        train_dataset = resolve_path(base_dir, cfg.get('defense_train_dataset', 'evaluate/selected_examples.json'))
        output_rule = resolve_path(base_dir, cfg.get('defense_train_output_rule', 'defense/defense_rule_orchestrator.json'))
        log_dir = resolve_path(base_dir, cfg.get('defense_train_log_dir', 'defense/gp_logs'))
        vocab_file = cfg.get('defense_train_vocab_file')
        if vocab_file:
            vocab_file = resolve_path(base_dir, vocab_file)
        ensure_directory(os.path.dirname(output_rule))
        ensure_directory(log_dir)
        env_overrides = {
            "DEFENSE_TRAIN_DATASET": train_dataset,
            "DEFENSE_OUT_RULE": output_rule,
            "DEFENSE_LOG_DIR": log_dir,
            "DEFENSE_MAX_EXAMPLES": cfg.get('defense_train_max_examples', 0),
            "DEFENSE_POP_SIZE": cfg.get('defense_train_pop_size', 10),
            "DEFENSE_N_GEN": cfg.get('defense_train_n_gen', 10),
            "DEFENSE_N_TRIES": cfg.get('defense_train_n_tries', 5),
            "DEFENSE_USE_VLLM_GEN": cfg.get('defense_train_use_vllm', True),
            "DEFENSE_VLLM_MODEL": cfg.get('defense_train_vllm_model'),
            "DEFENSE_VLLM_TP": cfg.get('defense_train_vllm_tp'),
            "DEFENSE_VLLM_MAX_MODEL_LEN": cfg.get('defense_train_vllm_max_model_len'),
            "DEFENSE_VLLM_GPU_MEM_UTIL": cfg.get('defense_train_vllm_gpu_mem_util'),
            "DEFENSE_GEN_MODEL": cfg.get('defense_train_gen_model'),
            "DEFENSE_JUDGE_MODEL": cfg.get('defense_train_judge_model'),
            "DEFENSE_VOCAB_TXT": vocab_file,
            "OLLAMA_HOST": cfg.get('ollama_host'),
        }
        env_overrides = {k: v for k, v in env_overrides.items() if v is not None}
        return train_script, output_rule, env_overrides

    def defense_apply_settings():
        """@brief Build command settings for applying a trained defense rule.

        @return Command list for `defense/apply_rules.py`.
        """
        apply_script = resolve_path(base_dir, cfg.get('defense_apply_script', 'defense/apply_rules.py'))
        input_dir = resolve_path(base_dir, cfg.get('defense_apply_input_dir', dataset_to_attack_dir))
        output_dir = resolve_path(base_dir, cfg.get('defense_apply_output_dir', os.path.join(results_dir, 'defense_applied')))
        rule_path = resolve_path(base_dir, cfg.get('defense_apply_rule_path', cfg.get('defense_train_output_rule', 'defense/defense_rule_orchestrator.json')))
        seed = cfg.get('defense_apply_seed', 42)
        ensure_directory(output_dir)
        cmd_list = [
            "python3", apply_script,
            "--input-dir", input_dir,
            "--output-dir", output_dir,
            "--defense-rule", rule_path,
            "--seed", str(seed),
        ]
        return cmd_list

    def run_attack_single_json(label_prefix="[ONLY-ATTACK]"):
        """Run prepared JSON attack files for the configured single target model."""
        ensure_directory(results_dir)
        only_attack_py = os.path.join(scripts_dir, 'only_attack.py')
        try:
            inputs = select_attack_inputs(dataset_to_attack_dir, single_attack)
        except ValueError as e:
            print(f"{label_prefix} {e}")
            return

        created = []
        for input_idx, inp in enumerate(inputs, start=1):
            base = os.path.basename(inp)
            name_stem = os.path.splitext(base)[0]
            job_name = f"onlyattack_{name_stem}"
            cmd_list = ["python3", only_attack_py, single_model_path, inp, results_dir, use_ollama, target_model]
            if args.interactive:
                try:
                    run_interactive(
                        cmd_list,
                        env_overrides=runtime_vllm_env,
                        label=f"[INTERACTIVE]{label_prefix} input {input_idx}/{len(inputs)} {base}, target_model={target_model}",
                        heartbeat_seconds=interactive_heartbeat_seconds,
                    )
                except subprocess.CalledProcessError as e:
                    print(f"{label_prefix} Error: {e}")
            else:
                cmd = with_env_command(cmd_list, runtime_vllm_env)
                script_path = os.path.join(jobs_dir, f"job_{job_name}.sh")
                template = job_template if is_true(use_ollama) else batch_template
                content = template(job_name, cmd, target_model,)
                write_and_submit_job(script_path, content, dry_run)
                created.append(script_path)
        if args.interactive:
            print(f"{label_prefix} Processed {len(inputs)} JSON input(s)")
        else:
            print(f"{label_prefix} Created {len(created)} jobs")

    if args.defense_create_vocab:
        vocab_script = resolve_path(base_dir, cfg.get('defense_vocab_script', 'defense/create_vocabulary.py'))
        model_name = cfg.get('defense_vocab_model_name', 'internlm2.5:latest')
        model_dir = resolve_path(base_dir, cfg.get('defense_vocab_model_dir', os.path.join('models', model_name)))
        output_file = resolve_path(base_dir, cfg.get('defense_vocab_output_file', os.path.join('defense', 'models_vocabularies', f'{model_name}_vocab.txt')))
        ensure_directory(os.path.dirname(output_file))
        cmd_list = ["python3", vocab_script, "--model-name", model_name, "--model-dir", model_dir, "--output-file", output_file]
        if args.interactive:
            try:
                run_interactive(cmd_list, label="[INTERACTIVE][DEFENSE-VOCAB]", heartbeat_seconds=interactive_heartbeat_seconds)
            except subprocess.CalledProcessError as e:
                print(f"[DEFENSE-VOCAB] Error: {e}")
        else:
            script_path = os.path.join(jobs_dir, "job_defense_create_vocab.sh")
            content = batch_template("defense_create_vocab", quote_cmd(cmd_list), target_model or "vocab")
            write_and_submit_job(script_path, content, dry_run)
        return

    if args.defense_train:
        if args.interactive and is_true(cfg.get('defense_train_use_vllm', True)) and not require_vllm_cuda_for_interactive(False, "--defense-train"):
            return
        train_script, _output_rule, env_overrides = defense_train_settings()
        cmd_list = ["python3", train_script]
        if args.interactive:
            try:
                run_interactive(cmd_list, merge_env_overrides(defense_runtime_vllm_env, env_overrides), "[INTERACTIVE][DEFENSE-TRAIN]", interactive_heartbeat_seconds)
            except subprocess.CalledProcessError as e:
                print(f"[DEFENSE-TRAIN] Error: {e}")
        else:
            script_path = os.path.join(jobs_dir, "job_defense_train.sh")
            content = job_template(
                "defense_train",
                with_env_command(cmd_list, merge_env_overrides(defense_runtime_vllm_env, env_overrides)),
                cfg.get('defense_train_judge_model', target_model or "defense"),
            )
            write_and_submit_job(script_path, content, dry_run)
        return

    if args.defense_apply_rules:
        cmd_list = defense_apply_settings()
        if args.interactive:
            try:
                run_interactive(cmd_list, label="[INTERACTIVE][DEFENSE-APPLY-RULES]", heartbeat_seconds=interactive_heartbeat_seconds)
            except subprocess.CalledProcessError as e:
                print(f"[DEFENSE-APPLY-RULES] Error: {e}")
        else:
            script_path = os.path.join(jobs_dir, "job_defense_apply_rules.sh")
            content = batch_template("defense_apply_rules", quote_cmd(cmd_list), target_model or "rules")
            write_and_submit_job(script_path, content, dry_run)
        return

    if args.defense_train_apply:
        if args.interactive and is_true(cfg.get('defense_train_use_vllm', True)) and not require_vllm_cuda_for_interactive(False, "--defense-train-apply"):
            return
        train_script, _output_rule, env_overrides = defense_train_settings()
        train_cmd_list = ["python3", train_script]
        apply_cmd_list = defense_apply_settings()
        if args.interactive:
            try:
                run_interactive(train_cmd_list, merge_env_overrides(defense_runtime_vllm_env, env_overrides), "[INTERACTIVE][DEFENSE-TRAIN]", interactive_heartbeat_seconds)
                run_interactive(apply_cmd_list, label="[INTERACTIVE][DEFENSE-APPLY-RULES]", heartbeat_seconds=interactive_heartbeat_seconds)
            except subprocess.CalledProcessError as e:
                print(f"[DEFENSE-TRAIN-APPLY] Error: {e}")
        else:
            script_path = os.path.join(jobs_dir, "job_defense_train_apply.sh")
            cmd = template_multiline_command(with_env_command(train_cmd_list, merge_env_overrides(defense_runtime_vllm_env, env_overrides)) + "\n" + quote_cmd(apply_cmd_list))
            content = job_template(
                "defense_train_apply",
                cmd,
                cfg.get('defense_train_judge_model', target_model or "defense"),
            )
            write_and_submit_job(script_path, content, dry_run)
        return

    # Run pipeline: create a temporary config for run_pipeline.py and execute it
    if args.run_pipeline:
        if args.interactive and not require_vllm_cuda_for_interactive(use_ollama, "--run-pipeline"):
            return
        base_dir = os.path.dirname(os.path.abspath(__file__))
        pipeline_results = args.pipeline_out or results_dir
        ensure_directory(pipeline_results)

        # Use dataset_to_attack_dir as dataset_to_train_attack_path if available, else fall back
        pipeline_dataset = dataset_to_attack_dir or dataset

        temp_cfg_path = os.path.join(base_dir, 'config_pipeline_temp.yaml')
        temp_cfg = {}
        # copy relevant keys from cfg
        # provide both new and legacy keys in temp config for run_pipeline compatibility
        temp_cfg['local_model_path'] = single_model_path
        temp_cfg['victim_llm'] = single_model_path
        temp_cfg['results_dir'] = pipeline_results
        temp_cfg['dataset_to_train_attack_path'] = pipeline_dataset
        temp_cfg['use_ollama'] = cfg.get('use_ollama', True)
        temp_cfg['target_model'] = cfg.get('target_model', cfg.get('ollama_model'))
        temp_cfg['ollama_model'] = cfg.get('ollama_model')
        temp_cfg['which_methods'] = cfg.get('which_methods', None)

        with open(temp_cfg_path, 'w', encoding='utf-8') as fh:
            yaml.safe_dump(temp_cfg, fh)

        run_pipeline_py = os.path.join(scripts_dir, 'run_pipeline.py')
        if args.interactive:
            try:
                run_interactive(
                    ["python3", run_pipeline_py, "--config_file", temp_cfg_path],
                    env_overrides=runtime_vllm_env,
                    label=f"[INTERACTIVE][PIPELINE] config={temp_cfg_path}",
                    heartbeat_seconds=interactive_heartbeat_seconds,
                )
            except subprocess.CalledProcessError as e:
                print(f"[PIPELINE] Error: {e}")
            return
        else:
            # create a single job script that runs the pipeline
            per_jobs_dir = os.path.join(pipeline_results, 'jobs')
            ensure_directory(per_jobs_dir)
            job_name = "run_pipeline"
            cmd = with_env_command(["python3", run_pipeline_py, "--config_file", temp_cfg_path], runtime_vllm_env)
            script_path = os.path.join(per_jobs_dir, f"job_{job_name}.sh")
            content = job_template(job_name, cmd, temp_cfg.get('target_model', 'pipeline'))
            write_and_submit_job(script_path, content, dry_run)
            return

    # ATTACK_MODULES defined above; use local build_per_model_config

    if args.defense:
        if args.interactive and not require_vllm_cuda_for_interactive(use_ollama, f"--defense {args.defense}"):
            return
        print(f"[DEFENSE] mode: {args.defense}")
        models = batch_models
        total_steps = len(models) * len([f for f in os.listdir(dataset_to_attack_dir) if f.endswith('.json')])
        step = 0
        for model_idx, model in enumerate(models, start=1):
            per_local_model, per_results_dir, per_target_model = build_model_specific_config(cfg, model)
            per_jobs_dir = os.path.join(per_results_dir, 'jobs')
            ensure_directory(per_jobs_dir)
            inputs = [os.path.join(dataset_to_attack_dir, f) for f in sorted(os.listdir(dataset_to_attack_dir)) if f.endswith('.json')]
            for input_idx, inp in enumerate(inputs, start=1):
                step += 1
                base = os.path.basename(inp)
                name_stem = os.path.splitext(base)[0]
                job_name = f"defense_{args.defense}_{name_stem}"
                cmd_list = ["python3", os.path.join(scripts_dir, "only_defense_batch.py"), "--per-victim", per_local_model, "--defense", args.defense, "--model", per_target_model, "--input", inp, "--out_dir", per_results_dir, "--use-ollama", use_ollama]
                if args.interactive:
                    try:
                        run_interactive(
                            cmd_list,
                            env_overrides=runtime_vllm_env,
                            label=f"[INTERACTIVE][DEFENSE] step {step}/{total_steps}, model {model_idx}/{len(models)} {model}, input {input_idx}/{len(inputs)} {base}",
                            heartbeat_seconds=interactive_heartbeat_seconds,
                        )
                    except subprocess.CalledProcessError as e:
                        print(f"[DEFENSE] Error: {e}")
                else:
                    cmd = with_env_command(cmd_list, runtime_vllm_env)
                    script_path = os.path.join(per_jobs_dir, f"job_{job_name}.sh")
                    content = batch_template(job_name, cmd, per_target_model,)
                    write_and_submit_job(script_path, content, dry_run)
        return

    if args.evaluate:
        models = batch_models
        for model_idx, model in enumerate(models, start=1):
            cmd_list = ["python3", "evaluate/evaluate_full_results_datasets.py", model, "DEFENSE_SAFEGUARD"]
            if args.interactive:
                try:
                    run_interactive(
                        cmd_list,
                        label=f"[INTERACTIVE][EVALUATE] model {model_idx}/{len(models)} {model}",
                        heartbeat_seconds=interactive_heartbeat_seconds,
                    )
                except subprocess.CalledProcessError as e:
                    print(f"[EVALUATE] Error: {e}")
            else:
                script_path = os.path.join(results_dir, 'benign', 'stats', 'SAFEGUARD', f"job_{model}.sh")
                os.makedirs(os.path.dirname(script_path), exist_ok=True)
                cmd = quote_cmd(cmd_list)
                content = results_eval_template(model, cmd)
                write_and_submit_job(script_path, content, dry_run)
        return

    if args.attack_batch:
        if args.interactive and not require_vllm_cuda_for_interactive(use_ollama, "--attack-batch"):
            return
        models = batch_models
        total_inputs = len([f for f in os.listdir(dataset_to_attack_dir) if f.lower().endswith('.json')])
        total_steps = len(models) * total_inputs
        step = 0
        for model_idx, model in enumerate(models, start=1):
            per_local_model, per_results_dir, per_target_model = build_model_specific_config(cfg, model)
            per_jobs_dir = os.path.join(per_results_dir, 'jobs')
            ensure_directory(per_jobs_dir)
            inputs = [os.path.join(dataset_to_attack_dir, f) for f in sorted(os.listdir(dataset_to_attack_dir)) if f.lower().endswith('.json')]
            created = []
            for input_idx, inp in enumerate(inputs, start=1):
                step += 1
                base = os.path.basename(inp)
                name_stem = os.path.splitext(base)[0]
                job_name = f"onlyattackbatch_{name_stem}"
                cmd_list = ["python3", os.path.join(scripts_dir, "only_attack_batch.py"), per_local_model, inp, per_results_dir, use_ollama, per_target_model]
                if args.interactive:
                    try:
                        run_interactive(
                            cmd_list,
                            env_overrides=runtime_vllm_env,
                            label=f"[INTERACTIVE][ONLY-ATTACK-BATCH] step {step}/{total_steps}, model {model_idx}/{len(models)} {model}, input {input_idx}/{len(inputs)} {base}",
                            heartbeat_seconds=interactive_heartbeat_seconds,
                        )
                    except subprocess.CalledProcessError as e:
                        print(f"[ONLY-ATTACK-BATCH] Error: {e}")
                else:
                    cmd = with_env_command(cmd_list, runtime_vllm_env)
                    script_path = os.path.join(per_jobs_dir, f"job_{job_name}.sh")
                    content = batch_template(job_name, cmd, per_target_model,)
                    write_and_submit_job(script_path, content, dry_run)
                    created.append(script_path)
            print(f"[ONLY-ATTACK-BATCH] Created {len(created)} jobs for model {model}")
        return

    if args.attack_single:
        if args.interactive and not require_vllm_cuda_for_interactive(use_ollama, "--attack-single"):
            return
        run_attack_single_json("[ONLY-ATTACK]")
        return


    # Default: run prepared JSON attack files for the configured single target
    # model. The old module-run path generated attacks from a CSV dataset, while
    # the orchestrator workflow now treats JSON files as the input datasets.
    if args.interactive and not require_vllm_cuda_for_interactive(use_ollama, "default JSON attack"):
        return
    run_attack_single_json("[DEFAULT-JSON-ATTACK]")


if __name__ == '__main__':
    main()
