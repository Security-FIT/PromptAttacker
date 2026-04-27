#!/usr/bin/env python3
import os
import textwrap
import subprocess
import argparse
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
    """Replace `{model}` placeholder in a template string with the given model."""
    if template is None:
        return template
    return template.replace("{model}", model)


def build_model_specific_config(cfg: dict, model: str):
    """Return (local_model_path, results_dir, target_model) with `{model}` substituted.

    Supports both new keys (`local_model_path`, `target_model`) and legacy keys
    (`victim_llm`, `ollama_model`) for backwards compatibility.
    """
    local_model_t = str(cfg.get("local_model_path", cfg.get("victim_llm", "")))
    results_dir_t = str(cfg.get("results_dir", ""))
    target_model_t = str(cfg.get("target_model", cfg.get("ollama_model", "")))

    local_model = substitute_model_in_string(local_model_t, model)
    results_dir = substitute_model_in_string(results_dir_t, model)
    target_model = substitute_model_in_string(target_model_t, model)

    return local_model, results_dir, target_model

# --- Utilities ---

def load_config(path: str) -> dict:
    """Load YAML configuration file and return a dict."""
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def ensure_directory(path: str):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


# --- Orchestration logic ---

def write_and_submit_job(script_path: str, content: str, dry_run: bool):
    """Write job script and optionally submit it with `qsub`.

    If `submit` is True and `dry_run` is False the script will be submitted.
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_orchestrator.yaml")
    parser.add_argument("--evaluate", action="store_true", help="Create eval job scripts for models in config")
    parser.add_argument("--attack-single", action="store_true", help="Create only-attack job scripts for the single model specified in `ollama_model` in config")
    parser.add_argument("--attack-batch", action="store_true", help="Create only-attack batch job scripts and iterate over `ollama_models` in config")
    parser.add_argument("--defense", choices=["rallm", "llamaguard", "safeguard"], help="Create defense job scripts for specified defense type") 
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
    results_dir = cfg.get('results_dir')
    dataset = cfg.get('dataset_to_train_attack_path')
    dataset_to_attack_dir = cfg.get('dataset_to_attack_path')
    use_ollama = str(cfg.get('use_ollama', True)).lower()
    target_model = cfg.get('target_model', cfg.get('ollama_model'))
    target_models = cfg.get('target_models', cfg.get('ollama_models', []))
    dry_run = bool(cfg.get('dry_run', False))

    jobs_dir = os.path.join(results_dir, 'jobs')
    ensure_directory(jobs_dir)
    ensure_directory(results_dir)

    # Run pipeline: create a temporary config for run_pipeline.py and execute it
    if args.run_pipeline:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        pipeline_results = args.pipeline_out or results_dir
        ensure_directory(pipeline_results)

        # Use dataset_to_attack_dir as dataset_to_train_attack_path if available, else fall back
        pipeline_dataset = dataset_to_attack_dir or dataset

        temp_cfg_path = os.path.join(base_dir, 'config_pipeline_temp.yaml')
        temp_cfg = {}
        # copy relevant keys from cfg
        # provide both new and legacy keys in temp config for run_pipeline compatibility
        temp_cfg['local_model_path'] = local_model_path
        temp_cfg['victim_llm'] = local_model_path
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
            print(f"[PIPELINE] Running run_pipeline.py interactively with config: {temp_cfg_path}")
            try:
                subprocess.run(["python3", run_pipeline_py, "--config_file", temp_cfg_path], check=True)
            except subprocess.CalledProcessError as e:
                print(f"[PIPELINE] Error: {e}")
            return
        else:
            # create a single job script that runs the pipeline
            per_jobs_dir = os.path.join(pipeline_results, 'jobs')
            ensure_directory(per_jobs_dir)
            job_name = "run_pipeline"
            cmd = f"python3 {run_pipeline_py} --config_file {temp_cfg_path}"
            script_path = os.path.join(per_jobs_dir, f"job_{job_name}.sh")
            content = job_template(job_name, cmd, temp_cfg.get('target_model', 'pipeline'))
            write_and_submit_job(script_path, content, dry_run)
            return

    # ATTACK_MODULES defined above; use local build_per_model_config

    if args.defense:
        print(f"[DEFENSE] mode: {args.defense}")
        models = target_models
        for model in models:
            per_local_model, per_results_dir, per_target_model = build_model_specific_config(cfg, model)
            per_jobs_dir = os.path.join(per_results_dir, 'jobs')
            ensure_directory(per_jobs_dir)
            inputs = [os.path.join(dataset_to_attack_dir, f) for f in os.listdir(dataset_to_attack_dir) if f.endswith('.json')]
            for inp in inputs:
                base = os.path.basename(inp)
                name_stem = os.path.splitext(base)[0]
                job_name = f"defense_{args.defense}_{name_stem}"
                cmd_list = ["python3", os.path.join(scripts_dir, "only_defense_batch.py"), "--per-victim", per_local_model, "--defense", args.defense, "--model", per_target_model, "--input", inp, "--out_dir", per_results_dir, "--use-ollama", use_ollama]
                if args.interactive:
                    print(f"[INTERACTIVE][DEFENSE] {job_name}")
                    try:
                        subprocess.run(cmd_list, check=True)
                    except subprocess.CalledProcessError as e:
                        print(f"[DEFENSE] Error: {e}")
                else:
                    cmd = " ".join(cmd_list)
                    script_path = os.path.join(per_jobs_dir, f"job_{job_name}.sh")
                    content = batch_template(job_name, cmd, per_target_model,)
                    write_and_submit_job(script_path, content, dry_run)
        return

    if args.evaluate:
        models = target_models
        for model in models:
            cmd_list = ["python3", "evaluate/evaluate_full_results_datasets.py", model, "DEFENSE_SAFEGUARD"]
            if args.interactive:
                print(f"[INTERACTIVE][EVALUATE] {model}")
                try:
                    subprocess.run(cmd_list, check=True)
                except subprocess.CalledProcessError as e:
                    print(f"[EVALUATE] Error: {e}")
            else:
                script_path = os.path.join(results_dir, 'benign', 'stats', 'SAFEGUARD', f"job_{model}.sh")
                os.makedirs(os.path.dirname(script_path), exist_ok=True)
                cmd = " ".join(cmd_list)
                content = results_eval_template(model, cmd)
                write_and_submit_job(script_path, content, dry_run)
        return

    if args.attack_batch:
        models = target_models
        for model in models:
            per_local_model, per_results_dir, per_target_model = build_model_specific_config(cfg, model)
            per_jobs_dir = os.path.join(per_results_dir, 'jobs')
            ensure_directory(per_jobs_dir)
            inputs = [os.path.join(dataset_to_attack_dir, f) for f in sorted(os.listdir(dataset_to_attack_dir)) if f.lower().endswith('.json')]
            created = []
            for inp in inputs:
                base = os.path.basename(inp)
                name_stem = os.path.splitext(base)[0]
                job_name = f"onlyattackbatch_{name_stem}"
                cmd_list = ["python3", os.path.join(scripts_dir, "only_attack_batch.py"), per_local_model, inp, per_results_dir, use_ollama, per_target_model]
                if args.interactive:
                    print(f"[INTERACTIVE][ONLY-ATTACK-BATCH] {job_name}")
                    try:
                        subprocess.run(cmd_list, check=True)
                    except subprocess.CalledProcessError as e:
                        print(f"[ONLY-ATTACK-BATCH] Error: {e}")
                else:
                    cmd = " ".join(cmd_list)
                    script_path = os.path.join(per_jobs_dir, f"job_{job_name}.sh")
                    content = batch_template(job_name, cmd, per_target_model,)
                    write_and_submit_job(script_path, content, dry_run)
                    created.append(script_path)
            print(f"[ONLY-ATTACK-BATCH] Created {len(created)} jobs for model {model}")
        return

    if args.attack_single:
        ensure_directory(results_dir)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        only_attack_py = os.path.join(scripts_dir, 'only_attack.py')
        inputs = [os.path.join(dataset_to_attack_dir, f) for f in sorted(os.listdir(dataset_to_attack_dir)) if f.lower().endswith('.json')]
        created = []
        for inp in inputs:
            base = os.path.basename(inp)
            name_stem = os.path.splitext(base)[0]
            job_name = f"onlyattack_{name_stem}"
            cmd_list = ["python3", only_attack_py, local_model_path, inp, results_dir, use_ollama, target_model]
            if args.interactive:
                print(f"[INTERACTIVE][ONLY-ATTACK] {job_name}")
                try:
                    subprocess.run(cmd_list, check=True)
                except subprocess.CalledProcessError as e:
                    print(f"[ONLY-ATTACK] Error: {e}")
            else:
                cmd = " ".join(cmd_list)
                script_path = os.path.join(jobs_dir, f"job_{job_name}.sh")
                content = job_template(job_name, cmd, target_model,)
                write_and_submit_job(script_path, content, dry_run)
                created.append(script_path)
        print(f"[ONLY-ATTACK] Created {len(created)} jobs")
        return


    # default: create attack module jobs
    created = []
    for module in ATTACK_MODULES:
        name = module.split('.')[1]
        safe_name = module.replace('attacks.', '').replace('.main', '').replace('.', '_')
        cmd_list = ["python3", "-m", module, local_model_path, results_dir, dataset, use_ollama, target_model]
        if args.interactive:
            print(f"[INTERACTIVE][MODULE] {safe_name}")
            try:
                subprocess.run(cmd_list, check=True)
            except subprocess.CalledProcessError as e:
                print(f"[MODULE] Error: {e}")
        else:
            cmd = " ".join(cmd_list)
            script_path = os.path.join(jobs_dir, f"job_{safe_name}.sh")
            content = job_template(safe_name, cmd, target_model,)
            write_and_submit_job(script_path, content, dry_run)
            created.append(script_path)
    print(f"[INFO] Hotovo. Vytvořeno jobů: {len(created)}")


if __name__ == '__main__':
    main()
