# run.py

import os, time, json, argparse, yaml, subprocess
from datetime import timedelta

from attacks._1_Cypher.main import run_cypher_attack
from attacks._2_Flip.main import run_flip_attack  
from attacks._3_PiF.PiF_CLM import run_pif_attack
from attacks._4_SQL_StructTransform.main import run_sql_attack
from attacks._5_suffix.main import run_suffix_attack
from attacks._6_Sequential.main import run_sequential_attack
from attacks._7_CitationBreak.main import run_cite_attack
from attacks._8_Bijection.main import run_bijection_attack
from attacks._9_Dialog_completition.main import run_dialog_attack
from attacks._10_Random_Search.main import run_random_attack
from attacks._11_Pair.main import run_pair_attack
from attacks._12_Tap.main import run_tap_attack
from attacks._13_GPT4cypher.main import run_GPTcypher_attack
from attacks._14_MultiLang.main import run_Multilang_attack
from attacks._15_Rewrite.main import run_rewrite_attack
from attacks._16_Ica.main import run_ica_attack
from attacks._17_Overload.main import run_overload_attack
from attacks._18_Gcg.main import run_gcg_attack
from attacks._19_Deepinception.main import run_inception_attack
from attacks._20_Base.main import run_base_attack
from attacks._21_Art_Prompt.main import run_artprompt_attack
from attacks._22_Renellm.main import run_renellm_attack
from attacks._24_Autodan.main import run_autodan_attack
from attacks._25_past.main import run_past_tense_attack
from attacks._26_Chameleon.main import run_chameleon_attack

from defense.defense_EA import DefenseEA

def log_and_print(msg: str, fh):
    """Vytiskne na stdout + zapíše do souboru."""
    print(msg)
    if fh:
        fh.write(msg + "\n")

def load_config(path):
    ext = os.path.splitext(path)[1].lower()
    with open(path, 'r', encoding='utf-8') as f:
        if ext in ('.yaml', '.yml'):
            return yaml.safe_load(f)
        elif ext == '.json':
            return json.load(f)
        else:
            raise ValueError(f"Unsupported config file: {path}")

works = [
    ("cypher", run_cypher_attack), #funguje
    # ("flip", run_flip_attack), #funguje
    # # ("pif", run_pif_attack), # FUNGUJE ALE OPATRENE S NIM - POUZIVA KNIHOVNU TRANSFORMERS, TEDY KONTROLUJ VELIKOST MODELU !!!
    # ("sql", run_sql_attack), 
    # ("suffix", run_suffix_attack),
    # ("sequential", run_sequential_attack),
    # ("cite", run_cite_attack),
    # ("bijection", run_bijection_attack),
    # ("dialog", run_dialog_attack),
    # ("random", run_random_attack),
    # ("pair", run_pair_attack),
    # ("tap", run_tap_attack), 
    # ("gptcypher", run_GPTcypher_attack),
    # ("MultiLang", run_Multilang_attack),
    # ("rewrite", run_rewrite_attack), 
    # ("ica", run_ica_attack),
    # ("overload", run_overload_attack), #funguje
    # # ("gcg", run_gcg_attack),
    # ("inception", run_inception_attack),
    # ("base", run_base_attack), #funguje
    # ("artprompt", run_artprompt_attack), #funguje
    # ("renellm", run_renellm_attack), #funguje
    # ("past_tense", run_past_tense_attack), #funguje
    # ("chameleon", run_chameleon_attack),
    # ("autodan", run_autodan_attack), #funguje
    # ("pif", run_pif_attack), #funguje
]

special_runs = [
    # ("pif", run_pif_attack),
    ("sequential", run_sequential_attack),
    ("dialog", run_dialog_attack),
]
test = [
    # ("gcg", run_gcg_attack), # funguje zatim jen na male modely, llama 7b jsem vyzkousel, protoze vetsi modely se mi nevejdou do pameti
    # ("overload", run_overload_attack),
    # ("pif", run_pif_attack),
    # ("citation", run_cite_attack)
    ("gptcypher", run_GPTcypher_attack),
    # ("MultiLang", run_Multilang_attack),
    # ("rewrite", run_rewrite_attack), 
    # ("renellm", run_renellm_attack)
    # ("renellm", run_renellm_attack), # nefunguje

]
does_not_work = [
    ("sql", run_sql_attack), # nefunguje
    ("artprompt", run_artprompt_attack), # nefunguje
    ("renellm", run_renellm_attack), # nefunguje
    ("past_tense", run_past_tense_attack), # nefunguje
    ("tap", run_tap_attack), # nefunguje
    ("pif", run_pif_attack), # FUNGUJE ALE OPATRENE S NIM - POUZIVA KNIHOVNU TRANSFORMERS, TEDY KONTROLUJ VELIKOST MODELU !!!
    ("autodan",    run_autodan_attack), 
]

CATEGORY_REGISTRY = {
    "works": works,
    "special_runs": special_runs,
    "does_not_work": does_not_work,
    "test": test,
    "all": works + special_runs + does_not_work + test,
}

def normalize_which_methods(val):
    """Přijme None / string / list a vrátí list kategorií (list[list[tuple]]).
       Povolí 'all' i kombinace (',' oddělené).
    """
    if val is None:
        return [works]  # default
    if isinstance(val, str):
        tokens = [t.strip().lower() for t in val.split(",") if t.strip()]
    elif isinstance(val, list):
        tokens = [str(t).strip().lower() for t in val if str(t).strip()]
    else:
        raise ValueError(f"Invalid which_methods type: {type(val)} (use str/list)")

    selected = []
    for t in tokens:
        if t not in CATEGORY_REGISTRY:
            valid = ", ".join(CATEGORY_REGISTRY.keys())
            raise ValueError(f"Unknown category '{t}'. Valid options: {valid}")
        if t == "all":
            return [CATEGORY_REGISTRY["all"]]
        selected.append(CATEGORY_REGISTRY[t])

    seen = set()
    unique = []
    for cat in selected:
        key = id(cat)
        if key not in seen:
            unique.append(cat)
            seen.add(key)
    return unique


def sub_jobs(victim_llm_path, results_dir, dataset_path, api_ollama_vllm, what_ollama_model):
    OUTDIR = "./jobs"
    os.makedirs(OUTDIR, exist_ok=True)

    gpu = "45gb"
    cpu = "200gb"
    ngpu = 1
    ncpu = 1
    walltime = "00:00:50"
    attack = "flip"
    # victim_llm_path = "/storage/brno2/home/xkaska01/master/prompt_attacker/models/Llama-2-7b-hf/"
    # results_dir = "/storage/brno2/home/xkaska01/master/prompt_attacker/results/TEST/"
    # dataset_path = "/storage/brno2/home/xkaska01/master/prompt_attacker/dataset/base_dataset.csv"
    # api_ollama_vllm = True
    # what_ollama_model = "falcon3:10b"

    script_path = os.path.join(OUTDIR, f"job_{what_ollama_model}_{attack}.sh")
    script_content = f"""#!/bin/bash
    #PBS -q default@pbs-m1.metacentrum.cz
    #PBS -N myjob_{what_ollama_model}_{attack}
    #PBS -l select=1:ncpus={ncpu}:ngpus={ngpu}:mem={cpu}:gpu_mem={gpu}
    #PBS -l walltime={walltime}

    HOMEDIR=/storage/brno2/home/xkaska01/master/

    export CUDA_VISIBLE_DEVICES=0
    module add mambaforge
    mamba activate /storage/brno2/home/xkaska01/.conda/envs/diplomka

    # spustit ollama (upravit cesty, pokud je potřeba)
    # spouštím jako background, loguji výstup
    /storage/brno2/home/xkaska01/test/bin/ollama serve > $HOMEDIR/ollama.log 2>&1 &
    /storage/brno2/home/xkaska01/test/bin/ollama pull qwen2.5:7b
    /storage/brno2/home/xkaska01/test/bin/ollama pull {what_ollama_model}

    echo "$PBS_JOBID běží na uzlu $(hostname -f)" >> $HOMEDIR/jobs_info.txt
    cd $HOMEDIR || exit 1

    # nainstalovat nltk jen pokud chybí (volitelné)
    python3 -m pip install --user nltk

    # spustit attack - upravte příkaz podle toho, jakým způsobem spouštíte attacky
    # pokud máte runner (run_pipeline.py) používejte -> python3 run_pipeline.py --config_file ... --only {attack}
    # níže zachovávám původní podobu, ale pravděpodobně budete chtít spouštět run_pipeline.py
    python3 {attack} {victim_llm_path} {results_dir} {dataset_path} {api_ollama_vllm} {what_ollama_model}

    """

    print(attack, victim_llm_path, results_dir, dataset_path, api_ollama_vllm, what_ollama_model)

    with open(script_path, "w", encoding="utf-8") as fh:
        fh.write(script_content)

    os.chmod(script_path, 0o755)
    print(f"Created script: {script_path}")

    try:
        res = subprocess.run(["qsub", script_path], check=True, capture_output=True, text=True)
        print(f"✅ Submitted job: {res.stdout.strip()}")
    except FileNotFoundError:
        print("❌ Chyba: příkaz 'qsub' nebyl nalezen v PATH. Přihlaste se na frontend/metacentrum uzel nebo upravte PATH.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error submitting job: {e.stderr.strip()}")



if __name__ == "__main__":

    defense = DefenseEA()  
    parser = argparse.ArgumentParser("FlipAttack runner")
    parser.add_argument(
        "--config_file", "-c", type=str, required=True,
        help="Path to YAML (.yaml/.yml) or JSON (.json) config file"
    )
    args = parser.parse_args()

    cfg = load_config(args.config_file)
    victim_llm_path = cfg.get("victim_llm")
    results_dir = cfg.get("results_dir")
    dataset_path = cfg.get("dataset_path")
    api_ollama_vllm = cfg.get("use_ollama")
    what_ollama_model = cfg.get("ollama_model")
    which_methods = cfg.get("which_methods")
    print(which_methods)

    all_attack_categories = normalize_which_methods(which_methods)

    print(f"[INFO] Victim LLM Path: {victim_llm_path}"
          f"\n[INFO] Results Directory: {results_dir}")
    
    print(f"[INFO] Running FlipAttack with config: {cfg.get('run_defense', False)}")

    os.makedirs(results_dir, exist_ok=True)
    log_file_path = os.path.join(results_dir, "log_runtime.txt")
    log_fh = open(log_file_path, "w", encoding="utf-8")

    timings = {}
    total_start = time.perf_counter()

    modely = ["deepseek-r1:32b", "llama3.1:70b", "falcon3:10b", "gemma3:27b", "qwen3:32b", "yi:34b", "internlm/internlm2.5:latest", "command-r:35b"]

    for category in all_attack_categories:
        for name, fn in category:
            log_and_print(f"\n[INFO] ➜ Spouštím {name}…", log_fh)
            t0 = time.perf_counter()
            sub_jobs(victim_llm_path, results_dir, dataset_path, api_ollama_vllm, what_ollama_model)
            elapsed = time.perf_counter() - t0
            timings[name] = elapsed
            log_and_print(f"[INFO]   {name} hotovo za {timedelta(seconds=elapsed)}", log_fh)

    total_elapsed = time.perf_counter() - total_start

    log_and_print("\n========== Souhrn ==========", log_fh)
    for name, secs in timings.items():
        log_and_print(f"{name:12s}: {timedelta(seconds=secs)}", log_fh)
    log_and_print(f"{'TOTAL':12s}: {timedelta(seconds=total_elapsed)}", log_fh)

    log_and_print(f"\n[INFO] Runtime log uložen do {log_file_path}", log_fh)
    log_fh.close()
    
     # print(defense.apply(" some prompt  "))  

    
    
# python3 run_pipeline.py --config_file config.yaml