#!/usr/bin/env python3
import os, yaml, textwrap, subprocess, argparse
from pathlib import Path
import re
import json
# from my_implementation.evaluate.evaluate_full_results_datasets import evaluate_model

# === Nastavení ===
CONFIG = "/storage/brno2/home/xkaska01/master/my_implementation/config.yaml"  # uprav dle sebe
SUBMIT = True
DRY_RUN = False

# Moduly (ty, co máš) – spouští se jako: python3 -m <module> <args>
ATTACK_MODULES = [
    "attacks._1_Cypher.main",
    "attacks._2_Flip.main",
    # "attacks._3_PiF.PiF_CLM",
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
    "attacks._18_Gcg.main",
    "attacks._19_Deepinception.main",
    "attacks._20_Base.main",
    "attacks._21_Art_Prompt.main",
    "attacks._22_Renellm.main",
    "attacks._24_Autodan.main",
    "attacks._25_past.main",
    "attacks._26_Chameleon.main",
]

def load_cfg(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def safe_filename(s: str) -> str:
    return re.sub(r'[^A-Za-z0-9_.-]+', '-', s)

def attack_key_from_module(mod: str) -> str:
    """
    'attacks._16_Ica.main' -> '_16_ica'
    """
    mid = mod.replace("attacks.", "").replace(".main", "")
    return mid.lower()

# odstraňuje „přílepky“ v názvech výstupních souborů
_SUFFIXES_TO_STRIP = (
    "_attack", "_attacks",
    "_completition", "_completion",
    "_structtransform", "_struct_transform",
    "_random_search", "_randomsearch",
    "_dialog", "_dialogs",
    "_gcg",
    "_past",
)

def normalize_json_stem(stem: str) -> str:
    """
    '_6_sequential_attack' -> '_6_sequential'
    '_4_sql'               -> '_4_sql' (ponecháme; ošetří alias níže)
    """
    s = stem.lower()
    for suf in _SUFFIXES_TO_STRIP:
        if s.endswith(suf):
            s = s[: -len(suf)]
            break
    # sjednocení vícenásobných podtržítek/mezery
    s = re.sub(r'[^a-z0-9:_]+', '_', s)
    s = re.sub(r'_+', '_', s).strip('_')
    if not s.startswith('_'):
        s = '_' + s
    return s


def parse_model_from_path(p: Path) -> str | None:
    """
    Hledá segment ve stylu 'DATASET_*_<MODEL>' a vrací <MODEL>.
    Příklad: /.../DATASET_FULL_gemma3:12b/_6_sequential_attack.json -> 'gemma3:12b'
    """
    for seg in p.parts:
        if seg.startswith("DATASET_"):
            toks = seg.split("_")
            if len(toks) >= 2:
                return toks[-1]
    return None

def map_attack_stem_to_module() -> dict:
    """
    Vytvoří mapu: '_16_ica' -> 'attacks._16_Ica.main'
    + přidá aliasy na nejčastější názvy souborů.
    """
    d = {}
    for m in ATTACK_MODULES:
        k = attack_key_from_module(m)  # např. '_6_sequential'
        d[k] = m

    # aliasy (levá strana = jak se typicky jmenuje JSON soubor)
    aliases = {
        "_4_sql":             "attacks._4_SQL_StructTransform.main",
        "_9_dialog":          "attacks._9_Dialog_completition.main",
        "_9_dialog_completition": "attacks._9_Dialog_completition.main",
        "_10_randomsearch":   "attacks._10_Random_Search.main",
        "_10_random_search":  "attacks._10_Random_Search.main",
        "_6_sequential_attack": "attacks._6_Sequential.main",
        "_6_sequential":        "attacks._6_Sequential.main",
        "_13_gpt4cypher":     "attacks._13_GPT4cypher.main",
        "_18_gcg":            "attacks._18_Gcg.main",
        "_25_past":           "attacks._25_past.main",
        "_19_deep_inception":  "attacks._19_Deepinception.main",
        "_21_artprompt":    "attacks._21_Art_Prompt.main",
    }
    # zapiš aliasy jen pokud už nejsou
    for alias_key, module in aliases.items():
        d.setdefault(alias_key, module)
    return d

def find_module_for_stem(raw_stem: str, attack_map: dict) -> str | None:
    """
    Pokus o robustní párování:
    1) exact match po normalizaci
    2) prefix match (normalized startswith key) – upřednostníme nejdelší klíč
    3) substring match – upřednostníme nejdelší klíč
    """
    s = normalize_json_stem(raw_stem)          # např. '_6_sequential'
    keys = list(attack_map.keys())

    # 1) exact
    if s in attack_map:
        return attack_map[s]

    # 2) longest prefix
    keys_sorted = sorted(keys, key=len, reverse=True)
    for k in keys_sorted:
        if s.startswith(k) or k.startswith(s):
            return attack_map[k]

    # 3) substring
    for k in keys_sorted:
        if k in s or s in k:
            return attack_map[k]

    return None


def job_script_content(name, cmd, ollama_model):

    gpu = "40gb"
    cpu = "15gb"
    ngpu = 2
    ncpu = 1
    walltime = "1:00:00"

    # if name in "_12_Tap":
    #     walltime = "1:00:00"
    # elif name in "_11_Pair":
    #     walltime = "04:00:00"
    # elif name in "_25_past":
    #     walltime = "10:00:00"
    print(name)
    print(walltime)
    # exit(0)
    return textwrap.dedent(f"""\
        #!/bin/bash
        #PBS -q default@pbs-m1.metacentrum.cz
        #PBS -N {name}_{ollama_model}
        #PBS -l select=1:ncpus={ncpu}:ngpus={ngpu}:mem={cpu}:gpu_mem={gpu}
        #PBS -l walltime={walltime}

        HOMEDIR=/storage/brno2/home/xkaska01/master/my_implementation

        export PYTHONPATH="$(pwd):$PYTHONPATH"
        cd $HOMEDIR
        
        export CUDA_VISIBLE_DEVICES=0
        module add mambaforge
        mamba activate /storage/brno2/home/xkaska01/.conda/envs/diplomka

        # spustit ollama (upravit cesty, pokud je potřeba)
        # spouštím jako background, loguji výstup
        /storage/brno2/home/xkaska01/test/bin/ollama serve > $HOMEDIR/ollama.log 2>&1 &
        /storage/brno2/home/xkaska01/test/bin/ollama pull {ollama_model}
        python3 -m pip install --user nltk

        {cmd}

        echo "End {name}: $(date)"
    """)


# 
#  gpu_cap=compute_80 TOTO SEM SEM PRIDAL !!!!! TAK TO PAK KDYZTAK ODDELEJ
def job_batch_script_content(name, cmd, ollama_model):

    gpu = "50gb"
    cpu = "20gb"
    ngpu = 1
    ncpu = 1
    walltime = "8:00:00"

    # if name in "_12_Tap":
    #     walltime = "1:00:00"
    # elif name in "_11_Pair":
    #     walltime = "04:00:00"
    # elif name in "_25_past":
    #     walltime = "10:00:00"
    print(name)
    print(walltime)
    # exit(0)
    return textwrap.dedent(f"""\
        #!/bin/bash
        #PBS -q default@pbs-m1.metacentrum.cz
        #PBS -N {name}_{ollama_model}
        #PBS -l select=1:ncpus={ncpu}:ngpus={ngpu}:mem={cpu}:gpu_mem={gpu}
        #PBS -l walltime={walltime}

        HOMEDIR=/storage/brno2/home/xkaska01/master/my_implementation

        export PYTHONPATH="$(pwd):$PYTHONPATH"
        cd $HOMEDIR
        
        export CUDA_VISIBLE_DEVICES=0
        module add mambaforge
        mamba activate /storage/brno2/home/xkaska01/.conda/envs/diplomka

        python3 -m pip install --user nltk

        {cmd}

        echo "End {name}: $(date)"
    """)



# toto zkusim na EVAL
# set -euo pipefail

# HOMEDIR=/storage/brno2/home/xkaska01/master/my_implementation
# mkdir -p $HOMEDIR/logs

# OLLAMA_LOG=$HOMEDIR/logs/ollama_{name}_${PBS_JOBID}.log

# export PYTHONPATH="$HOMEDIR:$PYTHONPATH"
# cd $HOMEDIR

# export CUDA_VISIBLE_DEVICES=0
# export OLLAMA_HOST=127.0.0.1:11434
# export OLLAMA_MODELS=/storage/brno2/home/xkaska01/.ollama/models

# module add mambaforge
# mamba activate /storage/brno2/home/xkaska01/.conda/envs/diplomka

# /storage/brno2/home/xkaska01/test/bin/ollama serve > "$OLLAMA_LOG" 2>&1 &
# OLLAMA_PID=$!

# for i in $(seq 1 60); do
#     if ! kill -0 $OLLAMA_PID 2>/dev/null; then
#         echo "ERROR: Ollama process skončil."
#         cat "$OLLAMA_LOG"
#         exit 1
#     fi

#     if curl -s http://127.0.0.1:11434/api/tags > /dev/null; then
#         echo "Ollama ready"
#         break
#     fi
#     sleep 2
# done

# curl -f http://127.0.0.1:11434/api/tags > /dev/null || {
#     echo "ERROR: Ollama nenaběhla"
#     cat "$OLLAMA_LOG"
#     exit 1
# }

# python3 /storage/brno2/home/xkaska01/master/my_implementation/evaluate/evaluate_full_results_datasets.py {name}

# echo "===== OLLAMA LOG ====="
# cat "$OLLAMA_LOG" || true

# kill $OLLAMA_PID || true


def results_eval_all(name):

    gpu = "30gb"
    cpu = "50gb"
    ngpu = 1
    ncpu = 1
    walltime = "15:00:00"

    # print(name)
    # print(walltime)
    # exit(0)
    return textwrap.dedent(f"""\
        #!/bin/bash
        #PBS -q default@pbs-m1.metacentrum.cz
        #PBS -N {name}
        #PBS -l select=1:ncpus={ncpu}:ngpus={ngpu}:mem={cpu}:gpu_mem={gpu}
        #PBS -l walltime={walltime}

        HOMEDIR=/storage/brno2/home/xkaska01/master/my_implementation

        export PYTHONPATH="$(pwd):$PYTHONPATH"
        cd $HOMEDIR
        
        export CUDA_VISIBLE_DEVICES=0
        module add mambaforge
        mamba activate /storage/brno2/home/xkaska01/.conda/envs/diplomka

        # spustit ollama (upravit cesty, pokud je potřeba)
        # spouštím jako background, loguji výstup
        /storage/brno2/home/xkaska01/test/bin/ollama serve > $HOMEDIR/ollama.log 2>&1 &

        echo "Ollama PID: $OLLAMA_PID"
        echo "Čekám, než Ollama naběhne..."

        # čekání na dostupnost API
        for i in $(seq 1 60); do
            if curl -s http://127.0.0.1:11434/api/tags > /dev/null; then
                echo "Ollama je připravená."
                break
            fi
            sleep 2
        done

        # finální kontrola
        curl -f http://127.0.0.1:11434/api/tags > /dev/null || {{
            echo "ERROR: Ollama nenaběhla."
            cat $HOMEDIR/ollama.log
            exit 1
        }}


        /storage/brno2/home/xkaska01/test/bin/ollama pull gemma3:12b
        python3 -m pip install --user nltk

        python3 /storage/brno2/home/xkaska01/master/my_implementation/evaluate/evaluate_full_results_datasets.py {name} DEFENSE_SAFEGUARD

        echo "End {name}: $(date)"
    """)


def try_query_ollama(prompt: str, ollama_path, model: str, timeout: int = 120) -> str:
    if not os.path.exists(ollama_path):
        return f"<<OLLAMA NOT FOUND at {ollama_path}>>"
    try:
        res = subprocess.run([ollama_path, "run", model, "--prompt", prompt],
                             capture_output=True, text=True, timeout=timeout, check=True)
        return (res.stdout or res.stderr).strip()
    except subprocess.CalledProcessError as e:
        return f"<<OLLAMA CALL ERROR: {e.stderr.strip() if e.stderr else str(e)}>>"
    except Exception as e:
        return f"<<OLLAMA EXC: {e}>>"

def only_attack_worker_for_file(in_path: str, out_dir: str, use_ollama: str, ollama_model: str, victim_llm: str):
    """Načte JSON pole ze souboru in_path, zavolá model pro každé 'prompt', uloží do out_dir/basename(in_path)."""
    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, os.path.basename(in_path))

    out = []
    for item in data:
        _id = item.get("id", item.get("idx"))
        original = item.get("original_prompt") or item.get("original prompt") or item.get("goal") or item.get("prompt_original")
        prompt = item.get("prompt") or item.get("adversarial_prompt") or item.get("adv_prompt") or item.get("translation_of_goal") or item.get("goal")

        if prompt is None:
            resp = "<<NO PROMPT FOUND IN ITEM>>"
        else:
            if str(use_ollama).lower() == "true" and not DRY_RUN:
                resp = try_query_ollama(prompt, ollama_model)
            elif DRY_RUN:
                resp = "<<DRY_RUN - would call ollama>>"
            else:
                resp = f"<<SKIPPED (use_ollama!=true). Would send to victim_llm: {victim_llm} >>"

        out.append({
            "id": _id,
            "original_prompt": original,
            "prompt": prompt,
            "response": resp
        })

    with open(out_path, "w", encoding="utf-8") as fo:
        json.dump(out, fo, ensure_ascii=False, indent=2)

    print(f"[ONLY-ATTACK] {os.path.basename(in_path)} -> {out_path}")
    return out_path

def is_in_jobs_folder(p: Path) -> bool:
    return "jobs" in p.parts

def apply_model_to_template(s: str, model: str) -> str:
    """
    Nahradí {model} v stringu. Když tam není, nechá beze změny.
    """
    if s is None:
        return s
    return s.replace("{model}", model)

def build_per_model_config(cfg: dict, model: str) -> tuple[str, str, str]:
    """
    Z cfg udělá per-model trio (victim_llm, results_dir, ollama_model),
    kde se do stringů dosadí aktuální model.
    """
    victim_llm_t = str(cfg["victim_llm"])
    results_dir_t = str(cfg["results_dir"])
    ollama_model_t = str(cfg["ollama_model"])

    victim_llm = apply_model_to_template(victim_llm_t, model)
    results_dir = apply_model_to_template(results_dir_t, model)
    ollama_model = apply_model_to_template(ollama_model_t, model)

    return victim_llm, results_dir, ollama_model

def train_defense_model():
    # TODO
    return


def apply_defense_model():
    # TODO
    return



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true", help="Spustit evaluaci všech výsledků před útoky")
    parser.add_argument("--fix", action="store_true", help="Najde prázdné/chybné výsledky a znovu je spustí")
    parser.add_argument("--only-attack", action="store_true", help="Použít dataset_to_attack z configu (JSON array) a spustit útoky nad ním.")
    parser.add_argument("--only-attack-batch", action="store_true", help="Stejné jako --only-attack, ale volá batch inference skript only_attack_batch.py.")
    parser.add_argument("--only-defense", choices=["rallm", "llamaguard", "safeguard"], help="Spustit inferenci s konkrétní obranou nad existujícími útoky.")
    
    args = parser.parse_args()


    cfg = load_cfg(CONFIG)
    victim_llm = cfg["victim_llm"]
    results_dir = cfg["results_dir"]
    dataset = cfg["dataset_to_train_attack_path"]
    dataset_to_attack_dir = cfg["dataset_to_attack_path"]
    use_ollama = str(cfg.get("use_ollama", True)).lower() 
    ollama_model = cfg["ollama_model"]

    jobs_dir = os.path.join(results_dir, "jobs")
    ensure_dir(jobs_dir)
    ensure_dir(results_dir)

    print(f"[INFO] Results dir: {results_dir}")
    print(f"[INFO] Jobs dir:    {jobs_dir}")

    if args.only_defense:
        defense_type = args.only_defense
        print(f"[DEFENSE] === Režim OBRANA: {defense_type} ===")

        # Definujeme modely, pro které chceš obrany pustit (můžeš použít své listy OLLAMA_MODELS)
        MODELS_TO_RUN = [  "falcon3:3b", "falcon3:7b", "falcon3:10b",
        "gemma3:1b", "gemma3:14b", "gemma3:12b", "gemma3:27b",
        "internlm2:1m", "internlm2:20b", "internlm2.5:latest",
        "llama2:7b", "llama2:13b",
        "llama3.1:8b", "llama3.2:1b", "llama3.2:3b",
        "qwen2.5:3b", "qwen2.5:7b", "qwen2.5:14b", "qwen2.5:32b",
        "qwen3:0.6b", "qwen3:14b", "qwen3:30b", "qwen3:32b",
        "yi:6b", "yi:9b", "yi:34b",
        "command-r:35b", "command-r:latest",
        "deepseek-r1:32b","phi3:8b", "phi3:14b", "phi4:14b", "mixtral:7b"
        ]

        MODELS_TO_TEST = [ "falcon3:3b"]

        for model in MODELS_TO_RUN:
            per_victim_llm, per_results_dir, per_ollama_model = build_per_model_config(cfg, model)
            
            # Výsledky obran chceme v separátní podsložce, aby se nepřebily s čistými útoky
            defense_results_dir = os.path.join(per_results_dir, f"DEFENSE_{defense_type.upper()}")
            ensure_dir(defense_results_dir)
            
            per_jobs_dir = os.path.join(defense_results_dir, "jobs")
            ensure_dir(per_jobs_dir)

            # Projdeme JSONy s útoky
            inputs = [os.path.join(dataset_to_attack_dir, f) for f in os.listdir(dataset_to_attack_dir) if f.endswith(".json")]

            for inp in inputs:
                base = os.path.basename(inp)
                name_stem = os.path.splitext(base)[0]
                job_name = f"defense_{defense_type}_{safe_filename(name_stem)}"

                # PŘÍKAZ: Voláme nový skript only_defense_batch.py (vytvoříme níže)
                # Předáváme typ obrany jako argument
                cmd = f"python3 only_defense_batch.py --per-victim {per_victim_llm} --defense {defense_type} --model {per_ollama_model} --input {inp} --out_dir {defense_results_dir} --use-ollama {use_ollama} "

                script_path = os.path.join(per_jobs_dir, f"job_{job_name}.sh")
                with open(script_path, "w", encoding="utf-8") as fh:
                    # Použijeme tvůj existující template pro batch joby
                    fh.write(job_batch_script_content(job_name, cmd, per_ollama_model))
                
                os.chmod(script_path, 0o755)
                
                if SUBMIT and not DRY_RUN:
                    print(f"[DEFENSE] Submitting {job_name} for model {model}")
                    subprocess.run(["qsub", script_path])
        return

    if args.eval:
        print("[INFO] Spouštím evaluaci všech výsledků před útoky...")
        OLLAMA_MODELS = [ "falcon3:3b", "falcon3:7b", "falcon3:10b",
        "gemma3:1b", "gemma3:14b", "gemma3:12b", "gemma3:27b",
        "internlm2:1m", "internlm2:20b", "internlm2.5:latest",
        "llama2:7b", "llama2:13b",
        "llama3.1:8b", "llama3.2:1b", "llama3.2:3b",
        "qwen2.5:3b", "qwen2.5:7b", "qwen2.5:14b", "qwen2.5:32b",
        "qwen3:0.6b", "qwen3:14b", "qwen3:30b", "qwen3:32b",
        "yi:6b", "yi:9b", "yi:34b",
        "command-r:35b", "command-r:latest",
        "deepseek-r1:32b","phi3:8b", "phi3:14b", "phi4:14b", "mixtral:7b"
        ]
        
        OLLAMA_REST = [
       "command-r:35b", "command-r:latest","deepseek-r1:32b", "llama3.1:8b", "llama3.2:3b", "qwen2.5:0.5b", "qwen3:14b "
        ]

        OLLAMA_ONE = ["gemma3:14b"]
        for oll_model in OLLAMA_MODELS:

            print(f"[INFO] Creating eval job for model: {oll_model}")
            # continue
            # script_path = os.path.join(f"/storage/brno2/home/xkaska01/master/my_implementation/results/stats_{output_stats}_defense", f"job_{oll_model}.sh")
            script_path = os.path.join(f"/storage/brno2/home/xkaska01/master/my_implementation/results/benign/stats/SAFEGUARD", f"job_{oll_model}.sh")
            # TOTO MUSIM DAT STEJNE JAKO ve funkci out_csv_path v evaluate_full_results_datasets.py, aby se to udrželo konzistentní
            # A POTOM JESTE U TVORBY JOBU, TEDA NAHORE V TOM TEXTU MUSIM NASTAVIT FOLDER RALLM, SAFEGUARD,..... NEBO NIC 
            with open(script_path, "w", encoding="utf-8") as fh:
                fh.write(results_eval_all(oll_model))
            os.chmod(script_path, 0o755)
            # created.append(script_path)
            print(f"[INFO] Created {script_path}")

            if SUBMIT:
                try:
                    res = subprocess.run(["qsub", script_path], check=True, capture_output=True, text=True)
                    print(f"  -> qsub: {res.stdout.strip()}")
                except subprocess.CalledProcessError as e:
                    print(f"  !! qsub error: {e.stderr.strip()}")

        return
    
    if args.only_attack_batch:

        OLLAMA_MODELS = [ "falcon3:3b", "falcon3:7b", "falcon3:10b",
        "gemma3:1b", "gemma3:4b", "gemma3:12b", "gemma3:27b",
        "internlm2:1m", "internlm2:20b", "internlm2.5:latest",
        "llama2:7b", "llama2:13b",
        "llama3.1:8b", "llama3.2:1b", "llama3.2:3b",
        "qwen2.5:3b", "qwen2.5:7b", "qwen2.5:14b", "qwen2.5:32b",
        "qwen3:0.6b", "qwen3:14b", "qwen3:30b", "qwen3:32b",
        "yi:6b", "yi:9b", "yi:34b",
        "command-r:35b", "command-r:latest",
        "deepseek-r1:32b","phi3:8b", "phi3:14b", "phi4:14b", "mixtral:7b"
        ]


        OLLAMA_TMP = ["yi:6b", "yi:9b", "yi:34b",
        "command-r:35b", "command-r:latest",
        "deepseek-r1:32b","phi3:8b", "phi3:14b", "phi4:14b", "mixtral:7b"]

        OLLAMA_MODELS_first = [ "falcon3:3b", "falcon3:7b", "falcon3:10b",
        "gemma3:1b", "gemma3:4b", "gemma3:12b", "gemma3:27b",
        "internlm2:1m", "internlm2:20b", "internlm2.5:latest",
        "llama2:7b", "llama2:13b"]

        OLLAMA_MODELS_second = [
        "llama3.1:8b", "llama3.2:1b", "llama3.2:3b",
        "qwen2.5:3b", "qwen2.5:7b", "qwen2.5:14b", "qwen2.5:32b",
        "qwen3:0.6b", "qwen3:14b", "qwen3:30b", "qwen3:32b"]
        
        OLLAMA_MODELS_third = [
        "yi:6b", "yi:9b", "yi:34b",
        "command-r:35b", "command-r:latest",
        "deepseek-r1:32b","phi3:8b", "phi3:14b", "phi4:14b", "mixtral:7b"
        ]

        OLLAMA_TEST = ["gemma3:14b", "qwen3:14b"]

        for model in OLLAMA_MODELS:

            per_victim_llm, per_results_dir, per_ollama_model = build_per_model_config(cfg, model)
            ensure_dir(per_results_dir)
            per_jobs_dir = os.path.join(per_results_dir, "jobs")
            ensure_dir(per_jobs_dir)

            print(f"[ONLY-ATTACK-BATCH] === Model: {model} ===")
            print(f"[ONLY-ATTACK-BATCH] results_dir: {per_results_dir}")
            print(f"[ONLY-ATTACK-BATCH] victim_llm : {per_victim_llm}")
            # return 
            if not dataset_to_attack_dir or not os.path.isdir(dataset_to_attack_dir):
                print(f"[ONLY-ATTACK-BATCH] cfg['dataset_to_attack_dir'] neexistuje nebo není složka: {dataset_to_attack_dir}")
                return

            ensure_dir(results_dir)

            base_dir = os.path.dirname(os.path.abspath(__file__))
            # očekáváme batch verzi skriptu:
    

            inputs = []
            for entry in sorted(os.listdir(dataset_to_attack_dir)):
                if entry.lower().endswith(".json"):
                    # přeskočit podsložku jobs, kdyby tam byla
                    if entry == "jobs":
                        continue
                    inputs.append(os.path.join(dataset_to_attack_dir, entry))

            if not inputs:
                print(f"[ONLY-ATTACK-BATCH] Žádné .json soubory v {dataset_to_attack_dir}")
                return

            created = []
            for inp in inputs:
                base = os.path.basename(inp)   # např. _1_cypher.json
                name_stem = os.path.splitext(base)[0]
                job_name = f"onlyattackbatch_{safe_filename(name_stem)}"

            
                cmd = f"python3 only_attack_batch.py {per_victim_llm} {inp} {per_results_dir} {use_ollama} {per_ollama_model}"

                script_path = os.path.join(per_jobs_dir, f"job_{job_name}.sh")
                with open(script_path, "w", encoding="utf-8") as fh:
                    fh.write(job_batch_script_content(job_name, cmd, per_ollama_model))
                os.chmod(script_path, 0o755)
                created.append(script_path)
                print(f"[ONLY-ATTACK-BATCH] Created {script_path}")

                if SUBMIT and not DRY_RUN:
                    try:
                        res = subprocess.run(["qsub", script_path], check=True, capture_output=True, text=True)
                        print(f"  -> qsub: {res.stdout.strip()}")
                    except subprocess.CalledProcessError as e:
                        print(f"  !! qsub error: {e.stderr.strip()}")

            print(f"[ONLY-ATTACK-BATCH] Hotovo. Vytvořeno jobů: {len(created)}")
        return
    
    if args.only_attack:
        if not dataset_to_attack_dir or not os.path.isdir(dataset_to_attack_dir):
            print(f"[ONLY-ATTACK] cfg['dataset_to_attack_dir'] neexistuje nebo není složka: {dataset_to_attack_dir}")
            return

        ensure_dir(results_dir)

        # cesta k only_attack.py (předpokládáme, že ho máš ve stejném adresáři)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        only_attack_py = os.path.join(base_dir, "only_attack.py")
        if not os.path.exists(only_attack_py):
            print(f"[ONLY-ATTACK] WARNING: {only_attack_py} nenalezen. Uprav cestu nebo vytvoř skript.")
            # i kdyz chybí, můžeme pokračovat a vytvořit joby, které volají non-existing skript
            # return

        inputs = []
        for entry in sorted(os.listdir(dataset_to_attack_dir)):
            if entry.lower().endswith(".json"):
                # preskoc podslozku jobs pokud tam je
                if entry == "jobs":
                    continue
                inputs.append(os.path.join(dataset_to_attack_dir, entry))

        if not inputs:
            print(f"[ONLY-ATTACK] Žádné .json soubory v {dataset_to_attack_dir}")
            return

        created = []
        for inp in inputs:
            base = os.path.basename(inp)   # např. _1_cypher.json
            name_stem = os.path.splitext(base)[0]
            job_name = f"onlyattack_{safe_filename(name_stem)}"

            # příkaz: voláme only_attack.py pro TEN KONKRÉTNÍ soubor
            # only_attack.py má mít signaturu:
            #   python3 only_attack.py <victim_llm_path> <input_json_file> <output_dir> <api_ollama_vllm> <what_ollama_model>
            cmd = f"python3 {only_attack_py} {victim_llm} {inp} {results_dir} {use_ollama} {ollama_model}"

            script_path = os.path.join(jobs_dir, f"job_{job_name}.sh")
            with open(script_path, "w", encoding="utf-8") as fh:
                fh.write(job_script_content(job_name, cmd, ollama_model))
            os.chmod(script_path, 0o755)
            created.append(script_path)
            print(f"[ONLY-ATTACK] Created {script_path}")

            if SUBMIT and not DRY_RUN:
                try:
                    res = subprocess.run(["qsub", script_path], check=True, capture_output=True, text=True)
                    print(f"  -> qsub: {res.stdout.strip()}")
                except subprocess.CalledProcessError as e:
                    print(f"  !! qsub error: {e.stderr.strip()}")

        print(f"[ONLY-ATTACK] Hotovo. Vytvořeno jobů: {len(created)}")
        return

    if args.fix:
            # dataset_to_attack_dir
        # 1) Spusť diagnostický skript a vyparsuj cesty k souborům
        script_path = Path("/storage/brno2/home/xkaska01/master/my_implementation/supporting_scripts/is_empty.py").resolve()
        if not script_path.exists():
            print(f"[FIX] ❌ Nenalezen skript: {script_path}")
            return

        print(f"[FIX] Spouštím: python3 {script_path}")
        try:
            proc = subprocess.run(["python3", str(script_path)], capture_output=True, text=True, check=True)
            lines = proc.stdout.splitlines()
        except subprocess.CalledProcessError as e:
            print(f"[FIX] ❌ Chyba při běhu {script_path}: {e.stderr}")
            return

        # Zachytáváme řádky začínající '- ' a končící '.json'
        json_paths: list[Path] = []
        for line in lines:
            s = line.strip()
            if s.startswith("- "):
                p = s[2:].strip()
                if p.endswith(".json"):
                    json_paths.append(Path(p))

        # print()
        # print()
        # print()
        # print()

        # print(json_paths)
        # print()
        # print()
        # print()
        # print()

        # exit()
        if not json_paths:
            print("[FIX] ✅ Nenašel jsem nic k opravě.")
            return

        print(f"[FIX] Nalezeno problémových souborů: {len(json_paths)}")
        attack_input_map: dict[str, Path] = {}

        dset_dir = Path(dataset_to_attack_dir)
        if not dset_dir.is_dir():
            print(f"[FIX] ❌ dataset_to_attack_dir není adresář: {dset_dir}")
            return

        for entry in sorted(dset_dir.iterdir()):
            if not entry.is_file():
                continue
            if entry.suffix.lower() != ".json":
                continue
            if entry.name == "jobs":
                continue

            stem = entry.stem.lower()            # např. "_11_pair" nebo "_6_sequential_attack"
            key = normalize_json_stem(stem)      # např. "_11_pair" nebo "_6_sequential"
            attack_input_map[key] = entry    
        # Mapa útok -> modul (používáme pro parsování názvů útoků - zachováváme stávající funkci)

        # print(attack_input_map)
        # exit()
        attack_map = map_attack_stem_to_module()

        # připravíme adresář pro joby - fixy
        fix_dir = Path(jobs_dir) / "fixes"
        fix_dir.mkdir(parents=True, exist_ok=True)

        # cesta k only_attack.py (použijeme stejnou, jako v --only-attack)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        only_attack_py = os.path.join(base_dir, "only_attack.py")
        if not os.path.exists(only_attack_py):
            print(f"[FIX] WARNING: {only_attack_py} nenalezen. Joby budou vytvořeny, ale spustí se neexistující skript.")
            # necháme to pokračovat (stejně jako v --only-attack)

        submitted = 0
        skipped = 0
        not_intended_model = 0

        for json_path in json_paths:
            attack_stem = json_path.stem.lower()       # např. "_16_ica"
            module = find_module_for_stem(attack_stem, attack_map)
            if not module:
                print(f"[FIX] ⚠️  Neznámý útok pro '{attack_stem}', přeskočeno: {json_path}")
                skipped += 1
                continue

            norm_key = normalize_json_stem(attack_stem)   # např. "_11_pair" nebo "_6_sequential"
            attack_in_path = attack_input_map.get(norm_key)

            if not attack_in_path:
                print(f"[FIX] ⚠️  Nenašel jsem vstupní attack JSON pro '{json_path.name}' (klíč '{norm_key}'), přeskočeno.")
                skipped += 1
                continue

            model_from_path = parse_model_from_path(json_path)
            if not model_from_path:
                print(f"[FIX] ⚠️  Nepodařilo se vyparsovat model z cesty, přeskočeno: {json_path}")
                skipped += 1
                continue


            # if model_from_path != "qwen2.5:14b":
            #     print(f"[FIX] ⚠️  Model z cesty '{model_from_path}' neodpovídá požadovanému '{ollama_model}', přeskočeno: {json_path}")
            #     not_intended_model += 1
            #     continue
            # --- ZDE JE ZÁSADNÍ ZMĚNA ---
            # místo spuštění plného modulu útoku vytvoříme job, který spustí only_attack.py
            # tj. pouze inference nad konkrétním JSONem a MODELEM parsed z cesty

            per_job_out_dir = str(json_path.parent)
            per_job_ollama_model = model_from_path
            # cmd stejný formát jako v --only-attack režimu:
            # python3 only_attack.py <victim_llm> <input_json_file> <output_dir> <api_ollama_vllm> <what_ollama_model>
            cmd = f"python3 {only_attack_py} {victim_llm} {attack_in_path} {per_job_out_dir} {use_ollama} {per_job_ollama_model}"

            # safe názvy pro job filename / qsuby
            safe_mod = module.replace("attacks.", "").replace(".main", "").replace(".", "_")
            safe_model = safe_filename(per_job_ollama_model)
            name = f"fix_onlyattack_{safe_mod}_{safe_model}"

            script_path = fix_dir / f"job_{name}.sh"
            with open(script_path, "w", encoding="utf-8") as fh:
                fh.write(job_script_content(name, cmd, per_job_ollama_model))
            os.chmod(script_path, 0o755)
            print(f"[FIX] Created {script_path}")

            if SUBMIT and not DRY_RUN:
                try:
                    res = subprocess.run(["qsub", str(script_path)], check=True, capture_output=True, text=True)
                    print(f"[FIX]  -> qsub: {res.stdout.strip()}")
                    submitted += 1
                except subprocess.CalledProcessError as e:
                    print(f"[FIX]  !! qsub error: {e.stderr.strip()}")
                    skipped += 1
            else:
                print(f"[FIX]  (dry-run) would run: qsub {script_path}")
                print(f"[FIX]  (dry-run) job command: {cmd}")

        print(f"[FIX] Hotovo. Odesláno: {submitted}, přeskočeno: {skipped} a preskoceno modelu, ktere nejsou pozadovany: {not_intended_model}")
        return


    created = []
    for module in ATTACK_MODULES:
        name = module.split(".")[1]  # např. "_1_Cypher" → můžeš si to přejmenovat, klidně zjemni:
        safe_name = module.replace("attacks.", "").replace(".main", "").replace(".", "_")
        # příkaz: python3 -m attacks.<...>.main <victim_llm> <results_dir> <dataset> <use_ollama> <ollama_model>
        cmd = f"python3 -m {module} {victim_llm} {results_dir} {dataset} {use_ollama} {ollama_model}"

        script_path = os.path.join(jobs_dir, f"job_{safe_name}.sh")
        with open(script_path, "w", encoding="utf-8") as fh:
            fh.write(job_script_content(safe_name, cmd, ollama_model))
        os.chmod(script_path, 0o755)
        created.append(script_path)
        print(f"[INFO] Created {script_path}")

        if SUBMIT:
            try:
                res = subprocess.run(["qsub", script_path], check=True, capture_output=True, text=True)
                print(f"  -> qsub: {res.stdout.strip()}")
            except subprocess.CalledProcessError as e:
                print(f"  !! qsub error: {e.stderr.strip()}")

    print(f"[INFO] Hotovo. Vytvořeno jobů: {len(created)}")

if __name__ == "__main__":
    main()
