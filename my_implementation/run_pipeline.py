# run.py

import os, time, json, argparse, yaml
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
from attacks._23_Cold.main import run_cold_attack
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
    ("flip", run_flip_attack), #funguje
    ("pif", run_pif_attack), # FUNGUJE ALE OPATRENE S NIM - POUZIVA KNIHOVNU TRANSFORMERS, TEDY KONTROLUJ VELIKOST MODELU !!!
    ("sql", run_sql_attack), 
    ("suffix", run_suffix_attack),
    ("sequential", run_sequential_attack),
    ("cite", run_cite_attack),
    ("bijection", run_bijection_attack),
    ("dialog", run_dialog_attack),
    ("random", run_random_attack),
    ("pair", run_pair_attack),
    ("tap", run_tap_attack), 
    ("gptcypher", run_GPTcypher_attack),
    ("MultiLang", run_Multilang_attack),
    ("rewrite", run_rewrite_attack), 
    ("ica", run_ica_attack),
    ("overload", run_overload_attack), #funguje
    ("gcg", run_gcg_attack),
    ("inception", run_inception_attack),
    ("base", run_base_attack), #funguje
    ("artprompt", run_artprompt_attack), #funguje
    ("renellm", run_renellm_attack), #funguje
    ("autodan",    run_autodan_attack), #funguje
    ("past_tense", run_past_tense_attack), #funguje
    ("chameleon", run_chameleon_attack),
]

special_runs = [
    # ("pair",       run_pair_attack), 

    ("tap", run_tap_attack), 
]
test = [
    # ("gcg", run_gcg_attack), # funguje zatim jen na male modely, llama 7b jsem vyzkousel, protoze vetsi modely se mi nevejdou do pameti
    ("overload", run_overload_attack),
    

    # ("cold", run_cold_attack) # zatim nefunguje a mozna misto nej najdu nahradu .....
]

all_attack_categories = [
    # works,
    # special_runs,
    test,
]



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


    print(f"[INFO] Victim LLM Path: {victim_llm_path}"
          f"\n[INFO] Results Directory: {results_dir}")
    
    print(f"[INFO] Running FlipAttack with config: {cfg.get('run_defense', False)}")

    os.makedirs(results_dir, exist_ok=True)
    log_file_path = os.path.join(results_dir, "log_runtime.txt")
    log_fh = open(log_file_path, "w", encoding="utf-8")

    timings = {}
    total_start = time.perf_counter()

    for category in all_attack_categories:
        for name, fn in category:
            log_and_print(f"\n[INFO] ➜ Spouštím {name}…", log_fh)
            t0 = time.perf_counter()
            try:
                fn(victim_llm_path, results_dir, dataset_path,
                   api_ollama_vllm, what_ollama_model)
            except Exception as e:
                log_and_print(f"[ERROR] {name} selhal: {e}", log_fh)
                continue
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