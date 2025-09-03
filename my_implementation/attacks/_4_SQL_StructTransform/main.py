# TENTO FILE JE MUUUUUUUUUUJ

import os, json, yaml, pandas as pd
from tqdm import tqdm
import gc
import torch

from attacks.common.llm import LLM
from attacks._4_SQL_StructTransform.SQL_attack import SQLAttack
from defense.defense_EA import DefenseEA 
from attacks.helpers import load_config 


# ---------- PIPELINE -------------------------------------------
def run_sql_attack(victim_llm_path, results_dir, dataset_path, api_ollama_vllm, what_ollama_model):

    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "configStruct.yaml")
    cfg = load_config(config_path)
    cfg = cfg["SQL"]

    # print(cfg)
    # exit(0)

    # LLM instance – M_attack
    attacker = LLM(
        model_path  = cfg['attacker_llm'],
        temperature = cfg['temperature_attack'],
        max_tokens  = cfg['max_tokens'],
        ollama_model="qwen2.5:7b",
        use_ollama=True, 
    )
    # LLM instance – M_target
    target = LLM(
        model_path  = victim_llm_path,
        temperature = cfg['temperature_target'],
        max_tokens  = cfg['max_tokens'],
        ollama_model=what_ollama_model,
        use_ollama=api_ollama_vllm, 
    )
    sql_attack = SQLAttack(
        attacker_llm = attacker, # zde ma patrit attacker_llm, ale zatim jsem nechal stejne na utok tak i na vyhodnoceni protoze se to nevejde na disk
        few_shot     = cfg['few_shot'],
        num_attempts = cfg['num_attempts']
    )
    defense = DefenseEA()

    df = pd.read_csv(dataset_path)
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "_4_sql.json")
    entries = []
    with open(out_path, "w", encoding="utf-8") as fo:
        for idx, goal in tqdm(
                enumerate(df["goal"][cfg["begin"]:cfg["end"]]),
                total=cfg["end"]-cfg["begin"]):
            log, prompts = sql_attack.generate(goal)

            response = target.response(prompts)
            print("[INFO] Response:", response)

            entry = {
                'id': idx,
                'original_prompt': goal,
                'prompt': prompts[-1]["content"],
                'response': response
            }
            entries.append(entry)

            # až po cyklu:
        json.dump(entries, fo, ensure_ascii=False, indent=2)

    print(f"[INFO] DirectSQL finished ➜ {out_path}")
