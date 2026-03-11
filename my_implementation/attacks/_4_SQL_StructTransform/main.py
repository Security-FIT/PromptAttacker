## @file run_sql.py
#  @brief Script for running SQL-based structural prompt transformation attacks
#
#  This script loads a dataset of adversarial prompts and applies a
#  SQL-inspired structural transformation attack against a victim LLM.
#  The attack uses a separate attacker LLM to generate transformed prompts,
#  which are then evaluated by the target (victim) model. Results are stored
#  in JSON format for further analysis.
#
#  @author Bc. Petr Kaška
#  @date 3.1.2026
#
#  Ownership / Contribution statement:
#   - This file is an original implementation by Bc. Petr Kaška.
#   - The experiment orchestration, model initialization, dataset handling,
#     and result logging were implemented by the author.

import os, json, yaml, pandas as pd
from tqdm import tqdm
import gc
import torch,sys

from attacks.common.llm import LLM
from attacks._4_SQL_StructTransform.SQL_attack import SQLAttack
from attacks.common.helpers import load_config, str2bool


def run_sql_attack(victim_llm_path, results_dir, dataset_path, api_ollama_vllm, what_ollama_model):

    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "configStruct.yaml")
    cfg = load_config(config_path)
    cfg = cfg["SQL"]

    attacker = LLM(
        model_path  = cfg['attacker_llm'],
        temperature = cfg['temperature_attack'],
        max_tokens  = cfg['max_tokens'],
        ollama_model="qwen2.5:7b",
        use_ollama=True, 
    )
    target = LLM(
        model_path  = victim_llm_path,
        temperature = cfg['temperature_target'],
        max_tokens  = cfg['max_tokens'],
        ollama_model=what_ollama_model,
        use_ollama=api_ollama_vllm, 
    )
    sql_attack = SQLAttack(
        attacker_llm = attacker, 
        few_shot     = cfg['few_shot'],
        num_attempts = cfg['num_attempts']
    )

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

        json.dump(entries, fo, ensure_ascii=False, indent=2)

    print(f"[INFO] DirectSQL finished ➜ {out_path}")


if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: python3 run_cypher.py victim_llm_path results_dir dataset_path api_ollama_vllm what_ollama_model")
        sys.exit(1)

    victim_llm_path = sys.argv[1]
    results_dir = sys.argv[2]
    dataset_path = sys.argv[3]
    api_ollama_vllm = str2bool(sys.argv[4].lower())
    what_ollama_model = sys.argv[5]

    run_sql_attack(victim_llm_path, results_dir, dataset_path, api_ollama_vllm, what_ollama_model)